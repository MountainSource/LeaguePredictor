"""
model_trainer.py
-----------------
Per-league model training pipeline.

Trains two models for each league:
  1. Dixon-Coles statistical model (attack/defence parameters per team)
  2. Market-Augmented ML model (rolling features + DC probs + Shin fair probs)

The Market-Augmented ML model was confirmed as the best performer in Phase 6
(Brier 0.5814 vs Dixon-Coles baseline 0.5914 on EPL walk-forward evaluation).

TRAINING DATA REQUIREMENTS:
  Minimum 1 season to fit DC model only.
  Minimum 2 seasons to fit the ML model (1 train, 1 calibration for isotonic).
  Recommended: 4+ seasons for reliable walk-forward performance.

SHOT DATA:
  All 6 supported leagues have shot data (HS/AS/HST/AST). The full feature
  set including shot volume and quality features is used for all leagues.
  If shot columns are missing in a specific CSV, those features become NaN
  and are handled natively by HistGradientBoostingClassifier.

LOOK-AHEAD PROTECTION:
  All rolling features use .shift(1) before the window, so feature for
  match N uses only data from matches 1..N-1. No future data ever enters
  a feature vector.

MODEL ARTIFACTS (saved to models/{DIVCODE}_*.json / *.pkl):
  {div}_dc_params.json   — Dixon-Coles fitted parameters
  {div}_ml_model.pkl     — Pickled CalibratedHistGBC (Market-Augmented)
  {div}_feature_db.pkl   — Latest rolling features per team (for live prediction)
  {div}_metadata.json    — Training summary (match count, timestamp, Brier)
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings('ignore')

# Sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Dixon-Coles
DECAY_HALF_LIFE_DAYS    = 133       # ~19 match half-life for EPL (literature-informed)
POISSON_MAX_GOALS       = 10        # Upper bound for Poisson convolution
ENTRANT_REG_STRENGTH    = 5.0       # L2 shrinkage weight for newly promoted teams
MIN_MATCHES_FOR_PARAMS  = 5         # Minimum matches before a team gets own parameters

# Feature engineering
ROLLING_WINDOWS         = [3, 6, 9, 12]
H2H_DECAY               = 0.85      # Per-meeting decay for H2H recency weighting
MIN_H2H_MEETINGS        = 5         # Minimum meetings before H2H features activate
MIN_ROLLING_MATCHES     = 3         # Below this, rolling features remain NaN

# Default tuned window config (from Phase 6 EPL tuning)
DEFAULT_WINDOW_CONFIG = {
    'shots': 9, 'sot': 9, 'hst_pct': 9,
    'goals_for': 9, 'goals_ag': 6,
    'win_rate': 9, 'draw_rate': 6,
    'fouls': 9, 'yellows': 9, 'reds': 9, 'corners': 9,
}

# Default ML hyperparameters (from Phase 6 tuning — favour regularisation)
DEFAULT_HPARAMS = {
    'max_iter': 200, 'max_depth': 3,
    'learning_rate': 0.01, 'min_samples_leaf': 50,
    'l2_regularization': 0.1,
}

# Shin normalisation convergence
SHIN_MAX_ITER = 1000
SHIN_TOL      = 1e-9

# Value bet edge threshold
DEFAULT_EDGE_THRESHOLD = 0.03


# ===========================================================================
# SECTION 1: DATA LOADING
# ===========================================================================

def load_league_data(data_dir: str, league_config, verbose: bool = True) -> pd.DataFrame:
    """
    Load and normalise all CSV files in data_dir for a given league.

    Returns a unified DataFrame with canonical column names, sorted by date.
    Handles schema variations (e.g. EPL 2017/18 BetBrain columns).
    """
    csv_files = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith('.csv')
    ])

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    frames = []
    for fname in csv_files:
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df = _normalise_df(df, league_config)
            frames.append(df)
            if verbose:
                print(f"  Loaded {fname}: {len(df)} rows")
        except Exception as e:
            if verbose:
                print(f"  WARNING: Could not load {fname}: {e}")
            continue

    if not frames:
        raise ValueError(f"No valid CSV files loaded from {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'], dayfirst=True, errors='coerce')
    combined = combined.dropna(subset=['date', 'fthg', 'ftag'])
    combined = combined.sort_values('date').reset_index(drop=True)

    # Namespace team names: EPL::Arsenal etc.
    ns = league_config.namespace
    combined['home_team'] = ns + '::' + combined['home_team'].str.strip()
    combined['away_team'] = ns + '::' + combined['away_team'].str.strip()

    # has_crowd flag
    combined['season'] = combined['date'].apply(_infer_season)
    combined['has_crowd'] = ~combined['season'].isin(league_config.no_crowd_seasons)

    # Deduplicate (same match in multiple season files)
    combined = combined.drop_duplicates(
        subset=['date', 'home_team', 'away_team']
    ).reset_index(drop=True)

    if verbose:
        print(f"  Total: {len(combined)} unique matches across {combined['season'].nunique()} seasons")

    return combined


def _normalise_df(df: pd.DataFrame, league_config) -> pd.DataFrame:
    """Map raw CSV columns to canonical names for a single season file."""
    try:
        from .league_configs import STANDARD_COLUMNS
    except ImportError:
        from league_configs import STANDARD_COLUMNS

    # Detect season from data to look up any overrides
    if 'Date' in df.columns:
        sample_dates = pd.to_datetime(df['Date'].dropna().head(20), dayfirst=True, errors='coerce')
        sample_dates = sample_dates.dropna()
        if len(sample_dates) > 0:
            year = sample_dates.iloc[len(sample_dates) // 2].year
            month = sample_dates.iloc[len(sample_dates) // 2].month
            # Football seasons start Aug, so a Jan 2018 match is 2017/18
            if month >= 8:
                season_label = f"{year}/{str(year + 1)[2:]}"
            else:
                season_label = f"{year - 1}/{str(year)[2:]}"
        else:
            season_label = None
    else:
        season_label = None

    # Build column map: start with standard, apply season overrides if present
    col_map = dict(STANDARD_COLUMNS)
    if season_label and season_label in league_config.season_column_overrides:
        col_map.update(league_config.season_column_overrides[season_label])

    # Rename: only map columns that exist in this CSV
    rename = {}
    for canonical, raw in col_map.items():
        if raw in df.columns:
            rename[raw] = canonical

    df = df.rename(columns=rename)

    # Ensure all expected canonical columns exist (fill with NaN if absent)
    for canonical in col_map.keys():
        if canonical not in df.columns:
            df[canonical] = np.nan

    # Resolve odds priority: AvgH preferred, B365 fallback
    for outcome in ['h', 'd', 'a']:
        avg_col = f'avg_{outcome}'
        b365_col = f'b365_{outcome}'
        if df[avg_col].isna().all() and not df[b365_col].isna().all():
            df[avg_col] = df[b365_col]

    return df


def _infer_season(date: pd.Timestamp) -> str:
    """Infer 'YYYY/YY' season label from a date."""
    if pd.isna(date):
        return 'Unknown'
    if date.month >= 8:
        return f"{date.year}/{str(date.year + 1)[2:]}"
    else:
        return f"{date.year - 1}/{str(date.year)[2:]}"


# ===========================================================================
# SECTION 2: DIXON-COLES MODEL
# ===========================================================================

def _dc_rho_correction(goals_h: int, goals_a: int, lam_h: float,
                        lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor τ for goals ≤ 1."""
    if goals_h == 0 and goals_a == 0:
        return 1.0 - lam_h * lam_a * rho
    elif goals_h == 0 and goals_a == 1:
        return 1.0 + lam_h * rho
    elif goals_h == 1 and goals_a == 0:
        return 1.0 + lam_a * rho
    elif goals_h == 1 and goals_a == 1:
        return 1.0 - rho
    return 1.0


def _dc_log_likelihood(params: np.ndarray, matches: pd.DataFrame,
                        teams: List[str], weights: np.ndarray) -> float:
    """
    Negative log-likelihood for Dixon-Coles model (to be minimised).

    params layout:
      [0 .. n_teams-1]         : log(alpha_i) — attack parameters
      [n_teams .. 2*n_teams-1] : log(beta_i)  — defence parameters
      [2*n_teams]              : log(gamma)   — home advantage
      [2*n_teams + 1]          : rho          — low-score correction
    """
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    log_alpha = params[:n]
    log_beta  = params[n:2*n]
    log_gamma = params[2*n]
    rho       = params[2*n + 1]

    alpha = np.exp(log_alpha)
    beta  = np.exp(log_beta)
    gamma = np.exp(log_gamma)

    total_ll = 0.0
    for i, row in enumerate(matches.itertuples()):
        hi = idx.get(row.home_team)
        ai = idx.get(row.away_team)
        if hi is None or ai is None:
            continue

        lam_h = gamma * alpha[hi] * beta[ai]
        lam_a = alpha[ai] * beta[hi]

        gh = int(row.fthg)
        ga = int(row.ftag)

        ll = (
            -lam_h + gh * np.log(max(lam_h, 1e-10))
            - lam_a + ga * np.log(max(lam_a, 1e-10))
            + np.log(max(abs(_dc_rho_correction(gh, ga, lam_h, lam_a, rho)), 1e-10))
        )
        total_ll += weights[i] * ll

    return -total_ll


def fit_dc_model(matches: pd.DataFrame,
                 exclude_no_crowd: bool = True,
                 new_entrant_teams: Optional[List[str]] = None) -> Dict:
    """
    Fit Dixon-Coles model on historical matches.

    Returns a dict of fitted parameters suitable for JSON serialisation
    and subsequent prediction.

    Parameters
    ----------
    matches : DataFrame with columns home_team, away_team, fthg, ftag, date, has_crowd
    exclude_no_crowd : if True, exclude COVID/behind-closed-doors matches from fitting
    new_entrant_teams : teams to apply L2 shrinkage to (newly promoted)
    """
    if exclude_no_crowd and 'has_crowd' in matches.columns:
        fit_df = matches[matches['has_crowd']].copy()
    else:
        fit_df = matches.copy()

    fit_df = fit_df.dropna(subset=['fthg', 'ftag']).copy()
    fit_df['fthg'] = fit_df['fthg'].astype(int)
    fit_df['ftag'] = fit_df['ftag'].astype(int)

    # Time-decay weights: exponential, half-life = DECAY_HALF_LIFE_DAYS
    latest_date = fit_df['date'].max()
    days_ago = (latest_date - fit_df['date']).dt.days.values
    weights = np.exp(-np.log(2) * days_ago / DECAY_HALF_LIFE_DAYS)

    teams = sorted(set(fit_df['home_team']) | set(fit_df['away_team']))
    n = len(teams)

    if n < 2:
        raise ValueError("Need at least 2 teams to fit DC model")

    # Initial parameters: all 1s in parameter space (0 in log space)
    x0 = np.zeros(2 * n + 2)
    x0[2*n] = np.log(1.2)   # gamma ~ 1.2 home advantage prior
    x0[2*n + 1] = -0.01     # rho small negative

    # Identifiability constraint: sum of log_alpha = 0
    # Implemented as equality constraint via L-BFGS-B and post-fit re-centring
    def neg_ll(params):
        ll = _dc_log_likelihood(params, fit_df, teams, weights)
        # L2 regularisation for new entrants
        if new_entrant_teams:
            idx = {t: i for i, t in enumerate(teams)}
            reg = 0.0
            for team in new_entrant_teams:
                i = idx.get(team)
                if i is not None:
                    obs = float((fit_df['home_team'] == team).sum() +
                                (fit_df['away_team'] == team).sum())
                    strength = ENTRANT_REG_STRENGTH / max(obs, 1.0)
                    reg += strength * (params[i]**2 + params[n + i]**2)
            ll += reg
        return ll

    bounds = (
        [(-3, 3)] * n +       # log_alpha bounds
        [(-3, 3)] * n +       # log_beta bounds
        [(np.log(0.8), np.log(2.0))] +  # log_gamma
        [(-0.5, 0.2)]         # rho
    )

    result = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 10000, 'maxfun': 200000, 'ftol': 1e-10, 'gtol': 1e-6})

    params = result.x
    n_teams = n

    # Post-fit identifiability: re-centre log_alpha so geometric mean = 1
    log_alpha = params[:n_teams]
    log_beta  = params[n_teams:2*n_teams]
    shift     = log_alpha.mean()
    log_alpha -= shift
    log_beta  += shift   # compensate so lambda values are preserved

    alpha = np.exp(log_alpha)
    beta  = np.exp(log_beta)
    gamma = np.exp(params[2*n_teams])
    rho   = params[2*n_teams + 1]

    return {
        'teams': teams,
        'alpha': dict(zip(teams, alpha.tolist())),
        'beta':  dict(zip(teams, beta.tolist())),
        'gamma': float(gamma),
        'rho':   float(rho),
        'converged': bool(result.success),
        'n_matches_fitted': int(len(fit_df)),
        'fit_date': datetime.now().isoformat(),
    }


def dc_predict_proba(home_team: str, away_team: str,
                     dc_params: Dict) -> Optional[np.ndarray]:
    """
    Predict H/D/A probabilities for a single match using Dixon-Coles.

    Returns np.array([p_home, p_draw, p_away]) or None if teams unknown.
    """
    alpha = dc_params['alpha']
    beta  = dc_params['beta']
    gamma = dc_params['gamma']
    rho   = dc_params['rho']

    if home_team not in alpha or away_team not in alpha:
        return None

    lam_h = gamma * alpha[home_team] * beta[away_team]
    lam_a = alpha[away_team] * beta[home_team]

    max_g = POISSON_MAX_GOALS
    prob_matrix = np.zeros((max_g + 1, max_g + 1))

    for gh in range(max_g + 1):
        for ga in range(max_g + 1):
            p = (poisson.pmf(gh, lam_h) * poisson.pmf(ga, lam_a) *
                 _dc_rho_correction(gh, ga, lam_h, lam_a, rho))
            prob_matrix[gh, ga] = max(p, 0)

    # Normalise
    prob_matrix /= prob_matrix.sum()

    p_home = float(np.sum(np.tril(prob_matrix, -1)))   # gh > ga
    p_draw = float(np.trace(prob_matrix))               # gh == ga
    p_away = float(np.sum(np.triu(prob_matrix, 1)))     # gh < ga

    return np.array([p_home, p_draw, p_away])


# ===========================================================================
# SECTION 3: SHIN NORMALISATION
# ===========================================================================

def shin_normalise(odds_h: float, odds_d: float, odds_a: float
                   ) -> Tuple[float, float, float]:
    """
    Apply Shin (1993) iterative algorithm to remove bookmaker overround.

    Returns fair probabilities (p_h, p_d, p_a) that sum to 1.0.
    The Shin method accounts for the asymmetric impact of informed bettors,
    typically lowering favourite implied probs and raising draw implied probs
    relative to simple proportional normalisation.
    """
    raw = np.array([1.0 / odds_h, 1.0 / odds_d, 1.0 / odds_a])
    total = raw.sum()

    if abs(total - 1.0) < 1e-8:
        return float(raw[0]), float(raw[1]), float(raw[2])

    # Newton's method to find z (fraction of bets from informed traders)
    z = 0.0
    for _ in range(SHIN_MAX_ITER):
        denom = np.sqrt(z**2 + 4 * (1 - z) * raw / total)
        p = (denom - z) / (2 * (1 - z))
        p /= p.sum()
        new_z = (total - 1) / (total - 1 + np.sum(raw * (1 - raw / total) /
                 (p * total + z * (1 - 2 * p))))
        if abs(new_z - z) < SHIN_TOL:
            break
        z = new_z

    denom = np.sqrt(z**2 + 4 * (1 - z) * raw / total)
    p = (denom - z) / (2 * (1 - z))
    p = np.clip(p, 0, 1)
    p /= p.sum()

    return float(p[0]), float(p[1]), float(p[2])


# ===========================================================================
# SECTION 4: FEATURE ENGINEERING
# ===========================================================================

def build_feature_matrix(matches: pd.DataFrame,
                          window_config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Build a match-level feature matrix from historical matches.

    Features per match:
      - Rolling team stats (both home and away perspectives): shots, sot,
        hst_pct, goals_for, goals_ag, win_rate, draw_rate, fouls, yellows,
        reds, corners — at the per-stat optimal window from window_config
      - Differential features (home minus away) for each stat
      - H2H features (if MIN_H2H_MEETINGS prior meetings exist)

    All rolling stats use .shift(1) to prevent look-ahead.
    Returns one row per match with NaN where insufficient history.
    """
    if window_config is None:
        window_config = DEFAULT_WINDOW_CONFIG

    df = matches.copy().sort_values('date').reset_index(drop=True)
    df['match_idx'] = df.index

    # Build long "team-match" DataFrame: 2 rows per match
    long_rows = []
    for _, row in df.iterrows():
        # Compute outcome from home team perspective
        gh, ga = row.get('fthg', np.nan), row.get('ftag', np.nan)
        if pd.isna(gh) or pd.isna(ga):
            continue

        gh, ga = int(gh), int(ga)
        h_win  = 1 if gh > ga else 0
        a_win  = 1 if ga > gh else 0
        draw   = 1 if gh == ga else 0

        has_shots = not pd.isna(row.get('hs', np.nan))

        # Home team row
        long_rows.append({
            'match_idx':  row['match_idx'],
            'date':       row['date'],
            'team_id':    row['home_team'],
            'is_home':    1,
            'goals_for':  gh,
            'goals_ag':   ga,
            'win':        h_win,
            'draw':       draw,
            'shots':      row.get('hs', np.nan) if has_shots else np.nan,
            'sot':        row.get('hst', np.nan) if has_shots else np.nan,
            'hst_pct':    (row.get('hst', np.nan) / max(row.get('hs', 1), 1))
                          if has_shots and not pd.isna(row.get('hs')) and row.get('hs', 0) > 0
                          else np.nan,
            'fouls':      row.get('hf', np.nan),
            'yellows':    row.get('hy', np.nan),
            'reds':       row.get('hr', np.nan),
            'corners':    row.get('hc', np.nan),
        })
        # Away team row
        long_rows.append({
            'match_idx':  row['match_idx'],
            'date':       row['date'],
            'team_id':    row['away_team'],
            'is_home':    0,
            'goals_for':  ga,
            'goals_ag':   gh,
            'win':        a_win,
            'draw':       draw,
            'shots':      row.get('as_', np.nan) if has_shots else np.nan,
            'sot':        row.get('ast', np.nan) if has_shots else np.nan,
            'hst_pct':    (row.get('ast', np.nan) / max(row.get('as_', 1), 1))
                          if has_shots and not pd.isna(row.get('as_')) and row.get('as_', 0) > 0
                          else np.nan,
            'fouls':      row.get('af', np.nan),
            'yellows':    row.get('ay', np.nan),
            'reds':       row.get('ar', np.nan),
            'corners':    row.get('ac', np.nan),
        })

    long_df = pd.DataFrame(long_rows).sort_values(['team_id', 'date'])

    # Compute rolling stats per team (shift(1) ensures no look-ahead)
    STAT_COLS = ['goals_for', 'goals_ag', 'win', 'draw',
                 'shots', 'sot', 'hst_pct', 'fouls', 'yellows', 'reds', 'corners']

    rolling_frames = []
    for team_id, grp in long_df.groupby('team_id'):
        grp = grp.sort_values('date').copy()
        roll_data = {'match_idx': grp['match_idx'].values, 'team_id': team_id}

        for stat in STAT_COLS:
            w = window_config.get(stat, 9)
            shifted = grp[stat].shift(1)
            rolled  = shifted.rolling(window=w, min_periods=MIN_ROLLING_MATCHES).mean()
            roll_data[f'{stat}_w{w}'] = rolled.values

        rolling_frames.append(pd.DataFrame(roll_data))

    rolling_df = pd.concat(rolling_frames, ignore_index=True)

    # Pivot: merge home and away team rolling stats into match-level DataFrame
    home_stats = rolling_df.copy()
    home_stat_cols = {c: f'h_{c}' for c in rolling_df.columns
                      if c not in ('match_idx', 'team_id')}
    home_stats = home_stats.rename(columns=home_stat_cols)

    away_stats = rolling_df.copy()
    away_stat_cols = {c: f'a_{c}' for c in rolling_df.columns
                      if c not in ('match_idx', 'team_id')}
    away_stats = away_stats.rename(columns=away_stat_cols)

    # Join to df using match_idx + team_id
    df_feat = df.copy()

    # Map home team stats
    home_map = (home_stats
                .rename(columns={'h_team_id': 'home_team', 'match_idx': 'match_idx'})
                .set_index(['match_idx']))
    df_feat = df_feat.merge(
        home_stats.rename(columns={'h_team_id': 'home_team'}).drop(columns=['home_team'], errors='ignore'),
        left_on='match_idx', right_on='match_idx', how='left'
    )

    # This approach is cleaner: merge via match_idx for home and away separately
    df_feat = df[['match_idx', 'date', 'home_team', 'away_team',
                  'fthg', 'ftag', 'ftr', 'has_crowd', 'season',
                  'avg_h', 'avg_d', 'avg_a']].copy()

    home_roll = rolling_df.merge(
        df[['match_idx', 'home_team']].rename(columns={'home_team': 'team_id'}),
        on=['match_idx', 'team_id'], how='inner'
    ).rename(columns={c: f'h_{c}' for c in rolling_df.columns if c not in ('match_idx', 'team_id')})

    away_roll = rolling_df.merge(
        df[['match_idx', 'away_team']].rename(columns={'away_team': 'team_id'}),
        on=['match_idx', 'team_id'], how='inner'
    ).rename(columns={c: f'a_{c}' for c in rolling_df.columns if c not in ('match_idx', 'team_id')})

    df_feat = df_feat.merge(home_roll.drop(columns=['team_id'], errors='ignore'),
                            on='match_idx', how='left')
    df_feat = df_feat.merge(away_roll.drop(columns=['team_id'], errors='ignore'),
                            on='match_idx', how='left')

    # Differential features (home minus away) for each rolling stat
    h_cols = [c for c in df_feat.columns if c.startswith('h_') and '_w' in c]
    for h_col in h_cols:
        a_col = 'a_' + h_col[2:]
        if a_col in df_feat.columns:
            stat_name = h_col[2:]  # strip 'h_'
            df_feat[f'diff_{stat_name}'] = df_feat[h_col] - df_feat[a_col]

    # H2H features
    df_feat = _add_h2h_features(df_feat)

    return df_feat


def _add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add recency-weighted H2H win/draw/loss rate features.
    Only populated when >= MIN_H2H_MEETINGS prior meetings exist.
    Decay factor H2H_DECAY applied per meeting going back in time.
    """
    df = df.copy().sort_values('date').reset_index(drop=True)
    h2h_records = {}

    h2h_h_win  = np.full(len(df), np.nan)
    h2h_draw   = np.full(len(df), np.nan)
    h2h_h_loss = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        ht, at = row['home_team'], row['away_team']
        key = tuple(sorted([ht, at]))

        prior = h2h_records.get(key, [])
        if len(prior) >= MIN_H2H_MEETINGS:
            weights = np.array([H2H_DECAY ** j for j in range(len(prior) - 1, -1, -1)])
            weights = weights / weights.sum()
            h_wins = np.array([r['h_win'] for r in prior])
            draws  = np.array([r['draw']  for r in prior])
            h_losses = 1 - h_wins - draws

            h2h_h_win[i]  = float(np.dot(weights, h_wins))
            h2h_draw[i]   = float(np.dot(weights, draws))
            h2h_h_loss[i] = float(np.dot(weights, h_losses))

        # Record this match result (for future lookups)
        if not pd.isna(row.get('ftr')):
            h_win  = 1 if row['ftr'] == 'H' else 0
            draw   = 1 if row['ftr'] == 'D' else 0
            h2h_records.setdefault(key, []).append({
                'h_win': h_win, 'draw': draw,
                'home': ht, 'away': at
            })

    df['h2h_h_win']  = h2h_h_win
    df['h2h_draw']   = h2h_draw
    df['h2h_h_loss'] = h2h_h_loss

    return df


def get_latest_team_features(feature_df: pd.DataFrame,
                              window_config: Optional[Dict] = None) -> Dict:
    """
    Extract the most recent rolling feature vector for each team.
    Vectorised: sort once, ffill per team, take last row. O(n) not O(n*teams*stats).
    Returns dict: {team_id -> {'goals_for_w9': float, ...}}
    """
    if window_config is None:
        window_config = DEFAULT_WINDOW_CONFIG

    stat_keys = [f'{stat}_w{w}' for stat, w in window_config.items()]

    df = feature_df.sort_values('date').copy()

    home_cols = {f'h_{s}': s for s in stat_keys if f'h_{s}' in df.columns}
    away_cols = {f'a_{s}': s for s in stat_keys if f'a_{s}' in df.columns}

    home_long = (df[['date', 'home_team'] + list(home_cols.keys())]
                 .rename(columns={'home_team': 'team_id', **home_cols}))
    away_long = (df[['date', 'away_team'] + list(away_cols.keys())]
                 .rename(columns={'away_team': 'team_id', **away_cols}))

    long = pd.concat([home_long, away_long], ignore_index=True)
    long = long.sort_values(['team_id', 'date']).reset_index(drop=True)

    # For each team: forward-fill NaNs within their time series, take last row
    # Use explicit dict accumulation to avoid pandas groupby index ambiguity
    feature_db = {}
    for team_id, grp in long.groupby('team_id'):
        grp_sorted = grp.sort_values('date').ffill()
        if len(grp_sorted) == 0:
            feature_db[team_id] = {}
            continue
        last = grp_sorted.iloc[-1]
        feats = {}
        for s in stat_keys:
            if s in last.index and not pd.isna(last[s]):
                feats[s] = float(last[s])
        feature_db[team_id] = feats

    return feature_db

def build_feature_cols(window_config: Dict, include_market: bool = True) -> List[str]:
    """Return the ordered list of feature column names used by the ML model."""
    cols = []
    for stat, w in window_config.items():
        cols += [f'h_{stat}_w{w}', f'a_{stat}_w{w}', f'diff_{stat}_w{w}']

    cols += ['h2h_h_win', 'h2h_draw', 'h2h_h_loss']

    if include_market:
        cols += ['dc_prob_h', 'dc_prob_d', 'dc_prob_a',
                 'shin_h', 'shin_d', 'shin_a']

    return cols


# ===========================================================================
# SECTION 5: ML MODEL
# ===========================================================================

class CalibratedHistGBC:
    """
    HistGradientBoostingClassifier with per-class isotonic calibration.
    NaN values are handled natively by HistGBC.
    Calibration on a held-out set corrects the probability scale
    (HistGBC raw probs are overconfident; isotonic regression fixes this).
    """
    def __init__(self, **hparams):
        self.hparams = {**DEFAULT_HPARAMS, **hparams}
        self.base        = None
        self.calibrators = []
        self.classes_    = ['H', 'D', 'A']
        self.alpha       = 0.5   # Market blend weight; overwritten during training

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_calib: np.ndarray, y_calib: np.ndarray) -> 'CalibratedHistGBC':
        self.base = HistGradientBoostingClassifier(
            max_iter             = self.hparams['max_iter'],
            max_depth            = self.hparams['max_depth'],
            learning_rate        = self.hparams['learning_rate'],
            min_samples_leaf     = self.hparams['min_samples_leaf'],
            l2_regularization    = self.hparams['l2_regularization'],
            early_stopping       = True,
            n_iter_no_change     = 20,
            validation_fraction  = 0.15,
            tol                  = 1e-4,
            random_state         = 42,
        )
        self.base.fit(X_train, y_train)

        raw_calib = self.base.predict_proba(X_calib)
        self.calibrators = []
        for c in range(raw_calib.shape[1]):
            y_bin = (y_calib == c).astype(int)
            iso   = IsotonicRegression(out_of_bounds='clip')
            iso.fit(raw_calib[:, c], y_bin)
            self.calibrators.append(iso)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)
        # Support both isotonic (predict) and Platt/logistic (predict_proba) calibrators
        cal_cols = []
        for c in range(len(self.calibrators)):
            cal = self.calibrators[c]
            if hasattr(cal, 'predict_proba'):
                # Logistic regression calibrator
                col = cal.predict_proba(raw[:, c].reshape(-1, 1))[:, 1]
            else:
                # Isotonic regression calibrator (legacy)
                col = cal.predict(raw[:, c])
            cal_cols.append(col)
        cal = np.stack(cal_cols, axis=1)
        row_sums = cal.sum(axis=1, keepdims=True)
        return cal / np.where(row_sums > 0, row_sums, 1.0)


def _outcome_to_int(ftr: str) -> int:
    return {'H': 0, 'D': 1, 'A': 2}.get(ftr, -1)


def train_ml_model(feature_df: pd.DataFrame,
                   dc_params: Dict,
                   window_config: Optional[Dict] = None,
                   hparams: Optional[Dict] = None) -> Optional[CalibratedHistGBC]:
    """
    Train Market-Augmented ML model on historical feature matrix.

    ARCHITECTURE: Market-anchored blend.

    The core insight from Phase 6 backtesting is that the market (Shin fair
    probs) is a better baseline than DC probabilities alone. DC systematically
    overestimates home win probability relative to the market because it lacks
    real-time information (injuries, form, travel). Training a model to predict
    absolute H/D/A probabilities inherits this home bias.

    Instead, the model is trained to predict the actual outcome, but the final
    probabilities are computed as a weighted blend:

        final = alpha * ML_probs + (1 - alpha) * Shin_fair_probs

    where alpha is a learned blending weight that represents how much to trust
    the ML model's view vs the market. Alpha is fitted by minimising Brier Score
    on held-out OOF predictions, constrained to [0, 0.5] so the market always
    has at least 50% weight. This prevents the ML model from overriding market
    consensus and producing spurious home-win edges.

    The rolling form features (shots, goals, discipline) and DC probs are still
    used as input features — the model learns which form signals are genuinely
    predictive beyond what the market has already priced.

    CALIBRATION: 5-fold time-ordered cross-validation with Platt scaling.
    """
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize_scalar

    if window_config is None:
        window_config = DEFAULT_WINDOW_CONFIG
    if hparams is None:
        hparams = DEFAULT_HPARAMS

    # Compute DC and Shin probabilities for all historical matches
    feature_df = _add_dc_probs(feature_df.copy(), dc_params)
    feature_df = _add_shin_probs(feature_df)

    # Filter to rows with valid outcomes, odds, and DC probs
    valid = feature_df.dropna(subset=['ftr', 'dc_prob_h', 'shin_h']).copy()
    valid['y'] = valid['ftr'].map(_outcome_to_int)
    valid = valid[valid['y'] >= 0]

    if len(valid) < 200:
        return None

    feat_cols = build_feature_cols(window_config, include_market=True)
    feat_cols = [c for c in feat_cols if c in valid.columns]

    X_all = valid[feat_cols].values.astype(float)
    y_all = valid['y'].values

    # Shin probs for blending (shape: n_matches x 3)
    shin_all = valid[['shin_h', 'shin_d', 'shin_a']].values.astype(float)

    hparams_full = {**DEFAULT_HPARAMS, **hparams}

    # --- Step 1: Out-of-fold ML probabilities (time-ordered KFold) ---
    n_splits = min(5, len(valid) // 100)
    if n_splits < 2:
        return None

    kf = KFold(n_splits=n_splits, shuffle=False)
    oof_raw = np.zeros((len(X_all), 3))

    for train_idx, val_idx in kf.split(X_all):
        base_cv = HistGradientBoostingClassifier(
            max_iter            = hparams_full['max_iter'],
            max_depth           = hparams_full['max_depth'],
            learning_rate       = hparams_full['learning_rate'],
            min_samples_leaf    = hparams_full['min_samples_leaf'],
            l2_regularization   = hparams_full['l2_regularization'],
            early_stopping      = True,
            n_iter_no_change    = 20,
            validation_fraction = 0.15,
            tol                 = 1e-4,
            random_state        = 42,
        )
        base_cv.fit(X_all[train_idx], y_all[train_idx])
        oof_raw[val_idx] = base_cv.predict_proba(X_all[val_idx])

    # --- Step 2: Platt calibration on OOF predictions ---
    calibrators = []
    for c in range(3):
        platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        platt.fit(oof_raw[:, c].reshape(-1, 1), (y_all == c).astype(int))
        calibrators.append(platt)

    # Apply calibration to OOF predictions to get calibrated OOF probs
    oof_cal = np.stack(
        [cal.predict_proba(oof_raw[:, c].reshape(-1, 1))[:, 1]
         for c, cal in enumerate(calibrators)], axis=1
    )
    # Renormalise rows
    oof_cal = oof_cal / oof_cal.sum(axis=1, keepdims=True)

    # --- Step 3: Find optimal market blend weight (alpha) ---
    # Minimise Brier Score of: alpha * oof_cal + (1-alpha) * shin_all
    # Constrained to [0, 0.5] — market always has at least 50% weight.
    outcomes_oh = np.zeros((len(y_all), 3))
    for i, yi in enumerate(y_all):
        outcomes_oh[i, yi] = 1.0

    def brier(alpha):
        blended = alpha * oof_cal + (1 - alpha) * shin_all
        return float(np.mean(np.sum((blended - outcomes_oh) ** 2, axis=1)))

    result = minimize_scalar(brier, bounds=(0.0, 0.5), method='bounded')
    alpha = float(result.x)

    # --- Step 4: Train final base model on ALL data ---
    final_base = HistGradientBoostingClassifier(
        max_iter            = hparams_full['max_iter'],
        max_depth           = hparams_full['max_depth'],
        learning_rate       = hparams_full['learning_rate'],
        min_samples_leaf    = hparams_full['min_samples_leaf'],
        l2_regularization   = hparams_full['l2_regularization'],
        early_stopping      = True,
        n_iter_no_change    = 20,
        validation_fraction = 0.15,
        tol                 = 1e-4,
        random_state        = 42,
    )
    final_base.fit(X_all, y_all)

    # --- Step 5: Assemble model ---
    model = CalibratedHistGBC(**hparams)
    model.base        = final_base
    model.calibrators = calibrators
    model.alpha       = alpha   # Market blend weight
    model.feat_cols   = feat_cols

    return model


def _add_dc_probs(df: pd.DataFrame, dc_params: Dict) -> pd.DataFrame:
    """Add dc_prob_h/d/a columns to feature DataFrame."""
    probs_h, probs_d, probs_a = [], [], []
    for _, row in df.iterrows():
        p = dc_predict_proba(row['home_team'], row['away_team'], dc_params)
        if p is not None:
            probs_h.append(p[0])
            probs_d.append(p[1])
            probs_a.append(p[2])
        else:
            probs_h.append(np.nan)
            probs_d.append(np.nan)
            probs_a.append(np.nan)

    df['dc_prob_h'] = probs_h
    df['dc_prob_d'] = probs_d
    df['dc_prob_a'] = probs_a
    return df


def _add_shin_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Add Shin fair probability columns to feature DataFrame."""
    shin_h, shin_d, shin_a = [], [], []
    for _, row in df.iterrows():
        try:
            ah, ad, aa = row['avg_h'], row['avg_d'], row['avg_a']
            if pd.isna(ah) or pd.isna(ad) or pd.isna(aa):
                raise ValueError
            ph, pd_, pa = shin_normalise(float(ah), float(ad), float(aa))
            shin_h.append(ph); shin_d.append(pd_); shin_a.append(pa)
        except Exception:
            shin_h.append(np.nan); shin_d.append(np.nan); shin_a.append(np.nan)

    df['shin_h'] = shin_h
    df['shin_d'] = shin_d
    df['shin_a'] = shin_a
    return df


# ===========================================================================
# SECTION 6: FULL TRAINING PIPELINE
# ===========================================================================

def train_league(data_dir: str, models_dir: str, div_code: str,
                 league_config,
                 window_config: Optional[Dict] = None,
                 hparams: Optional[Dict] = None,
                 verbose: bool = True) -> Dict:
    """
    Full training pipeline for a single league.

    Loads all CSVs from data_dir, fits DC model, builds feature matrix,
    trains Market-Augmented ML model, and saves all artifacts to models_dir.

    Returns a metadata dict summarising the training run.
    """
    if window_config is None:
        window_config = DEFAULT_WINDOW_CONFIG
    if hparams is None:
        hparams = DEFAULT_HPARAMS

    os.makedirs(models_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {league_config.name} ({div_code})")
        print(f"{'='*60}")

    # 1. Load data
    matches = load_league_data(data_dir, league_config, verbose=verbose)

    # 2. Fit Dixon-Coles
    if verbose:
        print(f"\nFitting Dixon-Coles model...")
    dc_params = fit_dc_model(matches, exclude_no_crowd=True)
    if verbose:
        print(f"  gamma={dc_params['gamma']:.3f}, rho={dc_params['rho']:.4f}, "
              f"n_teams={len(dc_params['teams'])}, "
              f"converged={dc_params['converged']}")

    # 3. Build feature matrix
    if verbose:
        print(f"\nBuilding feature matrix...")
    feature_df = build_feature_matrix(matches, window_config)
    if verbose:
        print(f"  Feature matrix: {len(feature_df)} rows, "
              f"{sum(c.startswith(('h_', 'a_', 'diff_', 'h2h')) for c in feature_df.columns)} feature cols")

    # 4. Extract latest team features (for live predictions)
    feature_db = get_latest_team_features(feature_df, window_config)

    # 5. Train ML model
    ml_model = None
    if verbose:
        print(f"\nTraining Market-Augmented ML model...")
    ml_model = train_ml_model(feature_df, dc_params, window_config, hparams)
    if ml_model is not None:
        if verbose:
            print(f"  ML model trained successfully "
                  f"({len(ml_model.feat_cols)} features)")
    else:
        if verbose:
            print(f"  Insufficient data for ML model — DC only")

    # 6. Save artifacts
    dc_path      = os.path.join(models_dir, f'{div_code}_dc_params.json')
    ml_path      = os.path.join(models_dir, f'{div_code}_ml_model.pkl')
    featdb_path  = os.path.join(models_dir, f'{div_code}_feature_db.pkl')
    meta_path    = os.path.join(models_dir, f'{div_code}_metadata.json')

    with open(dc_path, 'w') as f:
        json.dump(dc_params, f, indent=2)

    if ml_model is not None:
        with open(ml_path, 'wb') as f:
            pickle.dump(ml_model, f)

    with open(featdb_path, 'wb') as f:
        pickle.dump(feature_db, f)

    metadata = {
        'div_code':      div_code,
        'league_name':   league_config.name,
        'n_matches':     int(len(matches)),
        'n_teams':       len(dc_params['teams']),
        'seasons':       sorted(matches['season'].unique().tolist()),
        'has_ml_model':  ml_model is not None,
        'dc_gamma':      round(dc_params['gamma'], 4),
        'dc_rho':        round(dc_params['rho'], 5),
        'dc_converged':  dc_params['converged'],
        'trained_at':    datetime.now().isoformat(),
        'data_dir':      data_dir,
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nArtifacts saved to {models_dir}/")
        print(f"  DC params:    {os.path.basename(dc_path)}")
        print(f"  ML model:     {os.path.basename(ml_path) if ml_model else 'N/A (DC only)'}")
        print(f"  Feature DB:   {os.path.basename(featdb_path)}")
        print(f"  Metadata:     {os.path.basename(meta_path)}")

    return metadata


def load_league_model(models_dir: str, div_code: str) -> Dict:
    """
    Load all saved artifacts for a league.

    Returns dict with keys: dc_params, ml_model (or None), feature_db, metadata.
    Raises FileNotFoundError if the league has not been trained yet.
    """
    meta_path = os.path.join(models_dir, f'{div_code}_metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No trained model found for {div_code}")

    with open(meta_path) as f:
        metadata = json.load(f)

    dc_path = os.path.join(models_dir, f'{div_code}_dc_params.json')
    with open(dc_path) as f:
        dc_params = json.load(f)

    ml_model = None
    ml_path  = os.path.join(models_dir, f'{div_code}_ml_model.pkl')
    if os.path.exists(ml_path):
        with open(ml_path, 'rb') as f:
            ml_model = pickle.load(f)

    featdb_path = os.path.join(models_dir, f'{div_code}_feature_db.pkl')
    with open(featdb_path, 'rb') as f:
        feature_db = pickle.load(f)

    return {
        'dc_params':   dc_params,
        'ml_model':    ml_model,
        'feature_db':  feature_db,
        'metadata':    metadata,
    }
