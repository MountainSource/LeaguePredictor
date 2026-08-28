"""
backtest.py
-----------
Walk-forward backtest for the Football Predictor — any league with data.

USAGE
-----
    python backtest.py                  # all leagues found in data/
    python backtest.py E0               # EPL only
    python backtest.py E0 D1 SP1        # three specific leagues

The script auto-discovers leagues by scanning data/ subdirectories and
matching them against LEAGUE_REGISTRY. Any league folder with at least 3
seasons of CSV files will be included.

METHODOLOGY
-----------
Season-by-season walk-forward per league. For each test season, DC and ML
are retrained on all prior seasons for that league only. Rolling features are
built once per league on all data — the shift(1) inside build_feature_matrix
prevents look-ahead.

Bets from all leagues are merged chronologically into one stream and staked
from a single shared bankroll. This mirrors real usage where you're betting
across leagues simultaneously.

Fold structure per league (minimum 2 prior seasons required):
  Fold 1: train on seasons 1-2  -> test season 3
  Fold 2: train on seasons 1-3  -> test season 4
  ...etc

STAKING
-------
  Quarter Kelly: f = (p*d - 1) / (d - 1), stake = (f/4) x bankroll, capped at 5%
  Flat stake:    1 GBP per bet (fixed)

OUTPUT
------
  backtest_report.html  standalone HTML report with charts
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))

from model_trainer import (
    load_league_data,
    fit_dc_model,
    build_feature_matrix,
    train_ml_model,
    dc_predict_proba,
    shin_normalise,
    build_feature_cols,
    DEFAULT_WINDOW_CONFIG,
    DEFAULT_HPARAMS,
)
from league_configs import LEAGUE_REGISTRY

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_ROOT       = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_PATH     = os.path.join(SCRIPT_DIR, 'backtest_report.html')
STARTING_BANK   = 100.0
FLAT_STAKE      = 1.0
KELLY_FRACTION  = 0.25
KELLY_CAP       = 0.05
EDGE_THRESHOLD  = 0.03
HOME_EDGE_PENALTY = 0.06
MIN_SEASONS     = 3   # need at least 3 seasons (2 train, 1 test) per league

BOOKMAKER_COLS = {
    'B365': 'Bet365',
    'BFD':  'Betfred',
    'BMGM': 'BetMGM',
    'BV':   'Betvictor',
    'BW':   'Bet&Win',
    'CL':   'Coral',
    'LB':   'Ladbrokes',
    'PS':   'Pinnacle',
    'BFE':  'Betfair Exch',
}

PALETTE = [
    '#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#a78bfa',
    '#22d3ee', '#fb923c',
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if not np.isnan(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def best_odds_for_outcome(row, outcome_idx: int) -> Tuple[Optional[float], Optional[str]]:
    suffix = ['H', 'D', 'A'][outcome_idx]
    best_odds, best_name = None, None
    for code, name in BOOKMAKER_COLS.items():
        val = safe_float(row.get(f'{code}{suffix}'))
        if val and val > 1.01:
            if best_odds is None or val > best_odds:
                best_odds, best_name = val, name
    return best_odds, best_name


def kelly_stake(model_prob: float, decimal_odds: float, bankroll: float) -> float:
    if decimal_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b = decimal_odds - 1.0
    f = (model_prob * decimal_odds - 1.0) / b
    qk = f * KELLY_FRACTION
    if qk <= 0:
        return 0.0
    return round(min(qk, KELLY_CAP) * bankroll, 4)


def get_team_features_at_date(all_feature_df, team, cutoff_date, window_config):
    """Rolling features for a team using only data before cutoff_date."""
    stat_keys = [f'{stat}_w{w}' for stat, w in window_config.items()]
    h_cols = {f'h_{s}': s for s in stat_keys if f'h_{s}' in all_feature_df.columns}
    a_cols = {f'a_{s}': s for s in stat_keys if f'a_{s}' in all_feature_df.columns}

    home_rows = (all_feature_df[all_feature_df['home_team'] == team]
                 [['date'] + list(h_cols.keys())].rename(columns=h_cols))
    away_rows = (all_feature_df[all_feature_df['away_team'] == team]
                 [['date'] + list(a_cols.keys())].rename(columns=a_cols))

    combined = pd.concat([home_rows, away_rows], ignore_index=True)
    combined = combined[combined['date'] < cutoff_date].sort_values('date')
    if len(combined) == 0:
        return {}
    combined = combined.ffill()
    last = combined.iloc[-1]
    return {s: float(last[s]) for s in stat_keys if s in last.index and not pd.isna(last[s])}


def build_live_feature_vector(home_team, away_team, feature_db,
                               dc_probs, shin_probs, feat_cols, window_config):
    h_feats = feature_db.get(home_team, {})
    a_feats = feature_db.get(away_team, {})
    vec = []
    for col in feat_cols:
        if col.startswith('h_') and '_w' in col:
            vec.append(h_feats.get(col[2:], np.nan))
        elif col.startswith('a_') and '_w' in col:
            vec.append(a_feats.get(col[2:], np.nan))
        elif col.startswith('diff_') and '_w' in col:
            stat = col[5:]
            h_val, a_val = h_feats.get(stat, np.nan), a_feats.get(stat, np.nan)
            vec.append(h_val - a_val if not (np.isnan(h_val) or np.isnan(a_val)) else np.nan)
        elif col in ('h2h_h_win', 'h2h_draw', 'h2h_h_loss'):
            vec.append(np.nan)
        elif col == 'dc_prob_h': vec.append(float(dc_probs[0]))
        elif col == 'dc_prob_d': vec.append(float(dc_probs[1]))
        elif col == 'dc_prob_a': vec.append(float(dc_probs[2]))
        elif col == 'shin_h':    vec.append(float(shin_probs[0]))
        elif col == 'shin_d':    vec.append(float(shin_probs[1]))
        elif col == 'shin_a':    vec.append(float(shin_probs[2]))
        else:
            vec.append(np.nan)
    return np.array(vec, dtype=float)


# ---------------------------------------------------------------------------
# LEAGUE DISCOVERY
# ---------------------------------------------------------------------------

def discover_leagues(data_root, requested=None):
    available = []
    for div_code, cfg in LEAGUE_REGISTRY.items():
        if requested and div_code not in requested:
            continue
        league_dir = os.path.join(data_root, div_code)
        if not os.path.isdir(league_dir):
            continue
        csv_count = sum(1 for f in os.listdir(league_dir) if f.lower().endswith('.csv'))
        if csv_count >= MIN_SEASONS:
            available.append(div_code)
        else:
            print(f"  Skipping {div_code} ({cfg.name}): only {csv_count} CSV(s), need {MIN_SEASONS}")
    return available


# ---------------------------------------------------------------------------
# SINGLE-LEAGUE BACKTEST
# ---------------------------------------------------------------------------

def run_league_backtest(div_code):
    league_cfg = LEAGUE_REGISTRY[div_code]
    data_dir   = os.path.join(DATA_ROOT, div_code)

    print(f"\n{'='*60}")
    print(f"  {league_cfg.name} ({div_code})")
    print(f"{'='*60}")

    all_data = load_league_data(data_dir, league_cfg, verbose=True)
    all_data = all_data.sort_values('date').reset_index(drop=True)
    seasons  = sorted(all_data['season'].unique())
    print(f"  Seasons: {seasons}")

    if len(seasons) < MIN_SEASONS:
        print(f"  Insufficient seasons ({len(seasons)}) - skipping")
        return []

    print(f"  Building feature matrix...")
    full_feature_df = build_feature_matrix(all_data, DEFAULT_WINDOW_CONFIG)

    all_bets = []

    for fold_idx, test_season in enumerate(seasons[2:], start=1):
        train_seasons = seasons[:seasons.index(test_season)]
        train_data    = all_data[all_data['season'].isin(train_seasons)].copy()
        test_data     = all_data[all_data['season'] == test_season].copy()

        print(f"\n  Fold {fold_idx}: train {train_seasons[0]}-{train_seasons[-1]}"
              f" -> test {test_season} ({len(test_data)} matches)")

        # Fit DC excluding no-crowd seasons where possible. If all training data
        # is COVID-flagged (leagues that start from 2019/20 only), fall back to
        # fitting on the full set rather than crashing. Home advantage estimates
        # will be slightly off but better than having no model at all.
        crowd_data = (train_data[train_data['has_crowd']]
                      if 'has_crowd' in train_data.columns else train_data)
        if len(crowd_data['home_team'].unique()) < 2:
            print(f"    WARNING: all training data is no-crowd — fitting DC on full set")
            dc_params = fit_dc_model(train_data, exclude_no_crowd=False)
        else:
            dc_params = fit_dc_model(train_data, exclude_no_crowd=True)

        train_feat = full_feature_df[full_feature_df['season'].isin(train_seasons)].copy()
        ml_model   = train_ml_model(train_feat, dc_params, DEFAULT_WINDOW_CONFIG, DEFAULT_HPARAMS)

        if ml_model is None:
            print(f"    ML model could not be trained - skipping fold")
            continue
        print(f"    alpha={ml_model.alpha:.3f}")

        fold_bets = 0
        for _, match in test_data.iterrows():
            home_team  = match['home_team']
            away_team  = match['away_team']
            match_date = match['date']

            odds_h = safe_float(match.get('avg_h'))
            odds_d = safe_float(match.get('avg_d'))
            odds_a = safe_float(match.get('avg_a'))
            if odds_h is None:
                odds_h = safe_float(match.get('b365_h'))
                odds_d = safe_float(match.get('b365_d'))
                odds_a = safe_float(match.get('b365_a'))
            if not (odds_h and odds_d and odds_a):
                continue

            actual_ftr = match.get('ftr', '')
            if actual_ftr not in ('H', 'D', 'A'):
                continue

            # DC prediction (fold model only)
            dc_probs = dc_predict_proba(home_team, away_team, dc_params)
            if dc_probs is None:
                mean_a = np.mean(list(dc_params['alpha'].values()))
                mean_b = np.mean(list(dc_params['beta'].values()))
                pseudo = dict(dc_params)
                pseudo['alpha'][home_team] = mean_a
                pseudo['alpha'][away_team] = mean_a
                pseudo['beta'][home_team]  = mean_b
                pseudo['beta'][away_team]  = mean_b
                dc_probs = dc_predict_proba(home_team, away_team, pseudo)
            dc_probs = (np.array(dc_probs) if dc_probs is not None
                        else np.array([1/3, 1/3, 1/3]))

            sh, sd, sa = shin_normalise(odds_h, odds_d, odds_a)
            shin_probs = np.array([sh, sd, sa])

            h_feats = get_team_features_at_date(
                full_feature_df, home_team, match_date, DEFAULT_WINDOW_CONFIG)
            a_feats = get_team_features_at_date(
                full_feature_df, away_team, match_date, DEFAULT_WINDOW_CONFIG)

            feat_vec = build_live_feature_vector(
                home_team, away_team,
                {home_team: h_feats, away_team: a_feats},
                dc_probs, shin_probs, ml_model.feat_cols, DEFAULT_WINDOW_CONFIG
            )

            try:
                ml_probs = ml_model.predict_proba(feat_vec.reshape(1, -1))[0]
            except Exception:
                ml_probs = None

            if ml_probs is not None:
                blended = ml_model.alpha * ml_probs + (1.0 - ml_model.alpha) * shin_probs
                final_probs = blended / blended.sum()
            else:
                final_probs = dc_probs

            for i, outcome in enumerate(['H', 'D', 'A']):
                raw_edge = float(final_probs[i]) - float(shin_probs[i])
                adj_edge = raw_edge - (HOME_EDGE_PENALTY if outcome == 'H' else 0.0)
                if adj_edge < EDGE_THRESHOLD:
                    continue

                best_odd, best_book = best_odds_for_outcome(match, i)
                if best_odd is None:
                    best_odd  = [odds_h, odds_d, odds_a][i]
                    best_book = 'Avg'

                all_bets.append({
                    'date':        match_date,
                    'season':      test_season,
                    'fold':        fold_idx,
                    'div_code':    div_code,
                    'league_name': league_cfg.name,
                    'home_team':   home_team.split('::')[-1],
                    'away_team':   away_team.split('::')[-1],
                    'outcome':     outcome,
                    'model_prob':  round(float(final_probs[i]), 4),
                    'fair_prob':   round(float(shin_probs[i]), 4),
                    'raw_edge':    round(raw_edge, 4),
                    'adj_edge':    round(adj_edge, 4),
                    'best_odds':   round(best_odd, 2),
                    'best_bookie': best_book,
                    'actual_ftr':  actual_ftr,
                    'won':         actual_ftr == outcome,
                    'alpha':       round(float(ml_model.alpha), 3),
                })
                fold_bets += 1

        print(f"    Value bets: {fold_bets}")

    print(f"\n  {league_cfg.name} total: {len(all_bets)} value bets")
    return all_bets


# ---------------------------------------------------------------------------
# STAKING SIMULATION
# ---------------------------------------------------------------------------

def simulate_staking(bets):
    if not bets:
        return pd.DataFrame()

    df = pd.DataFrame(bets).sort_values(['date', 'div_code']).reset_index(drop=True)
    qk_bank, fl_bank = STARTING_BANK, STARTING_BANK
    qk_stakes, qk_pnls, qk_banks = [], [], []
    fl_stakes, fl_pnls, fl_banks = [], [], []

    for _, row in df.iterrows():
        # Quarter Kelly
        stake_qk = min(kelly_stake(row['model_prob'], row['best_odds'], qk_bank), qk_bank)
        pnl_qk   = round(stake_qk * (row['best_odds'] - 1.0) if row['won'] else -stake_qk, 4)
        qk_bank  = round(max(qk_bank + pnl_qk, 0.0), 4)
        qk_stakes.append(stake_qk); qk_pnls.append(pnl_qk); qk_banks.append(qk_bank)

        # Flat stake
        stake_fl = min(FLAT_STAKE, fl_bank)
        pnl_fl   = round(stake_fl * (row['best_odds'] - 1.0) if row['won'] else -stake_fl, 4)
        fl_bank  = round(max(fl_bank + pnl_fl, 0.0), 4)
        fl_stakes.append(stake_fl); fl_pnls.append(pnl_fl); fl_banks.append(fl_bank)

    df['qk_stake'] = qk_stakes; df['qk_pnl'] = qk_pnls; df['qk_bank'] = qk_banks
    df['fl_stake'] = fl_stakes; df['fl_pnl'] = fl_pnls; df['fl_bank'] = fl_banks
    return df


# ---------------------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------------------

def _max_drawdown(series, start):
    peak, max_dd = start, 0.0
    for v in series:
        if v > peak:
            peak = v
        if peak - v > max_dd:
            max_dd = peak - v
    return max_dd


def compute_summary(df):
    if df.empty:
        return {}
    total = len(df); wins = int(df['won'].sum())
    qk_st = df['qk_stake'].sum(); qk_pr = df['qk_pnl'].sum()
    fl_st = df['fl_stake'].sum(); fl_pr = df['fl_pnl'].sum()
    return {
        'total_bets':   total,
        'wins':         wins,
        'win_rate':     round(wins / total * 100, 1),
        'avg_edge':     round(df['adj_edge'].mean() * 100, 2),
        'avg_odds':     round(df['best_odds'].mean(), 2),
        'n_leagues':    int(df['div_code'].nunique()),
        'qk_staked':    round(qk_st, 2),
        'qk_profit':    round(qk_pr, 2),
        'qk_roi':       round(qk_pr / qk_st * 100 if qk_st else 0, 2),
        'qk_final':     round(float(df['qk_bank'].iloc[-1]), 2),
        'qk_max_dd':    round(_max_drawdown(df['qk_bank'].values, STARTING_BANK), 2),
        'qk_avg_stake': round(float(df['qk_stake'].mean()), 2),
        'fl_staked':    round(fl_st, 2),
        'fl_profit':    round(fl_pr, 2),
        'fl_roi':       round(fl_pr / fl_st * 100 if fl_st else 0, 2),
        'fl_final':     round(float(df['fl_bank'].iloc[-1]), 2),
        'fl_max_dd':    round(_max_drawdown(df['fl_bank'].values, STARTING_BANK), 2),
    }


def per_league_summary(df):
    rows = []
    for div_code, grp in df.groupby('div_code'):
        n  = len(grp); wins = grp['won'].sum()
        st = grp['fl_stake'].sum(); pr = grp['fl_pnl'].sum()
        rows.append({
            'div_code':    div_code,
            'league_name': grp['league_name'].iloc[0],
            'bets':        n,
            'wins':        int(wins),
            'win_rate':    round(wins / n * 100, 1) if n else 0,
            'avg_edge':    round(grp['adj_edge'].mean() * 100, 2),
            'avg_odds':    round(grp['best_odds'].mean(), 2),
            'profit':      round(pr, 2),
            'roi':         round(pr / st * 100 if st else 0, 1),
        })
    rows.sort(key=lambda r: r['bets'], reverse=True)
    return rows


def per_season_summary(df):
    rows = []
    for season, grp in df.groupby('season'):
        n  = len(grp); wins = grp['won'].sum()
        st = grp['fl_stake'].sum(); pr = grp['fl_pnl'].sum()
        rows.append({
            'season': season, 'bets': n, 'wins': int(wins),
            'profit': round(pr, 2),
            'roi':    round(pr / st * 100 if st else 0, 1),
        })
    rows.sort(key=lambda r: r['season'])
    return rows


def outcome_breakdown(df):
    result = {}
    for o in ['H', 'D', 'A']:
        grp = df[df['outcome'] == o]; n = len(grp)
        st  = grp['fl_stake'].sum(); pr = grp['fl_pnl'].sum()
        result[o] = {
            'bets': n, 'wins': int(grp['won'].sum()),
            'roi':  round(pr / st * 100 if st else 0, 1),
        }
    return result


# ---------------------------------------------------------------------------
# CHART DATA
# ---------------------------------------------------------------------------

def build_chart_data(df):
    step   = max(1, len(df) // 50)
    labels = [f"{row['date'].strftime('%b %y')}" if i % step == 0 else ''
              for i, row in df.iterrows()]

    bankroll_data = {
        'labels': labels,
        'qk': df['qk_bank'].round(2).tolist(),
        'fl': df['fl_bank'].round(2).tolist(),
    }

    # Per-league independent flat-stake curves
    league_curves = {}
    for div, grp in df.groupby('div_code'):
        grp_s  = grp.sort_values('date')
        bank   = STARTING_BANK
        curve  = []
        for _, row in grp_s.iterrows():
            stake = min(FLAT_STAKE, bank)
            pnl   = stake * (row['best_odds'] - 1.0) if row['won'] else -stake
            bank  = round(max(bank + pnl, 0.0), 2)
            curve.append(bank)
        league_curves[div] = {'name': grp['league_name'].iloc[0], 'values': curve}

    season_rows = per_season_summary(df)
    season_data = {
        'seasons': [r['season'] for r in season_rows],
        'roi':     [r['roi'] for r in season_rows],
    }

    league_rows = per_league_summary(df)
    league_cmp  = {
        'names': [r['div_code'] for r in league_rows],
        'bets':  [r['bets'] for r in league_rows],
        'roi':   [r['roi'] for r in league_rows],
    }

    ob = outcome_breakdown(df)
    outcome_data = {
        'bets': [ob[o]['bets'] for o in ['H', 'D', 'A']],
        'wins': [ob[o]['wins'] for o in ['H', 'D', 'A']],
    }

    edges_pct = (df['adj_edge'] * 100).values
    e_max = min(float(edges_pct.max()) + 2, 25)
    bins  = np.arange(3, e_max, 1)
    counts, _ = np.histogram(edges_pct, bins=bins)
    edge_data = {
        'labels': [f"{b:.0f}-{b+1:.0f}%" for b in bins[:-1]],
        'counts': counts.tolist(),
    }

    stakes = df['qk_stake'].values
    s_max  = min(float(stakes.max()) + 0.5, 8)
    s_bins = np.arange(0, s_max, 0.5)
    sc, _  = np.histogram(stakes, bins=s_bins)
    stake_data = {
        'labels': [f"GBP{b:.1f}" for b in s_bins[:-1]],
        'counts': sc.tolist(),
    }

    return {
        'bankroll':      bankroll_data,
        'league_curves': league_curves,
        'season':        season_data,
        'league_cmp':    league_cmp,
        'outcome':       outcome_data,
        'edge':          edge_data,
        'stake':         stake_data,
    }


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def render_html(df, summary, div_codes):
    if df.empty:
        return "<html><body><h1>No value bets found.</h1></body></html>"

    charts    = build_chart_data(df)
    generated = datetime.now().strftime('%d %b %Y %H:%M')
    seasons   = sorted(df['season'].unique())

    def pcolor(v):
        return 'green' if v > 0 else ('red' if v < 0 else 'amber')

    def rclass(v):
        return 'pos' if v >= 0 else 'neg'

    # --- League table ---
    league_rows_html = ''
    for r in per_league_summary(df):
        wp = round(r['wins'] / r['bets'] * 100, 1) if r['bets'] else 0
        league_rows_html += (
            f'<tr>'
            f'<td>{r["div_code"]}</td>'
            f'<td style="color:var(--text)">{r["league_name"]}</td>'
            f'<td>{r["bets"]}</td><td>{r["wins"]}</td><td>{wp}%</td>'
            f'<td class="amber">+{r["avg_edge"]}%</td>'
            f'<td>{r["avg_odds"]}</td>'
            f'<td class="{rclass(r["profit"])}">GBP{r["profit"]:+.2f}</td>'
            f'<td class="{rclass(r["roi"])}">{r["roi"]:+.1f}%</td>'
            f'</tr>'
        )

    # --- Season table ---
    season_rows_html = ''
    for r in per_season_summary(df):
        wp = round(r['wins'] / r['bets'] * 100, 1) if r['bets'] else 0
        season_rows_html += (
            f'<tr>'
            f'<td>{r["season"]}</td><td>{r["bets"]}</td><td>{r["wins"]}</td><td>{wp}%</td>'
            f'<td class="{rclass(r["profit"])}">GBP{r["profit"]:+.2f}</td>'
            f'<td class="{rclass(r["roi"])}">{r["roi"]:+.1f}%</td>'
            f'</tr>'
        )

    # --- Outcome table ---
    ob = outcome_breakdown(df)
    olabels = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    outcome_rows_html = ''
    for o in ['H', 'D', 'A']:
        od = ob[o]
        wp = round(od['wins'] / od['bets'] * 100, 1) if od['bets'] else 0
        outcome_rows_html += (
            f'<tr>'
            f'<td><span class="badge badge-{o.lower()}">{olabels[o]}</span></td>'
            f'<td>{od["bets"]}</td><td>{od["wins"]}</td><td>{wp}%</td>'
            f'<td class="{rclass(od["roi"])}">{od["roi"]:+.1f}%</td>'
            f'</tr>'
        )

    def crows(items):
        return '\n'.join(
            f'<div class="compare-row">'
            f'<span class="compare-key">{k}</span>'
            f'<span class="compare-val">{v}</span>'
            f'</div>'
            for k, v in items
        )

    qk_items = [
        ('Final Bankroll', f'GBP{summary["qk_final"]:,.2f}'),
        ('Total Profit',   f'GBP{summary["qk_profit"]:+,.2f}'),
        ('ROI on Staked',  f'{summary["qk_roi"]:+.2f}%'),
        ('Total Staked',   f'GBP{summary["qk_staked"]:,.2f}'),
        ('Avg Stake',      f'GBP{summary["qk_avg_stake"]:.2f}'),
        ('Max Drawdown',   f'GBP{summary["qk_max_dd"]:.2f}'),
    ]
    fl_items = [
        ('Final Bankroll', f'GBP{summary["fl_final"]:,.2f}'),
        ('Total Profit',   f'GBP{summary["fl_profit"]:+,.2f}'),
        ('ROI on Staked',  f'{summary["fl_roi"]:+.2f}%'),
        ('Total Staked',   f'GBP{summary["fl_staked"]:,.2f}'),
        ('Stake Per Bet',  f'GBP{FLAT_STAKE:.2f} fixed'),
        ('Max Drawdown',   f'GBP{summary["fl_max_dd"]:.2f}'),
    ]

    # --- Bets table (200 most recent) ---
    bet_rows_html = ''
    for _, row in df.sort_values('date', ascending=False).head(200).iterrows():
        won_cls = 'pos' if row['won'] else 'neg'
        qk_cls  = 'pos' if row['qk_pnl'] > 0 else 'neg'
        fl_cls  = 'pos' if row['fl_pnl'] > 0 else 'neg'
        bet_rows_html += (
            f'<tr>'
            f'<td>{row["date"].strftime("%d/%m/%y")}</td>'
            f'<td style="color:var(--blue)">{row["div_code"]}</td>'
            f'<td>{row["home_team"]} v {row["away_team"]}</td>'
            f'<td><span class="badge badge-{row["outcome"].lower()}">{row["outcome"]}</span></td>'
            f'<td>{row["model_prob"]*100:.1f}%</td>'
            f'<td>{row["fair_prob"]*100:.1f}%</td>'
            f'<td class="amber">+{row["adj_edge"]*100:.1f}%</td>'
            f'<td>{row["best_odds"]:.2f}</td>'
            f'<td>{row["best_bookie"]}</td>'
            f'<td class="{won_cls}">{"V" if row["won"] else "X"} {row["actual_ftr"]}</td>'
            f'<td>GBP{row["qk_stake"]:.2f}</td>'
            f'<td class="{qk_cls}">GBP{row["qk_pnl"]:+.2f}</td>'
            f'<td class="{fl_cls}">GBP{row["fl_pnl"]:+.2f}</td>'
            f'</tr>'
        )

    # Build per-league curve datasets
    league_curve_datasets = []
    for i, (div, curve_data) in enumerate(charts['league_curves'].items()):
        colour = PALETTE[i % len(PALETTE)]
        league_curve_datasets.append({
            'label':           curve_data['name'],
            'data':            curve_data['values'],
            'borderColor':     colour,
            'backgroundColor': 'transparent',
            'borderWidth':     1.5,
            'pointRadius':     0,
            'tension':         0.3,
        })

    max_len = max((len(c['values']) for c in charts['league_curves'].values()), default=0)
    lc_labels = list(range(1, max_len + 1))
    for ds in league_curve_datasets:
        while len(ds['data']) < max_len:
            ds['data'].append(ds['data'][-1] if ds['data'] else STARTING_BANK)

    n_leagues = summary['n_leagues']
    league_subtitle = (f'{n_leagues} League{"s" if n_leagues != 1 else ""}')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Football Predictor - Backtest Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Barlow+Condensed:wght@400;500;600;700&family=Barlow:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:#080c14;--surface:#0e1420;--surface2:#151d2e;--border:#1e2d44;--border2:#243450;
  --text:#e2e8f0;--text-muted:#64748b;--text-dim:#334155;
  --green:#10b981;--green-glow:rgba(16,185,129,0.12);
  --amber:#f59e0b;--red:#ef4444;--blue:#3b82f6;--blue-dim:#1e3a5f;
  --mono:'IBM Plex Mono',monospace;--sans:'Barlow',sans-serif;--cond:'Barlow Condensed',sans-serif;
}}
html,body{{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5}}
body::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.025) 2px,rgba(0,0,0,0.025) 4px)}}
.page{{max-width:1280px;margin:0 auto;padding:32px 24px}}
header{{border-bottom:1px solid var(--border);padding-bottom:20px;margin-bottom:32px;display:flex;align-items:flex-end;justify-content:space-between;gap:20px}}
.logo-mark{{display:inline-flex;align-items:center;gap:10px}}
.logo-box{{width:28px;height:28px;border:2px solid var(--green);display:grid;place-items:center;font-family:var(--mono);font-size:11px;font-weight:600;color:var(--green)}}
.logo-title{{font-family:var(--cond);font-size:22px;font-weight:700;letter-spacing:2px;text-transform:uppercase}}
.logo-sub{{font-family:var(--mono);font-size:10px;color:var(--text-muted);letter-spacing:1px;text-transform:uppercase}}
.report-meta{{font-family:var(--mono);font-size:11px;color:var(--text-muted);text-align:right}}
h2{{font-family:var(--cond);font-size:13px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:var(--text-muted);margin-bottom:16px;border-bottom:1px solid var(--border);padding-bottom:6px;margin-top:8px}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-bottom:32px}}
.stat-card{{background:var(--surface);border:1px solid var(--border);padding:16px 18px}}
.stat-label{{font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin-bottom:6px}}
.stat-value{{font-family:var(--mono);font-size:24px;font-weight:600;line-height:1}}
.stat-sub{{font-family:var(--mono);font-size:10px;color:var(--text-dim);margin-top:4px}}
.green{{color:var(--green)}}.amber{{color:var(--amber)}}.red{{color:var(--red)}}.white{{color:var(--text)}}.blue{{color:var(--blue)}}
.charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:32px}}
.chart-full{{grid-column:1/-1}}
.chart-card{{background:var(--surface);border:1px solid var(--border);padding:20px}}
.chart-title{{font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);margin-bottom:16px}}
canvas{{max-height:300px}}canvas.tall{{max-height:380px}}
.comparison-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:32px}}
.compare-card{{background:var(--surface);border:1px solid var(--border);padding:20px}}
.compare-title{{font-family:var(--cond);font-size:16px;font-weight:700;letter-spacing:1px;margin-bottom:16px}}
.compare-title.qk{{color:var(--green)}}.compare-title.fl{{color:var(--blue)}}
.compare-row{{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--border)}}
.compare-row:last-child{{border-bottom:none}}
.compare-key{{font-family:var(--mono);font-size:11px;color:var(--text-muted)}}
.compare-val{{font-family:var(--mono);font-size:13px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px;margin-bottom:32px}}
th{{background:var(--surface2);padding:8px 12px;text-align:left;font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);border-bottom:1px solid var(--border2);white-space:nowrap}}
td{{padding:8px 12px;border-bottom:1px solid var(--border);color:var(--text-muted)}}
tr:hover td{{background:var(--surface2)}}
.pos{{color:var(--green)}}.neg{{color:var(--red)}}
.badge{{display:inline-block;padding:1px 6px;font-size:9px;letter-spacing:1px}}
.badge-h{{background:rgba(59,130,246,0.15);color:var(--blue)}}
.badge-d{{background:rgba(100,116,139,0.15);color:var(--text-muted)}}
.badge-a{{background:rgba(239,68,68,0.15);color:var(--red)}}
.methodology{{background:var(--surface);border:1px solid var(--border);padding:20px;margin-bottom:32px;font-family:var(--mono);font-size:11px;color:var(--text-muted);line-height:1.8}}
.methodology p{{margin-bottom:8px}}.methodology strong{{color:var(--text)}}
</style>
</head>
<body>
<div class="page">

<header>
  <div class="logo-mark">
    <div class="logo-box">FP</div>
    <div>
      <div class="logo-title">Backtest Report</div>
      <div class="logo-sub">Walk-Forward · Market-Augmented ML · {league_subtitle}</div>
    </div>
  </div>
  <div class="report-meta">
    Generated: {generated}<br>
    Seasons: {seasons[0]} to {seasons[-1]}<br>
    Leagues: {', '.join(div_codes)}<br>
    Starting Bankroll: GBP{STARTING_BANK:.0f}
  </div>
</header>

<h2>Methodology</h2>
<div class="methodology">
  <p><strong>Walk-forward folds:</strong> Per league, DC and ML are retrained on all prior seasons for each test season. No future data used at prediction time. Rolling features use a 1-match lag throughout. Bets from all leagues are merged chronologically and staked from a single shared bankroll.</p>
  <p><strong>Value bet threshold:</strong> Flagged when adjusted edge is 3% or more (after 6pp home-win penalty compensating for DC home bias). Best odds taken across: {', '.join(BOOKMAKER_COLS.values())}.</p>
  <p><strong>Quarter Kelly:</strong> f = (p x d - 1) / (d - 1), stake = (f / 4) x bankroll, capped at 5% of current bankroll per bet.</p>
  <p><strong>Flat stake:</strong> GBP{FLAT_STAKE:.0f} per bet (fixed).</p>
</div>

<h2>Key Statistics - All Leagues Combined</h2>
<div class="stats-grid">
  <div class="stat-card"><div class="stat-label">Value Bets</div>
    <div class="stat-value white">{summary["total_bets"]}</div>
    <div class="stat-sub">{n_leagues} league{"s" if n_leagues != 1 else ""} across {len(seasons)} seasons</div></div>
  <div class="stat-card"><div class="stat-label">Win Rate</div>
    <div class="stat-value white">{summary["win_rate"]}%</div>
    <div class="stat-sub">{summary["wins"]} winners</div></div>
  <div class="stat-card"><div class="stat-label">Avg Edge</div>
    <div class="stat-value amber">+{summary["avg_edge"]}%</div>
    <div class="stat-sub">avg best odds {summary["avg_odds"]}</div></div>
  <div class="stat-card"><div class="stat-label">QK Final Bankroll</div>
    <div class="stat-value {pcolor(summary["qk_profit"])}">GBP{summary["qk_final"]:,.2f}</div>
    <div class="stat-sub">ROI {summary["qk_roi"]:+.2f}% · Max DD GBP{summary["qk_max_dd"]:.2f}</div></div>
  <div class="stat-card"><div class="stat-label">Flat Final Bankroll</div>
    <div class="stat-value {pcolor(summary["fl_profit"])}">GBP{summary["fl_final"]:,.2f}</div>
    <div class="stat-sub">ROI {summary["fl_roi"]:+.2f}% · Max DD GBP{summary["fl_max_dd"]:.2f}</div></div>
  <div class="stat-card"><div class="stat-label">QK Avg Stake</div>
    <div class="stat-value blue">GBP{summary["qk_avg_stake"]:.2f}</div>
    <div class="stat-sub">total staked GBP{summary["qk_staked"]:,.2f}</div></div>
</div>

<h2>Combined Bankroll Over Time</h2>
<div class="chart-card chart-full" style="margin-bottom:20px">
  <div class="chart-title">Combined Bankroll - Quarter Kelly vs Flat Stake (all leagues, shared bankroll)</div>
  <canvas id="bankrollChart" class="tall"></canvas>
</div>

<div class="charts-grid">
  <div class="chart-card"><div class="chart-title">ROI by Season (Flat Stake, all leagues)</div>
    <canvas id="seasonChart"></canvas></div>
  <div class="chart-card"><div class="chart-title">Bets and ROI by League</div>
    <canvas id="leagueCmpChart"></canvas></div>
  <div class="chart-card"><div class="chart-title">Value Bets by Outcome Type</div>
    <canvas id="outcomeChart"></canvas></div>
  <div class="chart-card"><div class="chart-title">Edge Distribution</div>
    <canvas id="edgeChart"></canvas></div>
</div>

<h2>Per-League Bankroll (Flat Stake, each simulated independently from GBP{STARTING_BANK:.0f})</h2>
<div class="chart-card chart-full" style="margin-bottom:32px">
  <div class="chart-title">Individual League Flat-Stake Bankroll Curves</div>
  <canvas id="leagueCurveChart" class="tall"></canvas>
</div>

<h2>Strategy Comparison</h2>
<div class="comparison-grid">
  <div class="compare-card">
    <div class="compare-title qk">Quarter Kelly</div>
    {crows(qk_items)}
  </div>
  <div class="compare-card">
    <div class="compare-title fl">Flat Stake (GBP{FLAT_STAKE:.0f} per bet)</div>
    {crows(fl_items)}
  </div>
</div>

<h2>Per-League Breakdown</h2>
<table><thead><tr>
  <th>Code</th><th>League</th><th>Bets</th><th>Wins</th><th>Win %</th>
  <th>Avg Edge</th><th>Avg Odds</th><th>Flat P&amp;L</th><th>Flat ROI</th>
</tr></thead><tbody>{league_rows_html}</tbody></table>

<h2>Per-Season Breakdown (all leagues combined, flat stake)</h2>
<table><thead><tr>
  <th>Season</th><th>Bets</th><th>Wins</th><th>Win %</th><th>P&amp;L</th><th>ROI</th>
</tr></thead><tbody>{season_rows_html}</tbody></table>

<h2>Outcome Type Breakdown</h2>
<table><thead><tr>
  <th>Outcome</th><th>Bets</th><th>Wins</th><th>Win %</th><th>Flat ROI</th>
</tr></thead><tbody>{outcome_rows_html}</tbody></table>

<h2>All Value Bets (200 most recent)</h2>
<table><thead><tr>
  <th>Date</th><th>Lg</th><th>Match</th><th>Bet</th>
  <th>Model%</th><th>Fair%</th><th>Edge</th><th>Odds</th><th>Bookie</th>
  <th>Result</th><th>QK Stake</th><th>QK P&amp;L</th><th>FL P&amp;L</th>
</tr></thead><tbody>{bet_rows_html}</tbody></table>

</div>
<script>
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = '#1e2d44';
Chart.defaults.font = {{family: "'IBM Plex Mono', monospace", size: 10}};

const bankrollData  = {json.dumps(charts["bankroll"])};
const seasonData    = {json.dumps(charts["season"])};
const leagueCmpData = {json.dumps(charts["league_cmp"])};
const outcomeData   = {json.dumps(charts["outcome"])};
const edgeData      = {json.dumps(charts["edge"])};
const lcDatasets    = {json.dumps(league_curve_datasets)};
const lcLabels      = {json.dumps(lc_labels)};
const startingBank  = {STARTING_BANK};

new Chart(document.getElementById('bankrollChart'), {{
  type: 'line',
  data: {{labels: bankrollData.labels, datasets: [
    {{label:'Quarter Kelly', data:bankrollData.qk, borderColor:'#10b981', backgroundColor:'rgba(16,185,129,0.05)', borderWidth:2, pointRadius:0, tension:0.3, fill:true}},
    {{label:'Flat Stake',    data:bankrollData.fl, borderColor:'#3b82f6', backgroundColor:'rgba(59,130,246,0.05)', borderWidth:1.5, pointRadius:0, tension:0.3, fill:true, borderDash:[4,3]}},
    {{label:'Starting',      data:Array(bankrollData.labels.length).fill(startingBank), borderColor:'#334155', borderWidth:1, pointRadius:0, borderDash:[6,4], fill:false}},
  ]}},
  options:{{responsive:true,
    plugins:{{legend:{{labels:{{color:'#64748b',font:{{size:10}}}}}}}},
    scales:{{
      x:{{ticks:{{maxTicksLimit:16,color:'#334155'}},grid:{{color:'#0e1420'}}}},
      y:{{ticks:{{color:'#64748b',callback:v=>'GBP'+v.toFixed(0)}},grid:{{color:'#1e2d44'}}}}
    }}
  }}
}});

new Chart(document.getElementById('seasonChart'), {{
  type:'bar',
  data:{{labels:seasonData.seasons, datasets:[{{label:'ROI %', data:seasonData.roi,
    backgroundColor:seasonData.roi.map(v=>v>=0?'rgba(16,185,129,0.7)':'rgba(239,68,68,0.7)'), borderWidth:0}}]}},
  options:{{responsive:true,plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b'}},grid:{{color:'#0e1420'}}}},
      y:{{ticks:{{color:'#64748b',callback:v=>v+'%'}},grid:{{color:'#1e2d44'}}}}
    }}
  }}
}});

new Chart(document.getElementById('leagueCmpChart'), {{
  type:'bar',
  data:{{labels:leagueCmpData.names, datasets:[
    {{label:'Bets', data:leagueCmpData.bets, backgroundColor:'rgba(59,130,246,0.5)', borderWidth:0, yAxisID:'y'}},
    {{label:'ROI %', data:leagueCmpData.roi,
      backgroundColor:leagueCmpData.roi.map(v=>v>=0?'rgba(16,185,129,0.7)':'rgba(239,68,68,0.7)'),
      borderWidth:0, yAxisID:'y2'}},
  ]}},
  options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#64748b',font:{{size:10}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#64748b'}},grid:{{color:'#0e1420'}}}},
      y:{{ticks:{{color:'#3b82f6'}},grid:{{color:'#1e2d44'}},title:{{display:true,text:'Bets',color:'#3b82f6'}}}},
      y2:{{position:'right',ticks:{{color:'#10b981',callback:v=>v+'%'}},grid:{{display:false}},title:{{display:true,text:'ROI %',color:'#10b981'}}}},
    }}
  }}
}});

new Chart(document.getElementById('outcomeChart'), {{
  type:'bar',
  data:{{labels:['Home Win','Draw','Away Win'], datasets:[
    {{label:'Bets', data:outcomeData.bets, backgroundColor:'rgba(59,130,246,0.5)', borderWidth:0}},
    {{label:'Wins', data:outcomeData.wins, backgroundColor:'rgba(16,185,129,0.7)', borderWidth:0}},
  ]}},
  options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#64748b',font:{{size:10}}}}}}}},
    scales:{{x:{{ticks:{{color:'#64748b'}},grid:{{color:'#0e1420'}}}},y:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e2d44'}}}}}}
  }}
}});

new Chart(document.getElementById('edgeChart'), {{
  type:'bar',
  data:{{labels:edgeData.labels, datasets:[{{label:'Bets', data:edgeData.counts, backgroundColor:'rgba(245,158,11,0.6)', borderWidth:0}}]}},
  options:{{responsive:true,plugins:{{legend:{{display:false}}}},
    scales:{{x:{{ticks:{{color:'#64748b'}},grid:{{color:'#0e1420'}}}},y:{{ticks:{{color:'#64748b'}},grid:{{color:'#1e2d44'}}}}}}
  }}
}});

new Chart(document.getElementById('leagueCurveChart'), {{
  type:'line',
  data:{{
    labels:lcLabels.map((v,i)=>i%Math.max(1,Math.floor(lcLabels.length/20))===0?'B'+v:''),
    datasets:[
      ...lcDatasets,
      {{label:'Starting', data:Array(lcLabels.length).fill(startingBank), borderColor:'#334155', borderWidth:1, pointRadius:0, borderDash:[6,4], fill:false}},
    ]
  }},
  options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#64748b',font:{{size:10}}}}}}}},
    scales:{{
      x:{{ticks:{{maxTicksLimit:20,color:'#334155'}},grid:{{color:'#0e1420'}}}},
      y:{{ticks:{{color:'#64748b',callback:v=>'GBP'+v.toFixed(0)}},grid:{{color:'#1e2d44'}}}}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    requested = sys.argv[1:] if len(sys.argv) > 1 else None

    print(f"\nFootball Predictor - Multi-League Backtest")
    print(f"Data root:  {DATA_ROOT}")
    print(f"Output:     {OUTPUT_PATH}")
    print(f"Bankroll:   GBP{STARTING_BANK} | Kelly {int(KELLY_FRACTION*100)}% | "
          f"Cap {int(KELLY_CAP*100)}% | Flat GBP{FLAT_STAKE}/bet")

    if requested:
        print(f"Leagues:    {requested} (specified)")
    else:
        print(f"Leagues:    auto-discover from data/")

    print(f"\nScanning data directory...")
    div_codes = discover_leagues(DATA_ROOT, requested)

    if not div_codes:
        print(f"\nNo qualifying leagues found in {DATA_ROOT}")
        print(f"Each league needs a subfolder (e.g. data/E0/) with at least "
              f"{MIN_SEASONS} CSV files.")
        sys.exit(1)

    print(f"\nLeagues to backtest ({len(div_codes)}):")
    for d in div_codes:
        print(f"  {d:5s} - {LEAGUE_REGISTRY[d].name}")

    all_bets = []
    for div_code in div_codes:
        bets = run_league_backtest(div_code)
        all_bets.extend(bets)

    if not all_bets:
        print("\nNo value bets found across any league.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"TOTAL VALUE BETS: {len(all_bets)} across {len(div_codes)} league(s)")

    print(f"\nSimulating staking...")
    df = simulate_staking(all_bets)

    summary = compute_summary(df)
    print(f"\n--- COMBINED RESULTS ---")
    print(f"  Total bets:    {summary['total_bets']:,}")
    print(f"  Win rate:      {summary['win_rate']}%")
    print(f"  Avg edge:      +{summary['avg_edge']}%")
    print(f"  Quarter Kelly: GBP{summary['qk_profit']:+.2f} profit | "
          f"ROI {summary['qk_roi']:+.2f}% | Final GBP{summary['qk_final']:.2f}")
    print(f"  Flat stake:    GBP{summary['fl_profit']:+.2f} profit | "
          f"ROI {summary['fl_roi']:+.2f}% | Final GBP{summary['fl_final']:.2f}")

    if summary['n_leagues'] > 1:
        print(f"\n--- PER-LEAGUE (flat stake) ---")
        for r in per_league_summary(df):
            print(f"  {r['div_code']:5s} {r['league_name'][:28]:28s} "
                  f"{r['bets']:4d} bets  ROI {r['roi']:+.1f}%  P&L GBP{r['profit']:+.2f}")

    print(f"\nGenerating HTML report...")
    html = render_html(df, summary, div_codes)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved: {OUTPUT_PATH}")
    print(f"\nDone.")
