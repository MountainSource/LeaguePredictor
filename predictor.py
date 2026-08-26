"""
predictor.py
-------------
Prediction engine for upcoming fixtures.

Given a fixtures CSV (football-data.co.uk format) and a set of trained
league models, this module:

  1. Routes each fixture to its correct league model via the 'Div' column
  2. Looks up the latest rolling form features for each team
  3. Runs Dixon-Coles to get baseline H/D/A probabilities
  4. Applies Shin normalisation to the fixture odds to get fair probs
  5. Runs the Market-Augmented ML model (if available) for final probabilities
  6. Computes edges: model_prob - shin_fair_prob for each outcome
  7. Flags value bets where any edge exceeds the configured threshold

FALLBACK BEHAVIOUR:
  - If a league has no trained model: fixture is skipped with status 'no_model'
  - If a team is not in the DC model (newly promoted etc.): uses league-average
    parameters with a warning flag
  - If ML model is unavailable (insufficient training data): falls back to DC only
  - If odds are missing from the fixture: edge cannot be computed, status 'no_odds'

OUTPUT:
  List of prediction dicts, one per fixture, suitable for JSON serialisation
  and direct consumption by the dashboard frontend.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from league_configs import LEAGUE_REGISTRY, get_league
from model_trainer import (
    dc_predict_proba, shin_normalise, load_league_model,
    build_feature_cols, DEFAULT_WINDOW_CONFIG, DEFAULT_EDGE_THRESHOLD,
)

# Confidence tiers — based on Brier-calibrated uncertainty
# (heuristic: how many reliable features we have for this match)
CONFIDENCE_HIGH   = 'HIGH'
CONFIDENCE_MEDIUM = 'MEDIUM'
CONFIDENCE_LOW    = 'LOW'

# Home win edge penalty — subtracted from home win edge before value bet flagging.
# Compensates for the systematic DC home bias (~6pp overestimation of home win
# probability vs market, confirmed in Phase 5/6 backtesting). Draw and away edges
# are unaffected. A home win is only flagged as value if the raw edge exceeds
# edge_threshold + HOME_EDGE_PENALTY, i.e. it must clear a meaningfully higher bar.
HOME_EDGE_PENALTY = 0.06


def predict_fixtures(fixtures_path: str,
                     models_dir: str,
                     edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
                     window_config: Optional[Dict] = None) -> Dict:
    """
    Main entry point: process a fixtures file and return predictions.

    Parameters
    ----------
    fixtures_path : path to fixtures CSV (football-data.co.uk format)
    models_dir    : directory containing trained model artifacts
    edge_threshold: minimum model_prob - shin_fair_prob to flag as value bet
    window_config : rolling window config (uses default if None)

    Returns
    -------
    dict with keys:
      'predictions'  : list of per-fixture prediction dicts
      'summary'      : high-level summary stats
      'errors'       : list of error messages for skipped fixtures
    """
    if window_config is None:
        window_config = DEFAULT_WINDOW_CONFIG

    # Load fixtures
    try:
        fixtures = pd.read_csv(fixtures_path, low_memory=False)
    except Exception as e:
        return {'predictions': [], 'summary': {}, 'errors': [f"Could not read fixtures file: {e}"]}

    if 'Div' not in fixtures.columns:
        return {'predictions': [], 'summary': {},
                'errors': ["Fixtures file has no 'Div' column — cannot route to league models"]}

    predictions = []
    errors = []
    model_cache = {}

    for _, row in fixtures.iterrows():
        div_code   = str(row.get('Div', '')).strip()
        home_team  = str(row.get('HomeTeam', '')).strip()
        away_team  = str(row.get('AwayTeam', '')).strip()
        match_date = str(row.get('Date', '')).strip()
        match_time = str(row.get('Time', '')).strip()

        if not div_code or not home_team or not away_team:
            errors.append(f"Skipping row with missing Div/HomeTeam/AwayTeam")
            continue

        # Check league support
        league_cfg = get_league(div_code)
        if league_cfg is None:
            # Silently skip unsupported leagues (not in top 6)
            continue

        # Load model (cached per league)
        if div_code not in model_cache:
            try:
                model_cache[div_code] = load_league_model(models_dir, div_code)
            except FileNotFoundError:
                errors.append(f"{div_code}: No trained model found. "
                              f"Add historical CSV files to data/{div_code}/ and retrain.")
                model_cache[div_code] = None

        league_model = model_cache[div_code]
        if league_model is None:
            predictions.append(_no_model_prediction(
                div_code, league_cfg, home_team, away_team, match_date, match_time
            ))
            continue

        # Namespace team names
        ns = league_cfg.namespace
        ht_ns = f"{ns}::{home_team}"
        at_ns = f"{ns}::{away_team}"

        # Get odds
        odds_h = _safe_float(row.get('AvgH'))
        odds_d = _safe_float(row.get('AvgD'))
        odds_a = _safe_float(row.get('AvgA'))

        # Fallback to B365 if AvgH missing
        if odds_h is None:
            odds_h = _safe_float(row.get('B365H'))
            odds_d = _safe_float(row.get('B365D'))
            odds_a = _safe_float(row.get('B365A'))

        # Compute prediction
        pred = _predict_single(
            div_code=div_code,
            league_name=league_cfg.name,
            home_team=home_team,
            away_team=away_team,
            home_team_ns=ht_ns,
            away_team_ns=at_ns,
            match_date=match_date,
            match_time=match_time,
            odds_h=odds_h,
            odds_d=odds_d,
            odds_a=odds_a,
            league_model=league_model,
            window_config=window_config,
            edge_threshold=edge_threshold,
            row=row,
        )
        predictions.append(pred)

    # Build summary
    n_predicted = sum(1 for p in predictions if p['status'] == 'ok')
    n_value_bets = sum(1 for p in predictions if p.get('has_value_bet', False))
    top_edge = max((p.get('max_edge', 0) for p in predictions if p['status'] == 'ok'),
                   default=0.0)

    leagues_covered = sorted(set(
        p['league_name'] for p in predictions if p['status'] == 'ok'
    ))

    summary = {
        'total_fixtures':  len(predictions),
        'predicted':       n_predicted,
        'value_bets':      n_value_bets,
        'top_edge':        round(top_edge, 4),
        'leagues_covered': leagues_covered,
        'edge_threshold':  edge_threshold,
    }

    # Sort: value bets first, then by max edge descending
    predictions.sort(key=lambda p: (
        -int(p.get('has_value_bet', False)),
        -p.get('max_edge', -999)
    ))

    return {
        'predictions': predictions,
        'summary':     summary,
        'errors':      errors,
    }


def _predict_single(div_code, league_name, home_team, away_team,
                    home_team_ns, away_team_ns,
                    match_date, match_time,
                    odds_h, odds_d, odds_a,
                    league_model, window_config,
                    edge_threshold, row) -> Dict:
    """Compute full prediction for a single fixture."""

    dc_params  = league_model['dc_params']
    ml_model   = league_model['ml_model']
    feature_db = league_model['feature_db']

    # --- DC probabilities ---
    dc_probs = dc_predict_proba(home_team_ns, away_team_ns, dc_params)
    team_known = dc_probs is not None

    if not team_known:
        # Unknown team: use league-average (gamma * mean_alpha * mean_beta)
        mean_alpha = np.mean(list(dc_params['alpha'].values()))
        mean_beta  = np.mean(list(dc_params['beta'].values()))
        pseudo_dc  = dict(dc_params)
        pseudo_dc['alpha'][home_team_ns] = mean_alpha
        pseudo_dc['alpha'][away_team_ns] = mean_alpha
        pseudo_dc['beta'][home_team_ns]  = mean_beta
        pseudo_dc['beta'][away_team_ns]  = mean_beta
        dc_probs = dc_predict_proba(home_team_ns, away_team_ns, pseudo_dc)

    dc_probs = np.array(dc_probs) if dc_probs is not None else np.array([1/3, 1/3, 1/3])

    # --- Shin fair probabilities from odds ---
    shin_probs = None
    if odds_h and odds_d and odds_a:
        try:
            sh, sd, sa = shin_normalise(odds_h, odds_d, odds_a)
            shin_probs = np.array([sh, sd, sa])
        except Exception:
            pass

    # --- Rolling features for ML ---
    h_feats = feature_db.get(home_team_ns, {})
    a_feats = feature_db.get(away_team_ns, {})
    n_h_feats = len(h_feats)
    n_a_feats = len(a_feats)
    feature_coverage = (n_h_feats + n_a_feats) / (2 * len(window_config))
    if n_h_feats == 0 or n_a_feats == 0:
        print(f"  [WARN] Sparse features: {home_team} h={n_h_feats} | {away_team} a={n_a_feats} "
              f"(ns: {home_team_ns} / {away_team_ns})")

    # --- ML probabilities (Market-Augmented) ---
    ml_probs = None
    if ml_model is not None and shin_probs is not None:
        try:
            feat_vec = _build_live_feature_vector(
                home_team_ns, away_team_ns,
                h_feats, a_feats,
                dc_probs, shin_probs,
                ml_model.feat_cols, window_config
            )
            feat_2d = feat_vec.reshape(1, -1)
            ml_probs_raw = ml_model.predict_proba(feat_2d)[0]
            ml_probs = ml_probs_raw  # [p_H, p_D, p_A]
        except Exception:
            ml_probs = None

    # Final model probabilities: market-anchored blend if ML available, else DC
    # alpha = learned weight for ML; (1-alpha) = weight for Shin market probs
    # Alpha is constrained to [0, 0.5] during training so market always dominates
    if ml_probs is not None and shin_probs is not None:
        alpha = getattr(ml_model, 'alpha', 0.5)
        blended = alpha * ml_probs + (1.0 - alpha) * shin_probs
        # Renormalise
        blended = blended / blended.sum()
        final_probs = blended
        model_source = f'ML+MKT(a={alpha:.2f})'
    elif ml_probs is not None:
        final_probs = ml_probs
        model_source = 'ML'
    else:
        final_probs = dc_probs
        model_source = 'DC'

    # --- Edge calculation ---
    outcomes = ['H', 'D', 'A']
    outcome_labels = ['Home', 'Draw', 'Away']
    edges = {}
    value_bets = []

    if shin_probs is not None:
        for i, outcome in enumerate(outcomes):
            edge = float(final_probs[i]) - float(shin_probs[i])
            edges[outcome] = round(edge, 4)

            # Apply home bias penalty to home win edges only.
            # The displayed edge is the honest model-vs-market gap; the penalty
            # only affects whether the value bet threshold is crossed.
            penalty = HOME_EDGE_PENALTY if outcome == 'H' else 0.0
            adjusted_edge = edge - penalty

            if adjusted_edge >= edge_threshold:
                best_odds, best_bookie = _best_odds_for_outcome(row, i)
                value_bets.append({
                    'outcome':      outcome_labels[i],
                    'outcome_code': outcome,
                    'model_prob':   round(float(final_probs[i]), 4),
                    'fair_prob':    round(float(shin_probs[i]), 4),
                    'edge':         round(edge, 4),
                    'adjusted_edge': round(adjusted_edge, 4),
                    'edge_pct':     f"+{round(edge * 100, 1)}%",
                    'adj_edge_pct': f"+{round(adjusted_edge * 100, 1)}% (adj)",
                    'best_odds':    best_odds,
                    'best_bookie':  best_bookie,
                    'implied_prob': round(1.0 / best_odds, 4) if best_odds else None,
                    'home_penalty_applied': outcome == 'H',
                })

    max_edge = max(edges.values(), default=0.0) if edges else 0.0

    # --- Confidence ---
    confidence = _assess_confidence(team_known, feature_coverage, ml_probs)

    return {
        'status':        'ok',
        'div_code':      div_code,
        'league_name':   league_name,
        'home_team':     home_team,
        'away_team':     away_team,
        'date':          match_date,
        'time':          match_time,
        'model_source':  model_source,
        'confidence':    confidence,
        'team_known':    team_known,
        # Probabilities
        'prob_h':        round(float(final_probs[0]), 4),
        'prob_d':        round(float(final_probs[1]), 4),
        'prob_a':        round(float(final_probs[2]), 4),
        'dc_prob_h':     round(float(dc_probs[0]), 4),
        'dc_prob_d':     round(float(dc_probs[1]), 4),
        'dc_prob_a':     round(float(dc_probs[2]), 4),
        'shin_h':        round(float(shin_probs[0]), 4) if shin_probs is not None else None,
        'shin_d':        round(float(shin_probs[1]), 4) if shin_probs is not None else None,
        'shin_a':        round(float(shin_probs[2]), 4) if shin_probs is not None else None,
        # Odds
        'odds_h':        odds_h,
        'odds_d':        odds_d,
        'odds_a':        odds_a,
        # Edges
        'edges':         edges,
        'max_edge':      round(max_edge, 4),
        'has_value_bet': len(value_bets) > 0,
        'value_bets':    value_bets,
    }


def _build_live_feature_vector(home_team_ns: str, away_team_ns: str,
                                h_feats: Dict, a_feats: Dict,
                                dc_probs: np.ndarray, shin_probs: np.ndarray,
                                feat_cols: List[str],
                                window_config: Dict) -> np.ndarray:
    """
    Build a single feature vector for a live (upcoming) fixture.
    Mirrors the column layout expected by the trained ML model.
    """
    vec = []
    for col in feat_cols:
        if col.startswith('h_') and '_w' in col:
            stat = col[2:]  # e.g. 'goals_for_w9'
            vec.append(h_feats.get(stat, np.nan))
        elif col.startswith('a_') and '_w' in col:
            stat = col[2:]
            vec.append(a_feats.get(stat, np.nan))
        elif col.startswith('diff_') and '_w' in col:
            stat = col[5:]  # e.g. 'goals_for_w9'
            h_val = h_feats.get(stat, np.nan)
            a_val = a_feats.get(stat, np.nan)
            if not np.isnan(h_val) and not np.isnan(a_val):
                vec.append(h_val - a_val)
            else:
                vec.append(np.nan)
        elif col == 'h2h_h_win':
            vec.append(np.nan)   # Not computed in live mode (no prior meetings lookup)
        elif col == 'h2h_draw':
            vec.append(np.nan)
        elif col == 'h2h_h_loss':
            vec.append(np.nan)
        elif col == 'dc_prob_h':
            vec.append(float(dc_probs[0]))
        elif col == 'dc_prob_d':
            vec.append(float(dc_probs[1]))
        elif col == 'dc_prob_a':
            vec.append(float(dc_probs[2]))
        elif col == 'shin_h':
            vec.append(float(shin_probs[0]))
        elif col == 'shin_d':
            vec.append(float(shin_probs[1]))
        elif col == 'shin_a':
            vec.append(float(shin_probs[2]))
        else:
            vec.append(np.nan)

    return np.array(vec, dtype=float)


def _best_odds_for_outcome(row, outcome_idx: int) -> Tuple[Optional[float], Optional[str]]:
    """
    Find the best (highest) available odds for a given outcome across all bookmakers.
    outcome_idx: 0=Home, 1=Draw, 2=Away
    """
    suffix = ['H', 'D', 'A'][outcome_idx]
    bookmakers = {
        'B365':  f'B365{suffix}',
        'BFD':   f'BFD{suffix}',
        'BMGM':  f'BMGM{suffix}',
        'BV':    f'BV{suffix}',
        'BW':    f'BW{suffix}',
        'CL':    f'CL{suffix}',
        'LB':    f'LB{suffix}',
        'PS':    f'PS{suffix}',
        'BFE':   f'BFE{suffix}',
        'Max':   f'Max{suffix}',
    }

    best_odds = None
    best_bookie = None
    for bookie, col in bookmakers.items():
        val = _safe_float(row.get(col))
        if val and val > 1.0:
            if best_odds is None or val > best_odds:
                best_odds = val
                best_bookie = bookie

    return best_odds, best_bookie


def _assess_confidence(team_known: bool, feature_coverage: float,
                        ml_probs) -> str:
    """
    Heuristic confidence assessment based on data availability.
    HIGH: both teams known, good feature coverage, ML available
    MEDIUM: teams known but limited features or DC-only
    LOW: unknown team(s) or very sparse features
    """
    if not team_known:
        return CONFIDENCE_LOW
    if ml_probs is not None and feature_coverage > 0.6:
        return CONFIDENCE_HIGH
    if feature_coverage > 0.3:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _no_model_prediction(div_code, league_cfg, home_team, away_team,
                          match_date, match_time) -> Dict:
    return {
        'status':      'no_model',
        'div_code':    div_code,
        'league_name': league_cfg.name,
        'home_team':   home_team,
        'away_team':   away_team,
        'date':        match_date,
        'time':        match_time,
        'message':     f"No trained model for {div_code}. Add data and retrain.",
        'has_value_bet': False,
        'max_edge':    0.0,
    }


def _safe_float(val) -> Optional[float]:
    """Convert to float, return None if missing or invalid."""
    try:
        f = float(val)
        return f if not np.isnan(f) and f > 0 else None
    except (TypeError, ValueError):
        return None
