# Football Match Outcome Forecasting System
## Project Plan & Delivery Roadmap
*Version 2.1 | August 2026*

---

## 1. Project Overview

This project builds a statistical forecasting framework for association football match outcomes across 13 supported European leagues. The core engine models team attack and defence strength using historical match data via the Dixon-Coles method, with outcome probabilities compared against bookmaker implied probabilities to surface value betting opportunities.

Two modelling layers are combined in production: Dixon-Coles provides a stable statistical baseline (Phases 2–3), and a Market-Augmented ML layer exploiting rolling team-level features (Phase 6) refines those probabilities. The final output blends ML predictions with Shin-normalised market fair probabilities, ensuring the market always has at least equal weight in the final number. A 6pp home win penalty compensates for the DC model's known systematic home bias.

The full system is deployed as a local Flask web application (Phase 7) with a drag-and-drop dashboard. The user drops a fixtures CSV from football-data.co.uk, and predictions with bookmaker edge calculations appear within seconds.

The codebase is built with league-agnostic abstractions throughout. A league configuration layer handles schema normalisation, team identifier namespacing, and per-league model separation, meaning all supported leagues run through the same pipeline with no structural code changes.

---

## 2. Objectives

- Build a reproducible outcome forecasting model trained on 13 leagues (EPL, Scottish Prem, Bundesliga, Serie A, La Liga, Ligue 1, plus English League One/Two, Scottish Championship/League One/League Two, and German 2. Bundesliga), extensible to additional leagues
- Design a league-agnostic data pipeline with a configuration layer handling schema differences and team namespacing per league
- Generate per-match win/draw/loss probabilities via Poisson (Dixon-Coles) methods, augmented by a rolling ML feature layer
- Compare model probabilities against bookmaker implied probabilities to identify value bets
- Produce a walk-forward backtesting framework to validate model performance
- Deploy as a local web application with drag-and-drop fixture ingestion and live predictions
- Correct for systematic DC home bias via a calibrated edge penalty

---

## 3. Delivery Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data Exploration & Pipeline | **COMPLETE** |
| 2 | Baseline Model (Dixon-Coles) | **COMPLETE** |
| 3 | Probability Conversion | **COMPLETE** |
| 4 | Backtesting Framework | **COMPLETE** |
| 5 | Value Bet Identification | **COMPLETE** |
| 6 | ML Feature Layer & Model Comparison | **COMPLETE** |
| 7 | Reporting & Automation | **COMPLETE** |

All phases complete. System deployed and validated in production. League scope has since been expanded from the original 6 top-tier leagues to 13 (see Decisions Log, Phase 7 entry).

---

## 4. Phase Detail

### Phase 1 — Data Exploration & Pipeline [COMPLETE]

Eight EPL CSV files spanning 2017/18–2024/25 with varying column structures normalised to a consistent canonical schema. League configuration layer built to handle schema drift, BetBrain→AvgH column bridging for 2018, COVID anomaly flags, and team namespacing.

Key outputs: `league_config.py`, `team_registry.py`, `data_pipeline.py`, `eda_visualisations.py`

Key findings: COVID distortion confirmed (home win rate fell to 37.9% behind closed doors). 2022/23 goals outlier (3.28 avg/game). Zero missing values across 2,810 rows. Schema drift from 62 cols (2018) to 132 cols (2025) fully handled.

---

### Phase 2 — Baseline Model (Dixon-Coles) [COMPLETE]

Dixon-Coles MLE fitted on 2,335 crowd-present matches using time-decay weighting (half-life 133 days). Post-fit identifiability re-centring applied. L2 shrinkage for newly promoted teams.

Key findings: γ = 1.244 (home advantage ~24% more expected goals). ρ = −0.014. Overall Brier Score = 0.613 vs random baseline 0.667. Attack rankings pass face-validity (Man City > Liverpool > Arsenal).

---

### Phase 3 — Probability Conversion [COMPLETE]

Poisson scoreline convolution to H/D/A probabilities. Bookmaker overround retained as explicit signal (mean 4.29%). Draw edge confirmed negative on average (mean −0.023) — no systematic draw value exists.

---

### Phase 4 — Backtesting Framework [COMPLETE]

Walk-forward validation: 3 seasons train, 1 season test, 4 folds (test seasons 2020/21–2023/24). Overall Brier Score 0.591 across 1,332 valid test matches. Flat-stake ROI negative at all thresholds (best: −9.2% at edge >0.03). Dixon-Coles alone cannot beat the bookmaker overround — ML layer required.

---

### Phase 5 — Value Bet Identification [COMPLETE]

Shin (1993) normalisation applied to all walk-forward test matches (mean z = 6.42%). Threshold sweep across 11 levels. Full Kelly goes bankrupt at all thresholds. Best ROI: −9.2% flat stake at 0.03 threshold. No positive ROI from DC alone confirmed.

Per-outcome at edge > 0.03: Home −10.4% ROI, Draw −1.9% ROI, Away −10.6% ROI.

---

### Phase 6 — ML Feature Layer & Model Comparison [COMPLETE]

Market-Augmented ML (HistGradientBoostingClassifier + DC probs + Shin fair probs as features) achieved Brier 0.5814 vs DC baseline 0.5914 — improvement of 0.0100.

Key findings:
- Pure ML (rolling features only) performs worse than DC — market signal is essential
- Window importance increases monotonically: W12 > W9 > W6 > W3
- Per-stat tuned windows: goals_ag → W6, draw_rate → W6, all others → W9
- Tuned hyperparameters all moved toward more regularisation vs defaults (max_depth=3, learning_rate=0.01, min_samples_leaf=50)
- Tuned config saved to phase6_tuned_config.json

---

### Phase 7 — Reporting & Automation [COMPLETE]

**Deployed system:** Local Flask web application at localhost:5000 with drag-and-drop fixtures ingestion, per-league model training with file watcher auto-retrain, and a dark analytics terminal dashboard UI.

**Supported leagues (13 total):** E0 (EPL), SC0 (Scottish Prem), D1 (Bundesliga), I1 (Serie A), SP1 (La Liga), F1 (Ligue 1), E1 (English Championship), E2 (English League One), E3 (English League Two), SC1 (Scottish Championship), SC2 (Scottish League One), SC3 (Scottish League Two), D2 (German 2. Bundesliga). All have shot data and run the full Market-Augmented ML pipeline — the pipeline is fully league-agnostic, so no code changes were needed to add the EFL, Scottish lower-division, or D2 leagues beyond registering them in `league_configs.py`.

**Prediction pipeline:**
1. Dixon-Coles per-league parameters → baseline H/D/A probabilities
2. Rolling form features (shots, SoT%, goals, discipline, corners) at tuned windows
3. Shin normalisation on fixture odds → fair probabilities
4. 5-fold cross-validated ML with Platt scaling calibration
5. Market-anchored blend: α × ML + (1−α) × Shin, where α ≤ 0.5 (learned per league)
6. Home win edge penalty: 6pp subtracted before threshold comparison

**Calibration journey:**
Several calibration issues were encountered and resolved during Phase 7 validation:

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| 40%+ edges on most fixtures | feature_db key mismatch — features not found at prediction time | Fixed get_latest_team_features to store keys without h_/a_ prefix |
| Still inflated edges after fix | ML model trained on old buggy feature_db; old pkl files loaded | Full retrain with corrected feature engineering |
| 22% edges after retrain | Single-season isotonic calibration overfitting on partial ~150-row season | Replaced with 5-fold KFold + Platt (logistic) scaling on OOF predictions |
| 16% edges after Platt scaling | DC systematic home bias flowing through ML into final probabilities | Market-anchored blend capping ML weight at α=0.5 |
| Still home-heavy value bets | DC overestimates home win by ~6pp vs market | HOME_EDGE_PENALTY = 0.06 applied to home edges only |

**Validated output (post all fixes, initial 6-league retrain):**
- 53 fixtures processed; 13 value bets flagged (24%)
- Maximum edge: 7.1% (Bayern Munich away — strong form, weak home side)
- Home win penalty working as intended: only genuinely strong home signals survive
- Away and draw value bets fire at standard 3% threshold

**Production files:**
- `server.py` — Flask server with file watcher and training/prediction API
- `src/league_configs.py` — league definitions (13 leagues)
- `src/model_trainer.py` — Full training pipeline with cross-validated Platt calibration
- `src/predictor.py` — Fixture routing, market-anchored blend, home penalty, edge calculation
- `static/dashboard.html` — Dark analytics terminal UI
- `requirements.txt`, `README.md`

**Repository restructure (August 2026):** The GitHub repository (`MountainSource/LeaguePredictor`) was found to have `league_configs.py`, `model_trainer.py`, and `predictor.py` committed at repo root rather than under `src/`, which `server.py`'s import path expects — this would break a fresh clone with `ModuleNotFoundError`. The repo was reorganised to match the documented `src/`/`static/`/`data/`/`models/` layout. `dashboard.html` was also found to be missing from the repository entirely and must be added from the working local copy before the repo is deployable standalone.

---

## 5. Decisions Log

| Phase | Topic | Decision |
|-------|-------|---------|
| 1 | Output format | Single merged CSV. Per-season CSVs are source data only. |
| 1 | Odds benchmark | AvgH/AvgD/AvgA (market average). BbAvH/D/A for 2017/18 only. |
| 1 | COVID handling | has_crowd flag; DC fitting excludes no-crowd matches. |
| 2 | Time-decay | Exponential, half-life 133 days. |
| 2 | Identifiability | Post-fit log-space re-centring: geometric mean of alpha = 1. |
| 2 | New entrant shrinkage | L2 regularisation, weight = 5.0 / n_observations. |
| 3 | Overround handling | Retain raw implied probs without normalising; margin kept as signal. |
| 3 | Draw value | No systematic draw value (mean edge −0.023). No draw-specific treatment. |
| 4 | Backtesting window | 3 seasons train, 1 test, rolling. 4 folds: test 2020/21–2023/24. |
| 5 | Overround removal | Shin (1993). Mean z = 6.42%. |
| 5 | Staking | Flat stake outperforms Kelly variants in ROI terms. No positive ROI from DC. |
| 6 | Architecture | Market-Augmented ML (rolling + DC + Shin as features). Pure ML worse than DC alone. |
| 6 | Classifier | HistGradientBoostingClassifier (handles NaN natively; XGBoost unavailable). |
| 6 | Calibration | Manual isotonic calibration (sklearn 1.8 removed CalibratedClassifierCV cv=prefit). |
| 6 | Windows | Per-stat tuned: goals_ag→W6, draw_rate→W6, all others→W9. |
| 7 | League scope | Expanded from top 6 to 13: E0, SC0, D1, I1, SP1, F1, E1, E2, E3, SC1, SC2, SC3, D2. Pipeline is fully league-agnostic; expansion required only registry entries, no structural code changes. |
| 7 | Deployment | Flask localhost server. No cloud, no cost. |
| 7 | Calibration method | Platt scaling on 5-fold OOF predictions (isotonic overfits with partial seasons). |
| 7 | Market blend | α × ML + (1−α) × Shin, α ≤ 0.5. Market always has ≥50% weight. |
| 7 | Home bias | HOME_EDGE_PENALTY = 0.06 applied to home win edges only. Tunable constant. |
| 7 | Auto-retrain | File watcher polls data/ every 30s; retrains on CSV change. |
| 7 | Repo layout | `league_configs.py`, `model_trainer.py`, `predictor.py` live under `src/`; `dashboard.html` lives under `static/`. Both required by `server.py`'s import and serving paths. |

---

## 6. Open Decisions / Future Work

| Topic | Notes |
|-------|-------|
| Expand to non-European leagues | Argentina, Brazil, Japan etc. available on football-data.co.uk; same pipeline applies |
| Live odds integration | Currently requires manual CSV download; could automate via odds API |
| HOME_EDGE_PENALTY tuning | Currently set at 0.06 based on Phase 5 backtesting. Review after 1–2 months of live use |
| Season cross-over | Models trained on completed seasons; current partial season included for DC recency only |
| Kelly staking | Do not use until positive ROI is confirmed on a meaningful live sample |
| Lower-league data quality | E3/SC2/SC3 etc. have thinner historical archives than top-flight leagues on football-data.co.uk; walk-forward folds for these leagues may have fewer usable seasons — worth checking `discover_leagues`' MIN_SEASONS=3 threshold is actually met before relying on them |

---

## 7. Key Risks — Final Status

| Risk | Status |
|------|--------|
| Schema inconsistency across seasons | Resolved — league_configs.py normalisation layer |
| COVID distortion | Resolved — has_crowd flag, DC excludes no-crowd matches |
| DC home bias (~6pp overestimate) | Resolved — HOME_EDGE_PENALTY = 0.06 |
| Calibration overfitting on partial seasons | Resolved — Platt scaling on 5-fold OOF |
| ML amplifying DC home bias | Resolved — market-anchored blend, α ≤ 0.5 |
| Feature_db key mismatch | Resolved — stat keys stored/looked up without h_/a_ prefix |
| Slow feature_db construction | Resolved — vectorised ffill+groupby (~0.02s) |
| SC0 2018 encoding issue | Resolved — file skipped automatically |
| Kelly staking ruin | Resolved — full Kelly bankrupt at all thresholds; not used |
| Newly promoted teams absent from training data | Open - league-average fallback used with warning flag |
| H2H features sparse for newly promoted sides | Open - minimum 5 meetings threshold; NaN passed through to model |
| Draw value does not exist systematically | Resolved - confirmed negative mean draw edge; no systematic draw treatment |
| README/plan understated league scope (said "top 6" while registry has 13) | Resolved — README and this plan updated to reflect all 13 leagues |
| GitHub repo missing src/ folder structure, breaking fresh-clone imports | Resolved — repo reorganised to match documented layout |
| dashboard.html absent from GitHub repo | Open — must be copied from local working copy into static/ before repo is standalone-deployable |


---

*Last updated: August 2026 — All 7 phases complete, and repository structure corrected to match documented layout. League scope confirmed at 13 (not 6) across README, this plan, and league_configs.py. dashboard.html still needs to be added to the repo from the local working copy.*
