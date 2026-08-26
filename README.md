# Football Predictor

A multi-league football match outcome forecasting system with a local web dashboard. Combines Dixon-Coles statistical modelling with a Market-Augmented ML layer to identify value betting opportunities against bookmaker odds across 6 top European leagues.

---

## Quick Start

### 1. Install dependencies (one time only)
```bash
pip install -r requirements.txt
```

On Windows, if you see a PowerShell execution policy error when activating your virtual environment:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Add historical data

Download historical season CSVs from football-data.co.uk and place them in the appropriate folder:

| League               | Folder    | football-data.co.uk section  |
|----------------------|-----------|-------------------------------|
| English Premier Lge  | data/E0/  | England > Premiership         |
| Scottish Premiership | data/SC0/ | Scotland > Premiership        |
| German Bundesliga    | data/D1/  | Germany > Bundesliga 1        |
| Italian Serie A      | data/I1/  | Italy > Serie A               |
| Spanish La Liga      | data/SP1/ | Spain > La Liga Premera       |
| French Ligue 1       | data/F1/  | France > Le Championnat       |

Download all available seasons per league (2018 onwards where available). The more historical data, the better calibrated the ML model will be. The EPL 2017/18 season uses a slightly different odds column format (BetBrain aggregates) which is handled automatically.

Other league codes (E1, E2, SC1 etc.) are silently ignored — the system only supports the top 6 leagues listed above.

### 3. Start the server
```bash
python server.py
```

### 4. Open the dashboard
Open http://localhost:5000 in your browser.

### 5. Train models
Click **"Train All Leagues"** in the sidebar. Training runs in parallel across all leagues with data and takes roughly 5–15 minutes total. Progress updates live in the sidebar — each league shows its training status, γ (home advantage multiplier), match count, and season range once complete.

The server must be running for training to work. Do not close the terminal.

### 6. Get predictions

**Updating models (Monday/Thursday after football-data.co.uk refreshes):**
Drop the updated current-season CSV into the relevant `data/{DIVCODE}/` folder. The file watcher detects the change within 30 seconds and automatically retrains that league's model in the background.

**Getting predictions:**
Download the fixtures file from football-data.co.uk (the upcoming fixtures file covering the next few days across all leagues) and drag it directly onto the dashboard drop zone. Predictions appear within a few seconds.

---

## Reading the Dashboard

Each fixture card shows:

- **H/D/A probability bars** — model probability (blue) vs Shin-adjusted market fair probability (grey)
- **Edge %** per outcome — how much the model's probability exceeds the market's fair probability
- **VALUE BET badge** — shown when the adjusted edge clears the threshold (see below)
- **Best bookmaker and odds** for each flagged value bet
- **Confidence rating** (HIGH / MEDIUM / LOW) based on data availability for those teams

### Home win penalty

The model applies a **6 percentage point penalty** to home win edges before flagging value bets. This compensates for a known systematic bias: the Dixon-Coles model overestimates home win probability relative to the market by approximately 6pp, because it was trained on historical data where home advantage was stronger than today. Draw and away win edges are unaffected.

In practice this means:
- A home win is only flagged as value if the model sees **at least 9% edge** (6pp penalty + 3% threshold)
- Away win and draw value bets fire at the standard **3% threshold**
- Genuine high-confidence home value (strong form against a weak visitor that the market has underestimated) still comes through

The `HOME_EDGE_PENALTY` constant in `src/predictor.py` can be adjusted if you want to tune this. Raising it to 0.07–0.08 makes home flagging more selective; lowering it to 0.04–0.05 allows more through.

### Edge threshold

The minimum edge threshold (top-right of dashboard, default 3%) applies after the home penalty. Increasing it to 5% gives a more selective set of bets; the tradeoff is fewer opportunities but higher expected quality.

---

## Workflow

**Twice weekly (after football-data.co.uk updates on Sunday night / Wednesday night):**
1. Download the updated current-season CSV for each league you follow
2. Drop the new file into `data/{DIVCODE}/` — retraining fires automatically

**When you want predictions:**
1. Run `python server.py` (if not already running)
2. Open http://localhost:5000
3. Download the fixtures file from football-data.co.uk
4. Drag it onto the dashboard

---

## Architecture

The prediction pipeline per fixture:

1. **Dixon-Coles** — attack/defence parameters fitted per team using time-decayed MLE → Poisson scoreline convolution → H/D/A baseline probabilities
2. **Rolling form features** — shots, shots on target %, goals scored/conceded, win rate, draw rate, fouls, yellows, reds, corners — computed at per-stat optimal window lengths over the preceding 6–12 matches for both teams
3. **Shin normalisation** — Shin (1993) iterative algorithm applied to fixture odds → fair H/D/A probabilities that remove the bookmaker overround
4. **Market-Augmented ML** — HistGradientBoostingClassifier trained on features 1+2+3, calibrated via 5-fold cross-validated Platt scaling
5. **Market-anchored blend** — final probabilities = α × ML_probs + (1−α) × Shin_fair_probs, where α ≤ 0.5 is learned by minimising Brier Score. The market always has at least equal weight, preventing the ML model from overriding market consensus
6. **Edge** = final probability − Shin fair probability, with home win penalty applied before flagging

**Per-league model artifacts** (stored in `models/`):
- `{DIV}_dc_params.json` — Dixon-Coles parameters (alpha, beta, gamma, rho per team)
- `{DIV}_ml_model.pkl` — Calibrated ML model with learned blend weight α
- `{DIV}_feature_db.pkl` — Latest rolling form features per team
- `{DIV}_metadata.json` — Training summary (match count, seasons, convergence status)

---

## Notes

- Teams not in the training data (newly promoted, first season in dataset) receive league-average Dixon-Coles parameters and are flagged with a warning badge
- The Scottish 2018 CSV has a character encoding issue and is automatically skipped; SC0 trains on remaining seasons without issue
- `converged: false` in DC metadata means the L-BFGS-B optimiser hit its iteration limit — parameters are still usable but may be slightly less stable. Rarely occurs with full data
- The dashboard polls model status every 5 seconds — you can watch training progress live without refreshing
