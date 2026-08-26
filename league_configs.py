"""
league_configs.py
-----------------
League configuration for all supported leagues.

All leagues are sourced from football-data.co.uk and share the same
canonical column structure (2019+ format: AvgH/AvgD/AvgA, shot data present).
The STANDARD_COLUMNS mapping handles this shared structure. Per-league overrides
are only needed for the 2017/18 EPL season which uses BetBrain aggregates.

Div codes match the 'Div' column in football-data.co.uk fixture files, enabling
automatic league routing when a multi-league fixtures file is uploaded.

Supported leagues (13 total):
  Top-tier:    E0, SC0, D1, I1, SP1, F1
  English EFL: E1 (Championship), E2 (League One), E3 (League Two)
  Scottish:    SC1 (Championship), SC2 (League One), SC3 (League Two)
  German:      D2 (2. Bundesliga)

Shot data availability:
  All supported leagues include HS, AS, HST, AST — full Market-Augmented
  ML model applies across the board.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LeagueConfig:
    """Configuration for a single supported league."""
    div_code: str           # Matches 'Div' column value in football-data.co.uk files
    name: str               # Human-readable name
    namespace: str          # Team ID prefix, e.g. 'EPL' -> 'EPL::Arsenal'
    has_shots: bool         # Whether HS/AS/HST/AST columns are expected
    # Seasons whose files may use non-standard column names (e.g. EPL 2017/18)
    # Maps season_label -> {canonical_col: actual_csv_col}
    season_column_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Seasons to exclude from home-advantage fitting (COVID etc.)
    # List of season labels where has_crowd = False
    no_crowd_seasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# STANDARD COLUMN MAP — applies to all leagues, all seasons (2019+ format)
# ---------------------------------------------------------------------------
STANDARD_COLUMNS = {
    'date':      'Date',
    'home_team': 'HomeTeam',
    'away_team': 'AwayTeam',
    'fthg':      'FTHG',
    'ftag':      'FTAG',
    'ftr':       'FTR',
    'hthg':      'HTHG',
    'htag':      'HTAG',
    'htr':       'HTR',
    'hs':        'HS',
    'as_':       'AS',
    'hst':       'HST',
    'ast':       'AST',
    'hf':        'HF',
    'af':        'AF',
    'hc':        'HC',
    'ac':        'AC',
    'hy':        'HY',
    'ay':        'AY',
    'hr':        'HR',
    'ar':        'AR',
    # Market average odds — primary benchmark for edge calculation
    'avg_h':     'AvgH',
    'avg_d':     'AvgD',
    'avg_a':     'AvgA',
    # Bet365 — present in all seasons; fallback if AvgH missing
    'b365_h':    'B365H',
    'b365_d':    'B365D',
    'b365_a':    'B365A',
    # Pinnacle — sharp market, useful signal
    'ps_h':      'PSH',
    'ps_d':      'PSD',
    'ps_a':      'PSA',
}

# EPL 2017/18 uses BetBrain aggregates instead of AvgH/AvgD/AvgA
# These are mapped to the canonical avg_h/d/a names during normalisation
_EPL_2018_OVERRIDES = {
    'avg_h': 'BbAvH',
    'avg_d': 'BbAvD',
    'avg_a': 'BbAvA',
}

# ---------------------------------------------------------------------------
# LEAGUE REGISTRY — add new leagues here, no structural code changes needed
# ---------------------------------------------------------------------------
LEAGUE_REGISTRY: Dict[str, LeagueConfig] = {

    'E0': LeagueConfig(
        div_code='E0',
        name='English Premier League',
        namespace='EPL',
        has_shots=True,
        season_column_overrides={
            '2017/18': _EPL_2018_OVERRIDES,
        },
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'SC0': LeagueConfig(
        div_code='SC0',
        name='Scottish Premiership',
        namespace='SCO',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'D1': LeagueConfig(
        div_code='D1',
        name='German Bundesliga',
        namespace='BUN',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'I1': LeagueConfig(
        div_code='I1',
        name='Italian Serie A',
        namespace='ITA',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'SP1': LeagueConfig(
        div_code='SP1',
        name='Spanish La Liga',
        namespace='SPA',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'F1': LeagueConfig(
        div_code='F1',
        name='French Ligue 1',
        namespace='FRA',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    # -----------------------------------------------------------------------
    # ENGLISH EFL
    # -----------------------------------------------------------------------

    'E1': LeagueConfig(
        div_code='E1',
        name='English Championship',
        namespace='CHMP',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'E2': LeagueConfig(
        div_code='E2',
        name='English League One',
        namespace='LGO',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'E3': LeagueConfig(
        div_code='E3',
        name='English League Two',
        namespace='LGT',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    # -----------------------------------------------------------------------
    # SCOTTISH FOOTBALL LEAGUE
    # -----------------------------------------------------------------------

    'SC1': LeagueConfig(
        div_code='SC1',
        name='Scottish Championship',
        namespace='SC1',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'SC2': LeagueConfig(
        div_code='SC2',
        name='Scottish League One',
        namespace='SC2',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    'SC3': LeagueConfig(
        div_code='SC3',
        name='Scottish League Two',
        namespace='SC3',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),

    # -----------------------------------------------------------------------
    # GERMAN
    # -----------------------------------------------------------------------

    'D2': LeagueConfig(
        div_code='D2',
        name='German 2. Bundesliga',
        namespace='BUN2',
        has_shots=True,
        no_crowd_seasons=['2019/20', '2020/21'],
    ),
}


def get_league(div_code: str) -> Optional[LeagueConfig]:
    """Return LeagueConfig for a Div code, or None if not supported."""
    return LEAGUE_REGISTRY.get(div_code)


def supported_div_codes() -> List[str]:
    """Return list of all supported Div codes."""
    return list(LEAGUE_REGISTRY.keys())
