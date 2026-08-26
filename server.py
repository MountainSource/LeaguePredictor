"""
server.py
----------
Football Predictor — local Flask server.

Run with: python server.py
Then open: http://localhost:5000

ENDPOINTS:
  GET  /                    → serves dashboard.html
  GET  /api/status          → model training status for all 6 leagues
  POST /api/predict         → accepts fixtures CSV upload, returns predictions
  POST /api/train           → trigger (re)training for one or all leagues
  GET  /api/leagues         → list supported leagues and their data status

FILE WATCHER:
  A background thread polls the data/ subdirectories every 30 seconds.
  If a new or modified CSV is detected in data/{DIVCODE}/, the corresponding
  league model is automatically queued for retraining.
  This fires asynchronously — the API returns immediately and the dashboard
  polls /api/status to show progress.
"""

import os
import sys
import json
import time
import pickle
import hashlib
import threading
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(BASE_DIR, 'src')
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
STATIC_DIR  = os.path.join(BASE_DIR, 'static')

sys.path.insert(0, SRC_DIR)

from league_configs import LEAGUE_REGISTRY, get_league
from model_trainer import train_league, DEFAULT_WINDOW_CONFIG, DEFAULT_HPARAMS
from predictor import predict_fixtures

# ---------------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=STATIC_DIR)

# ---------------------------------------------------------------------------
# TRAINING STATE
# ---------------------------------------------------------------------------
# Thread-safe state tracking for each league
training_state = {}   # div_code -> {'status': 'idle'|'training'|'done'|'error', 'message': str}
training_lock  = threading.Lock()

def _get_league_data_dir(div_code: str) -> str:
    return os.path.join(DATA_DIR, div_code)


def _has_data(div_code: str) -> bool:
    data_dir = _get_league_data_dir(div_code)
    if not os.path.isdir(data_dir):
        return False
    return any(f.lower().endswith('.csv') for f in os.listdir(data_dir))


def _is_trained(div_code: str) -> bool:
    meta_path = os.path.join(MODELS_DIR, f'{div_code}_metadata.json')
    return os.path.exists(meta_path)


def _load_metadata(div_code: str) -> dict:
    meta_path = os.path.join(MODELS_DIR, f'{div_code}_metadata.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _run_training(div_code: str):
    """Run training for a single league in a background thread."""
    with training_lock:
        training_state[div_code] = {'status': 'training', 'message': 'Training in progress...'}

    try:
        league_cfg = get_league(div_code)
        if league_cfg is None:
            raise ValueError(f"Unknown league code: {div_code}")

        data_dir = _get_league_data_dir(div_code)
        os.makedirs(MODELS_DIR, exist_ok=True)

        metadata = train_league(
            data_dir=data_dir,
            models_dir=MODELS_DIR,
            div_code=div_code,
            league_config=league_cfg,
            window_config=DEFAULT_WINDOW_CONFIG,
            hparams=DEFAULT_HPARAMS,
            verbose=True,
        )

        with training_lock:
            training_state[div_code] = {
                'status':  'done',
                'message': f"Trained on {metadata['n_matches']} matches across {len(metadata['seasons'])} seasons",
                'trained_at': metadata['trained_at'],
            }
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training complete: {div_code}")

    except Exception as e:
        with training_lock:
            training_state[div_code] = {'status': 'error', 'message': str(e)}
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training FAILED for {div_code}: {e}")


def _dir_fingerprint(data_dir: str) -> str:
    """Compute a hash representing the current state of all CSVs in a directory."""
    if not os.path.isdir(data_dir):
        return ''
    parts = []
    for f in sorted(os.listdir(data_dir)):
        if f.lower().endswith('.csv'):
            fpath = os.path.join(data_dir, f)
            stat  = os.stat(fpath)
            parts.append(f"{f}:{stat.st_size}:{stat.st_mtime:.0f}")
    return hashlib.md5('|'.join(parts).encode()).hexdigest() if parts else ''


# ---------------------------------------------------------------------------
# FILE WATCHER THREAD
# ---------------------------------------------------------------------------
_watcher_fingerprints = {}   # div_code -> fingerprint str

def _file_watcher():
    """
    Background thread: polls data/ subdirectories every 30 seconds.
    Triggers retraining when a CSV file is added or modified.
    """
    global _watcher_fingerprints
    poll_interval = 30  # seconds

    print(f"[FileWatcher] Started — polling data/ every {poll_interval}s")

    while True:
        time.sleep(poll_interval)
        for div_code in LEAGUE_REGISTRY:
            data_dir    = _get_league_data_dir(div_code)
            fingerprint = _dir_fingerprint(data_dir)

            prev = _watcher_fingerprints.get(div_code, '')
            if fingerprint and fingerprint != prev:
                _watcher_fingerprints[div_code] = fingerprint
                # Check if not already training
                with training_lock:
                    current = training_state.get(div_code, {}).get('status', 'idle')
                if current != 'training':
                    print(f"[FileWatcher] Change detected in data/{div_code}/ — queuing retrain")
                    t = threading.Thread(target=_run_training, args=(div_code,), daemon=True)
                    t.start()

    
# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'dashboard.html')


@app.route('/api/status')
def api_status():
    """Return training status and metadata for all 6 leagues."""
    result = {}
    for div_code, league_cfg in LEAGUE_REGISTRY.items():
        has_data  = _has_data(div_code)
        is_trained = _is_trained(div_code)
        metadata  = _load_metadata(div_code) if is_trained else {}

        with training_lock:
            train_status = training_state.get(div_code, {})

        result[div_code] = {
            'div_code':     div_code,
            'league_name':  league_cfg.name,
            'namespace':    league_cfg.namespace,
            'has_data':     has_data,
            'is_trained':   is_trained,
            'train_status': train_status.get('status', 'idle'),
            'train_message': train_status.get('message', ''),
            'n_matches':    metadata.get('n_matches', 0),
            'n_teams':      metadata.get('n_teams', 0),
            'seasons':      metadata.get('seasons', []),
            'has_ml_model': metadata.get('has_ml_model', False),
            'trained_at':   metadata.get('trained_at', ''),
            'dc_gamma':     metadata.get('dc_gamma', None),
        }

    return jsonify(result)


@app.route('/api/train', methods=['POST'])
def api_train():
    """
    Trigger (re)training for one or all leagues.
    Request body: {"div_code": "E0"} or {"div_code": "ALL"}
    """
    body     = request.get_json(silent=True) or {}
    div_code = body.get('div_code', 'ALL').upper()

    if div_code == 'ALL':
        queued = []
        for code in LEAGUE_REGISTRY:
            if _has_data(code):
                with training_lock:
                    current = training_state.get(code, {}).get('status', 'idle')
                if current != 'training':
                    t = threading.Thread(target=_run_training, args=(code,), daemon=True)
                    t.start()
                    queued.append(code)
        return jsonify({'status': 'queued', 'leagues': queued})

    elif div_code in LEAGUE_REGISTRY:
        if not _has_data(div_code):
            return jsonify({'status': 'error',
                            'message': f'No data found in data/{div_code}/'}), 400
        with training_lock:
            current = training_state.get(div_code, {}).get('status', 'idle')
        if current == 'training':
            return jsonify({'status': 'already_training', 'div_code': div_code})
        t = threading.Thread(target=_run_training, args=(div_code,), daemon=True)
        t.start()
        return jsonify({'status': 'queued', 'leagues': [div_code]})

    else:
        return jsonify({'status': 'error', 'message': f'Unknown league: {div_code}'}), 400


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Accept a fixtures CSV upload and return predictions.
    Form field: 'file' (multipart/form-data)
    Optional query param: edge_threshold (default 0.03)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send file as multipart/form-data field "file"'}), 400

    uploaded = request.files['file']
    if not uploaded.filename:
        return jsonify({'error': 'Empty filename'}), 400

    edge_threshold = float(request.args.get('edge_threshold', 0.03))

    # Save to temp file
    suffix = '.csv' if uploaded.filename.lower().endswith('.csv') else '.xlsx'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_fixtures(
            fixtures_path=tmp_path,
            models_dir=MODELS_DIR,
            edge_threshold=edge_threshold,
        )
    finally:
        os.unlink(tmp_path)

    return jsonify(result)


@app.route('/api/leagues')
def api_leagues():
    """Return list of supported leagues with data folder status."""
    leagues = []
    for div_code, cfg in LEAGUE_REGISTRY.items():
        data_dir = _get_league_data_dir(div_code)
        csv_files = []
        if os.path.isdir(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.csv')]

        leagues.append({
            'div_code':    div_code,
            'name':        cfg.name,
            'namespace':   cfg.namespace,
            'data_dir':    f'data/{div_code}/',
            'csv_files':   sorted(csv_files),
            'n_files':     len(csv_files),
        })
    return jsonify(leagues)


# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Create directories
    for div_code in LEAGUE_REGISTRY:
        os.makedirs(_get_league_data_dir(div_code), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Initialise watcher fingerprints
    for div_code in LEAGUE_REGISTRY:
        _watcher_fingerprints[div_code] = _dir_fingerprint(_get_league_data_dir(div_code))

    # Start file watcher thread
    watcher_thread = threading.Thread(target=_file_watcher, daemon=True)
    watcher_thread.start()

    # Print startup summary
    print("\n" + "="*60)
    print("  FOOTBALL PREDICTOR — Local Server")
    print("="*60)
    print(f"  Dashboard:  http://localhost:5000")
    print(f"  Data dir:   {DATA_DIR}")
    print(f"  Models dir: {MODELS_DIR}")
    print()

    for div_code, cfg in LEAGUE_REGISTRY.items():
        has_data   = _has_data(div_code)
        is_trained = _is_trained(div_code)
        data_status = "✓ data" if has_data else "  no data"
        model_status = " | ✓ model" if is_trained else " |   not trained"
        print(f"  {div_code:5s} {cfg.name:30s} {data_status}{model_status}")

    print()
    print("  Drop season CSVs into data/{DIVCODE}/ to trigger retraining.")
    print("  Use the dashboard to upload fixtures and get predictions.")
    print("="*60 + "\n")

    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
