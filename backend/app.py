"""
app.py
Simple Flask API exposing metrics derived from the eCFR.  The API exposes
two endpoints: ``/api/update`` triggers a refresh of the dataset by
downloading and processing current eCFR content, while ``/api/summary``
returns the in‑memory dataset (loading from disk on first access).  A CORS
header is added to all responses so that the JavaScript front‑end can call
these endpoints from a browser without running into cross‑origin issues.

Usage:

    # install dependencies
    pip install flask flask_cors requests

    # run the server
    python app.py

The server will download and cache the metrics the first time
``/api/summary`` is requested or whenever ``/api/update`` is called.

"""

from __future__ import annotations

import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from ecfr_analyzer.backend import ecfr

import signal
import sys

app = Flask(__name__)
CORS(app)

_cache: list[dict] | None = None
# Paths for persistent files.  The data file holds the computed metrics; the
# two log files provide insight into the progress and any errors that occur
# during updates.  They live alongside the package root so they persist
# across runs.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
PROCESS_LOG = os.path.join(BASE_DIR, "process.log")
ERROR_LOG = os.path.join(BASE_DIR, "error.log")

# Track whether an update operation is currently underway.  This flag
# enables us to record a message when the process exits unexpectedly
# mid‑update.  Both the signal handler and an atexit hook check this
# flag before logging to avoid spurious messages when the server
# terminates normally.
_update_in_progress: bool = False

# Register signal handlers so that if the process is interrupted (Ctrl+C or kill)
# a message is appended to the error log before exiting.  Without this the
# application would exit silently and any partial progress might be lost.
def _handle_exit(signum, frame):
    """Signal handler that records termination and exits.

    When the process receives SIGINT or SIGTERM we append a message to
    the error log.  If an update is currently running we also note
    that partial progress may have been lost.  Finally, we exit the
    process which triggers any atexit hooks.
    """
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            if _update_in_progress:
                f.write(f"Process terminated by signal {signum} during update\n")
            else:
                f.write(f"Process terminated by signal {signum}\n")
    except Exception:
        pass
    # gracefully exit
    sys.exit(0)

# Use an atexit hook to capture cases where the interpreter exits
# without raising a signal (for example, when the host process is
# shutdown externally).  If an update was in progress we record a
# message.  This hook fires after signal handlers have run.
import atexit

def _log_exit():
    if _update_in_progress:
        try:
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write("Process exited unexpectedly during update\n")
        except Exception:
            pass

atexit.register(_log_exit)

# Only register the handlers once when the module is imported
signal.signal(signal.SIGINT, _handle_exit)
signal.signal(signal.SIGTERM, _handle_exit)


@app.route("/api/update", methods=["POST", "GET"])
def update() -> Any:
    """Rebuild the dataset and save it to disk.

    Returns a JSON object with the number of entries processed.
    """
    global _cache
    # Determine whether a specific title has been requested
    title_param = request.args.get("title")
    selected = None
    if title_param:
        try:
            selected = [int(title_param)]
        except ValueError:
            return jsonify({"status": "error", "message": "invalid title"}), 400
    # Clear log files for this run
    for path in (PROCESS_LOG, ERROR_LOG):
        try:
            with open(path, "w") as f:
                f.write("")
        except Exception:
            pass

    def log_progress(msg: str) -> None:
        # Append messages to the process log.  Use a try/except so
        # errors in logging do not interrupt the update.
        try:
            with open(PROCESS_LOG, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def log_error(ctx: str, exc: Exception) -> None:
        # Append errors to the error log.  Use try/except to avoid
        # cascading failures if the log cannot be written.
        try:
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"{ctx}: {exc}\n")
        except Exception:
            pass

    # Load existing dataset to determine progress
    existing_data = []
    if os.path.exists(DATA_FILE):
        existing_data = ecfr.load_dataset(DATA_FILE)
    # Normalise processed title numbers to integers for comparison
    processed_titles = set()
    for entry in existing_data:
        try:
            processed_titles.add(int(entry.get("title_number")))
        except Exception:
            continue

    # Determine titles to process
    try:
        titles_list = ecfr._fetch_json("/titles.json")["titles"]
    except Exception as e:
        return jsonify({"status": "error", "message": "failed to fetch titles"}), 500

    new_entries_total: list[dict] = []

    # Indicate that an update is now in progress so that the signal
    # handler and atexit hook can record an appropriate message if
    # the process is terminated.  We'll clear this flag in a
    # finally block below.
    global _update_in_progress
    _update_in_progress = True

    try:
        for t in titles_list:
            # eCFR's titles may have their number as a string.  Convert to int
            # for reliable comparisons with the selected list and processed set.
            try:
                num = int(t.get("number"))
            except Exception:
                continue
            # skip if a specific title was requested and this isn't it
            if selected and num not in selected:
                continue
            # skip if already processed (only when doing a full update)
            if not title_param and num in processed_titles:
                continue
            # process this title
            per_title = ecfr.build_dataset(
                selected_titles=[num],
                skip_titles=None,
                log_progress=log_progress,
                log_error=log_error,
            )
            if not per_title:
                continue
            # remove any existing entries for this title in existing_data. Convert
            # existing entry numbers to int for comparison
            cleaned = []
            for d in existing_data:
                try:
                    if int(d.get("title_number")) != num:
                        cleaned.append(d)
                except Exception:
                    cleaned.append(d)
            existing_data = cleaned
            existing_data.extend(per_title)
            ecfr.save_dataset(existing_data, DATA_FILE)
            processed_titles.add(num)
            new_entries_total.extend(per_title)
            # If only a single title was requested, stop after processing it
            if selected and num in selected:
                break
    except KeyboardInterrupt as ki:
        # Catch Ctrl‑C on platforms where SIGINT doesn't trigger our signal handler
        log_error("update interrupted", ki)
        # Re‑raise so that the process will terminate and the atexit hook fires
        raise
    except Exception as exc:
        # catch any unexpected exception and log it before returning
        log_error("update", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        # Clear the update flag regardless of success or failure so that
        # termination messages are only logged when the update is
        # interrupted.
        _update_in_progress = False
    # update the in‑memory cache
    _cache = existing_data
    return jsonify({"status": "ok", "count": len(new_entries_total)})


@app.route("/api/summary", methods=["GET"])
def summary() -> Any:
    """Return the cached dataset, loading from disk if necessary."""
    global _cache
    # Allow clients to request a specific title
    title_param = request.args.get("title")
    if _cache is None:
        if os.path.exists(DATA_FILE):
            _cache = ecfr.load_dataset(DATA_FILE)
        else:
            _cache = ecfr.build_dataset()
            ecfr.save_dataset(_cache, DATA_FILE)
    if title_param:
        try:
            t_num = int(title_param)
        except ValueError:
            return jsonify({"status": "error", "message": "invalid title"}), 400
        filtered = [d for d in _cache if d["title_number"] == t_num]
        return jsonify(filtered)
    return jsonify(_cache)


if __name__ == "__main__":
    # expose on port 5000 by default
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))