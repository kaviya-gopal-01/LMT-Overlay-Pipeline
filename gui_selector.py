"""
gui_selector.py

Tkinter-based file selection for the LMT overlay pipeline: lets the user
pick a SQLite database and one or more raw video files from anywhere on
disk, with no assumption about folder layout. Also persists the last
selected database's directory and the last video directory across runs
via a small JSON settings file, so repeat runs open in the same place
instead of the OS's default folder every time.

Tkinter is used because it ships with the Python standard library on
Windows and macOS, and is a single `apt install python3-tk` away on most
Linux distributions -- avoiding a third-party GUI dependency for what is
otherwise a stdlib-only project (see requirements.txt).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import config

logger = logging.getLogger(__name__)


class SelectionCancelled(Exception):
    """Raised when the user closes/cancels a file selection dialog."""

def _load_settings() -> dict:
    if not config.SETTINGS_FILE_PATH.exists():
        return {}
    try:
        return json.loads(config.SETTINGS_FILE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not read settings file %s (%s); starting with defaults.",
            config.SETTINGS_FILE_PATH, exc,
        )
        return {}


def _save_settings(settings: dict) -> None:
    try:
        config.SETTINGS_FILE_PATH.write_text(json.dumps(settings, indent=2))
    except OSError as exc:
        logger.warning(
            "Could not write settings file %s (%s); selections won't be "
            "remembered next run.", config.SETTINGS_FILE_PATH, exc,
        )


def select_database_and_videos() -> Tuple[Path, List[Path]]:
    """
    Show, in order: a single-file picker for the SQLite database, then a
    multi-file picker for raw videos. Remembers the last database's
    directory and the last video directory across runs via settings.json.

    Raises SelectionCancelled (with a message indicating which dialog)
    if the user cancels either one. Tkinter itself is imported inside
    this function, not at module level, so a missing/broken Tkinter
    installation only breaks this one function -- the rest of the
    pipeline (and even this module's settings helpers, which callers may
    want to unit test independently) stays importable.
    """
    import tkinter as tk
    from tkinter import filedialog

    settings = _load_settings()

    root = tk.Tk()
    root.withdraw()  # we only want the native dialogs, not an empty window
    root.attributes("-topmost", True)  # dialogs shouldn't hide behind other windows

    try:
        last_db = settings.get("last_database_path")
        initial_db_dir = (str(Path(last_db).parent) if last_db and Path(last_db).parent.is_dir() else None)

        db_path_str = filedialog.askopenfilename(
            title="Select the LMT SQLite database",
            filetypes=[("SQLite database", "*.sqlite *.db"), ("All files", "*.*")],
            initialdir=initial_db_dir,
            parent=root,
        )
        if not db_path_str:
            raise SelectionCancelled("No database file was selected (dialog was cancelled).")
        database_path = Path(db_path_str).resolve()

        last_video_dir = settings.get("last_video_dir")
        initial_video_dir = (last_video_dir if last_video_dir and Path(last_video_dir).is_dir() else None)

        video_path_strs = filedialog.askopenfilenames(
            title="Select one or more raw video files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
            initialdir=initial_video_dir,
            parent=root,
        )
        if not video_path_strs:
            raise SelectionCancelled("No video files were selected (dialog was cancelled).")
        video_paths = [Path(p).resolve() for p in video_path_strs]
    finally:
        root.destroy()

    settings["last_database_path"] = str(database_path)
    settings["last_video_dir"] = str(video_paths[0].parent)
    _save_settings(settings)

    return database_path, video_paths
