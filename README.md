# LMT Overlay Pipeline Documentation

This README documents a GUI-driven pipeline that takes a Live Mouse Tracker
(LMT) SQLite output and a set of raw (no-overlay) video files, and produces
new videos with **corrected** tracking overlays burned in: per-animal
segmentation masks, pose markers, posture labels, and the global SQLite
frame number.

The pipeline exists because LMT's own live overlay is generated in real
time, during acquisition, and is not always correct: it reflects whatever
the tracker detected *at that instant*, not the corrected, offline-processed 
result that ends up in the SQLite database afterward. The SQLite database
is treated as ground truth throughout this pipeline; the raw, overlay-free
videos are the only thing ever drawn on, and are never modified in place, a
fresh corrected copy is always written to a new output file.

Unlike the QC pipeline, this is not a chain of independently-run scripts
that hand a file to the next stage. It is a single program (`main.py`) built
from focused, single-responsibility modules that are all imported and run
together in one pass: select inputs -> decode -> render -> write. 

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires **Python 3.10+** (the codebase uses `dataclass(slots=True)`, a
3.10 feature). On most Linux distributions, Tkinter is a separate system
package and must be installed alongside Python itself:

```bash
sudo apt-get install python3-tk    # Debian/Ubuntu
```

Tkinter ships built into the standard Python installer on Windows and
macOS, no extra step needed there.

## Usage

```bash
python main.py
```

Running this launches two file-picker dialogs in order:

1. **Select the LMT SQLite database.**
2. **Select one or more raw video files** (may span multiple folders, any
   naming convention, as long as the filename ends in `_t<digits>.<ext>`).

Every video selected is then processed against the selected database, and
corrected output is written into a freshly created, timestamped folder:

```
output/2026-07-20_17-34-12/
    video_t18024.mp4
    video_t25000.mp4
```

Optional flags:

| Flag | Purpose |
|---|---|
| `--output-base-dir PATH` | Base folder under which each run's timestamped subfolder is created (default: `./output`). |
| `--no-progress` | Disable `tqdm` progress bars (useful for log-file/non-interactive runs). |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Logging verbosity (default: `INFO`). |

## Project layout

| File | Responsibility |
|---|---|
| `config.py` | Every tunable constant: paths, colors, alpha, frame-conversion factor, timestamp format. No logic, just values. |
| `gui_selector.py` | Tkinter file-picker dialogs for the database and videos; persists last-used locations to `settings.json`. |
| `database.py` | The only module that talks to SQLite; opens read-only connections, runs bounded range queries against `DETECTION`. |
| `xml_parser.py` | Parses the `DATA` XML column of one detection row into structured pose + ROI data. |
| `mask_decoder.py` | Decodes the `boolMaskData` hex payload inside that XML into a 2D NumPy segmentation mask. |
| `overlay_renderer.py` | Draws decoded masks, pose markers, posture labels, and the frame number onto a video frame. |
| `video_processor.py` | Discovers videos (from a folder or an explicit file list), streams through each one frame-by-frame, and writes corrected output. |
| `main.py` | CLI entrypoint: wires everything above together into one run. |

Dependency direction is strictly one-way, low-level to high-level:

```
main.py
 |- gui_selector.py            (get database path + video paths from the user)
 |- video_processor.py
     |- database.py            (fetch detection rows for a frame range)
     |- overlay_renderer.py
         |- xml_parser.py      (parse DATA XML -> pose + ROI)
         |- mask_decoder.py    (decode boolMaskData -> numpy mask)
```

`config.py` sits underneath all of them; every other module reads
`config.SOMETHING` at call time, never caches it, so CLI overrides applied
early in `main.py` propagate correctly everywhere else.

---

## Module: `config.py`

### Overview
Pure data: every path, color, threshold, and format string the rest of the
pipeline depends on lives here, plus one function, `validate_config()`,
that sanity-checks those values (and creates the output directory) before
processing begins. Nothing in this file talks to a database or opens a
video.

### Key settings

| Setting | Meaning |
|---|---|
| `DATABASE_PATH`, `OUTPUT_VIDEO_DIR` | Set at runtime by `main.py` from the GUI selection and the computed timestamped output folder, respectively. |
| `INPUT_VIDEO_DIR`, `VIDEO_GLOB_PATTERN` | Only relevant to `video_processor.discover_videos()` (the folder-glob discovery path), not used by the default GUI flow, which uses `discover_videos_from_paths()` instead. |
| `FRAME_CONVERSION` | Ratio between SQLite's frame numbering and raw video fps (default `2`: DB frames run at 30fps-equivalent, video at 15fps). The single most consequential constant in the project, everything else's frame math derives from it. |
| `MASK_STRICT_SIZE_VALIDATION` | Keep `True` in production; a decompressed mask whose size doesn't match its `ROI` bounds raises rather than silently misaligning. |
| `ANIMAL_COLORS_BGR`, `MASK_ALPHA`, `DRAW_*` flags | Visual tuning; safe to adjust by eye. |
| `OUTPUT_VIDEO_PREFIX` | Output filenames are built as `{prefix}_t{start_frame}{ext}`, e.g. `video_t18024.mp4`, independent of the input filename's own prefix. |
| `SETTINGS_FILE_PATH` | Where `gui_selector.py` persists the last-used database and video-folder paths. |
| `OUTPUT_TIMESTAMP_FORMAT` | `strftime` pattern for each run's output subfolder name. |

### Key Design Decisions & Assumptions
- **`INPUT_VIDEO_DIR` is intentionally not validated by `validate_config()`.**
  The GUI-driven flow supplies an explicit list of video files, not a
  directory to glob, so this constant may be unset or nonexistent and
  that's fine, it only matters if `discover_videos()` is called directly.
- **Every other module reads `config.X` at call time, never at import
  time.** This is what allows `main.py` to reassign `config.DATABASE_PATH`
  and `config.OUTPUT_VIDEO_DIR` *after* every other module has already
  imported `config`, and have that change still take effect everywhere.

### Do NOT Modify
- `FRAME_CONVERSION`'s meaning (SQLite-frame-units per video-frame); every
  frame-mapping computation in `video_processor.py` assumes this exact
  semantic.
- `MASK_STRICT_SIZE_VALIDATION` should not be disabled in production; it is
  the runtime check that would otherwise catch a wrong mask-decoding
  assumption or genuinely corrupt data.

### Open Source Notes
- **External dependencies**: none.
- **Standard library**: `os`, `re`, `pathlib`, `typing`.
- **Configuration files / environment variables**: `LMT_INPUT_DIR`,
  `LMT_DATABASE_PATH`, `LMT_OUTPUT_DIR`, `LMT_LOG_LEVEL` optionally override
  the corresponding defaults.

---

## Module: `gui_selector.py`

### Overview
Presents two native file-picker dialogs, database then videos, and
persists the last-used locations across runs so repeat sessions don't
start back at the OS default folder.

### Inputs
None programmatic; entirely interactive (native OS file dialogs).

### Outputs

| Output | Type | Purpose |
|---|---|---|
| `(database_path, video_paths)` | `Tuple[Path, List[Path]]` | Returned to `main.py`; feeds `config.DATABASE_PATH` and `video_processor.discover_videos_from_paths()`. |
| `settings.json` | JSON file, next to the project code | Remembers the last selected database's folder and last selected video folder. |

### Processing Steps
1. **Load `settings.json`** if present; a missing or corrupted file is
   treated as empty defaults (logged, never fatal).
2. **Show the database picker**, pre-populated with the last-used folder if
   it still exists. An empty return value (user cancelled) raises
   `SelectionCancelled`.
3. **Show the video picker** (multi-select), pre-populated with the last
   video folder if it still exists. Same cancellation handling.
4. **Persist the new selections** to `settings.json` and return.

### Key Design Decisions & Assumptions
- **`import tkinter` happens inside the function, not at module level.** A
  missing/broken Tkinter installation only breaks
  `select_database_and_videos()` itself; the settings-persistence helpers
  (and this module generally) stay importable and independently testable.
- **Cancellation is a distinct, expected outcome, not an error.** Both
  dialogs raise the same `SelectionCancelled` exception (with a message
  identifying which dialog was cancelled) rather than returning a sentinel
  value a caller could forget to check.
- **A stale remembered path is not used blindly.** Both the database and
  video initial directories are checked with `.is_dir()` before being
  passed to the dialog, falling back to the OS default location if the
  remembered folder no longer exists.

### Do NOT Modify
- The `SelectionCancelled` contract, `main.py` catches this exact exception
  type to exit gracefully; changing it to a different exception or a
  falsy return value breaks that handling.
- `settings.json`'s two keys (`last_database_path`, `last_video_dir`) are
  the full extent of what's persisted; anything else added here should
  degrade gracefully if absent (as these two already do).

### Open Source Notes
- **External dependencies**: `tkinter` (standard library on Windows/macOS;
  `python3-tk` system package on most Linux distributions).
- **Standard library**: `json`, `logging`, `pathlib`.
- **Configuration files**: reads/writes `config.SETTINGS_FILE_PATH`
  (default: `settings.json` next to the project code).
- **Platform assumptions**: requires a desktop environment capable of
  displaying a Tkinter window; not headless-safe.

---

## Module: `database.py`

### Overview
The only module that opens a connection to the LMT SQLite database.
Connections are opened in SQLite's URI **read-only** mode, enforced at the
SQLite level rather than by convention, so this pipeline cannot mutate the
ground-truth database regardless of bugs elsewhere in the codebase.

### Inputs

| Input | Type | Purpose |
|---|---|---|
| `config.DATABASE_PATH` | Path | The LMT SQLite database, set by `main.py` from the GUI selection. |
| `(start_frame, end_frame)` | Integers | Half-open global-frame range to query, per call. |

### Outputs

| Output | Type | Purpose |
|---|---|---|
| `DetectionRow` instances | Python objects (`__slots__`) | Attribute-access wrapper around one `DETECTION` row, consumed by `overlay_renderer.py`. |
| Grouped detection dicts | `Dict[int, List[DetectionRow]]` | Pre-grouped by `FRAMENUMBER` for O(1) lookup while iterating video frames. |

### Processing Steps
1. **`connect()`** opens a read-only URI connection as a context manager;
   always closed via `finally`, even on error.
2. **`fetch_detections_in_range()`** runs one bounded SQL query per call
   (`WHERE FRAMENUMBER >= ? AND FRAMENUMBER < ?`), never an unbounded
   `SELECT *`.
3. **`iter_detection_chunks()`** wraps repeated calls to the above into a
   generator yielding pre-grouped dicts, bounding peak memory to one
   chunk's worth of detections regardless of recording length.
4. **`warn_if_unindexed()`** checks (via `PRAGMA index_list`) whether
   `DETECTION` has an index covering `FRAMENUMBER`, and logs a suggested
   `CREATE INDEX` statement if not, it does not attempt to create one
   itself, since the connection is intentionally read-only.

### Key Design Decisions & Assumptions
- **Read-only enforced at the SQLite level (`mode=ro` in the connection
  URI), not just "we promise not to write."** This matches the project's
  core premise that the database is ground truth.
- **`DetectionRow` uses `__slots__`** rather than raw `sqlite3.Row`
  dict-style access, since this class is instantiated once per animal per
  frame, potentially millions of times across a long recording.

### Do NOT Modify
- The read-only connection mode. Any future feature that seems to need
  write access to this database should write to a *different* file
  instead.
- `DetectionRow`'s attribute names, `overlay_renderer.py` and
  `video_processor.py` both access these by name.

### Open Source Notes
- **External dependencies**: none.
- **Standard library**: `sqlite3`, `contextlib`, `logging`, `pathlib`,
  `typing`.
- **Configuration files / environment variables**: none directly; reads
  `config.DATABASE_PATH`, `config.DETECTION_TABLE`,
  `config.DETECTION_COLUMNS`, `config.DB_QUERY_CHUNK_FRAMES`.

---

## Module: `xml_parser.py`

### Overview
Parses the `DATA` XML blob stored in each `DETECTION.DATA` cell into two
structured, immutable dataclasses: pose/posture data, and (when present) an
`ROIBounds` describing the segmentation mask's bounding box and raw,
still-encoded payload. Does not decode the mask itself, that is
`mask_decoder.py`'s job; this module stops at "here is the raw hex string
and the box it belongs in."

### Inputs

| Input | Type | Purpose |
|---|---|---|
| `detection.data_xml` | `str` | Raw `DATA` XML for one detection row. |

### Outputs

| Output | Type | Purpose |
|---|---|---|
| `ParsedDetectionData` | Frozen, slotted dataclass | Pose coordinates, posture flags, optional `roi`. |
| `ROIBounds` | Frozen, slotted dataclass | Mask bounding box (`x`, `y`, `w`, `h`) + raw `bool_mask_data` hex string. |

### Processing Steps
1. **Parse the XML** with `xml.etree.ElementTree`; empty/unparseable input
   raises `XmlParseError`.
2. **Require `<DATA>`** to exist; its absence also raises `XmlParseError`,
   this indicates genuinely corrupt data.
3. **`<ROI>` is optional and its absence is not an error.** Gap-filled or
   undetected frames legitimately have no segmentation; `_parse_roi()`
   returns `None` (not an exception) for a missing, incomplete, or
   zero-sized `<ROI>` block.
4. **Return a fully populated `ParsedDetectionData`**, with `roi=None` when
   there is no mask.

### Key Design Decisions & Assumptions
- **Missing `<ROI>` = expected (`roi=None`). Missing `<DATA>` or malformed
  XML = actually broken (`XmlParseError`).** This asymmetry is the core
  design of this module: it lets `overlay_renderer.py` treat "no mask this
  frame" as a normal branch rather than wrapping every single detection in
  a `try/except`.
- **Frozen, `slots=True` dataclasses**, since a new instance is constructed
  per animal per frame, potentially millions of times per recording.

### Do NOT Modify
- The has-mask contract (`roi is None` means "no mask available", never an
  error state), `overlay_renderer.py`'s branching depends on this exactly.
- Field names on both dataclasses, referenced directly (not via dict
  lookup) throughout `overlay_renderer.py`.

### Open Source Notes
- **External dependencies**: none.
- **Standard library**: `xml.etree.ElementTree`, `dataclasses`, `logging`,
  `typing`.
- **Configuration files / environment variables**: none.

---

## Module: `mask_decoder.py`

### Overview
Decodes the `boolMaskData` payload from a parsed `ROIBounds` into a 2D
NumPy segmentation mask. The encoding, colon-separated hex bytes,
zlib-compressed (Java `Deflater` default level), decompressing to exactly
`width * height` bytes of `{0, 1}` values in row-major order, was
determined empirically against real sample data and is
documented in full in this module's docstring.

### Inputs

| Input | Type | Purpose |
|---|---|---|
| `bool_mask_data` | `str` | Raw colon-separated hex string from `ROIBounds.bool_mask_data`. |
| `width`, `height` | `int` | `boundsW` / `boundsH` from the same `<ROI>` block. |

### Outputs

| Output | Type | Purpose |
|---|---|---|
| `mask` | `np.ndarray`, shape `(height, width)`, `uint8` | `{0, 1}`-valued segmentation mask, `1` = foreground (mouse) pixel. |

### Processing Steps
1. **Parse hex tokens individually** (`int(tok, 16)` per colon-separated
   token, not fixed-width pairs, tokens are not zero-padded).
2. **`zlib.decompress()`** the resulting byte string.
3. **Validate decompressed length** equals `width * height` exactly; raises
   `MaskDecodeError` if `config.MASK_STRICT_SIZE_VALIDATION` is `True`
   (default).
4. **Reshape row-major** to `(height, width)` via `np.frombuffer(...).reshape(height, width)`.
5. **Validate the value domain** is exactly `{0, 1}`; same strict/non-strict
   handling as the length check.

### Key Design Decisions & Assumptions
- **The decode recipe is evidence-based, not assumed**: the `0x78 0x5e`
  zlib header, the exact `width*height` byte-count match, the `{0,1}`
  value domain, and the row-major (not column-major) reshape producing a
  coherent silhouette were each independently verified against real data
  before being treated as established.
- **Two independent runtime validation layers** (size check, value-domain
  check) guard against a wrong assumption or corrupt data reaching
  `overlay_renderer.py` and being silently baked into an output video.

### Do NOT Modify
- The decode sequence itself (hex-token parse -> zlib decompress ->
  row-major reshape), this is a verified fact about the data format, not a
  stylistic choice.
- `MASK_STRICT_SIZE_VALIDATION`'s default of `True` in production use.

### Open Source Notes
- **External dependencies**: `numpy`.
- **Standard library**: `zlib`, `logging`, `typing`.
- **Configuration files / environment variables**: none directly; reads
  `config.MASK_BYTE_SEPARATOR`, `config.MASK_FOREGROUND_VALUE`,
  `config.MASK_STRICT_SIZE_VALIDATION`.

---

## Module: `overlay_renderer.py`

### Overview
Draws everything onto a single video frame, given that frame's
`DetectionRow` list: colorized mask fill, mask outline, pose-point
markers, a posture text label, and (called separately by
`video_processor.py`) the global frame number. This is the only module
that calls both `xml_parser.py` and `mask_decoder.py`, it is the glue
between "raw DB row" and "pixels."

### Inputs

| Input | Type | Purpose |
|---|---|---|
| `frame` | `np.ndarray` | The video frame to draw onto, modified in place. |
| `detections` | `Iterable[DetectionRow]` | That frame's detection rows (may be empty). |
| `global_frame` | `int` | For `draw_global_frame_number()` only. |

### Outputs
None returned separately; `frame` is mutated in place (`render_frame()`
also returns it, for convenient chaining).

### Processing Steps (per detection, in `render_detection`)
1. **Pick a color** via `get_animal_color()` (dict lookup with a
   deliberately-obvious fallback color for any unconfigured `ANIMALID`).
2. **Parse the XML** (`xml_parser.parse_detection_xml`); a parse failure is
   logged and this detection is skipped, it does not abort the frame.
3. **Draw pose markers first** (if enabled), so the mask fill drawn next
   paints over them where they fall inside the silhouette.
4. **Decode and draw the mask** (if `parsed.has_mask`): alpha-blend via
   `cv2.addWeighted`, committed only where the mask is `1` (boolean-indexed
   assignment), then an outline traced from the mask's own contour, offset
   into frame coordinates. A decode failure is logged and only the mask
   step is skipped, pose markers/label still render.
5. **Draw the posture label last**, so it's always legible on top of
   everything else.

### Key Design Decisions & Assumptions
- **Failure isolation is per-detection, not per-frame.** A corrupt XML
  blob or undecodable mask for one animal in one frame logs a warning and
  is skipped; it never aborts rendering for the rest of that frame's
  animals, or the rest of the video.
- **Draw order is deliberate**: markers -> mask -> label, so nothing
  visually important ends up hidden underneath something else.
- **ROI clipping** (`_clip_roi_to_frame`) handles animals detected at the
  edge of the cage, where the mask's bounding box partially exceeds frame
  bounds, without raising an `IndexError`.
- **The frame number is a separate, frame-level function** (not part of
  `render_detection`/`render_frame`), since it must appear even on frames
  with zero detections; `video_processor.py` calls it directly.

### Do NOT Modify
- The draw order (markers, then mask, then label) without deliberately
  intending the visual change it causes.
- Per-detection `try/except` isolation around XML parsing and mask
  decoding, removing this turns one corrupt row into a run-ending crash.

### Open Source Notes
- **External dependencies**: `opencv-python` (`cv2`), `numpy`.
- **Standard library**: `logging`, `typing`.
- **Configuration files / environment variables**: none directly; reads
  numerous `config.*` rendering constants (colors, alpha, marker/label/
  frame-number styling).

---

## Module: `video_processor.py`

### Overview
The orchestration layer between raw video files and rendered output.
Provides two independent **discovery** paths, folder-glob (`discover_videos`,
for scripted/headless use) and explicit file list (`discover_videos_from_paths`,
used by the GUI flow), both producing the same `VideoInfo` objects and
sharing the same duplicate-protection logic. **Processing**
(`process_single_video`) opens each video exactly once, streams through it
sequentially, and pulls detections from SQLite in bounded chunks aligned to
that video's own frame range.

### Inputs

| Input | Type | Purpose |
|---|---|---|
| `input_dir` (folder mode) or `video_paths` (explicit-list mode) | `Path` / `List[Path]` | Which raw videos to process. |
| SQLite connection | `sqlite3.Connection` | Passed through to `database.py` calls. |

### Outputs

| Output | Type | Purpose |
|---|---|---|
| `VideoInfo` list | Dataclass instances | One per discovered/probed video: path, start frame, frame count, dimensions, computed `output_path`. |
| Corrected `.mp4` files | Video files | Written to `config.OUTPUT_VIDEO_DIR`, named `{prefix}_t{start_frame}{ext}`. |

### Processing Steps
1. **Discovery** (`probe_video`): opens a video briefly to read its frame
   count/fps/dimensions and extracts its global start frame from the
   filename (`_t<digits>` pattern), then closes it immediately, decoupled
   from the later processing open.
2. **Duplicate protection** (`_deduplicate_by_start_frame`): since output
   filenames are keyed only by `start_frame`, two videos sharing one would
   silently overwrite each other's output; this is detected and the second
   one is skipped with a clear log message, shared by both discovery
   paths.
3. **Processing** (`process_single_video`): opens the video once, computes
   a local-frame chunk size from `config.DB_QUERY_CHUNK_FRAMES`, and for
   each chunk: one bounded SQL query, pre-grouped by `FRAMENUMBER`, then a
   plain sequential `cap.read()` loop (never a seek) matching each local
   frame to its global frame number, rendering, and writing.
4. **Per-video failure isolation** (`process_all_videos`): a video that
   fails to open, or truncates mid-read, logs an error/warning and the run
   continues to the next video rather than aborting the whole batch.

### Key Design Decisions & Assumptions
- **Sequential `cap.read()`, never `cap.set(CAP_PROP_POS_FRAMES, ...)`**,
  since every frame of every video is processed anyway, sequential reads
  are both faster and more reliable across codecs than seeking.
- **One SQL query per chunk, not per frame or per whole video.** Chunk size
  is derived in local-frame units from the global-frame
  `DB_QUERY_CHUNK_FRAMES` setting, so it covers a consistent span of
  wall-clock recording time regardless of `FRAME_CONVERSION`.
- **Output filename is derived only from already-trusted fields**
  (`start_frame`, original extension), never from string-manipulating the
  input filename, so it produces identical output naming
  (`video_t18024.mp4`) regardless of the input file's own prefix or which
  folder it came from.

### Do NOT Modify
- The output filename scheme (`{OUTPUT_VIDEO_PREFIX}_t{start_frame}{ext}`),
  and the duplicate-`start_frame` check that protects it.
- Sequential frame reading; do not reintroduce `CAP_PROP_POS_FRAMES`
  seeking without re-verifying reliability across your target codecs.
- The per-video `try/except VideoProcessingError` boundary in
  `process_all_videos`, this is what keeps one bad file from losing an
  entire batch's worth of already-completed processing.

### Open Source Notes
- **External dependencies**: `opencv-python` (`cv2`), `tqdm` (optional,
  degrades gracefully if absent).
- **Standard library**: `logging`, `dataclasses`, `pathlib`, `typing`,
  `sqlite3`.
- **Configuration files / environment variables**: none directly; reads
  `config.VIDEO_GLOB_PATTERN`, `config.VIDEO_FILENAME_PATTERN`,
  `config.FRAME_CONVERSION`, `config.DB_QUERY_CHUNK_FRAMES`,
  `config.OUTPUT_FOURCC`, `config.OUTPUT_FPS`, `config.PROGRESS_BAR`.

---

## Module: `main.py`

### Overview
The CLI entrypoint. Parses a small set of optional flags, launches the GUI
selection flow, computes a unique timestamped output directory, validates
configuration, discovers/probes the selected videos, and runs processing,
with clean, specific exit codes for every failure path.

### Inputs

| Input | Type | Purpose |
|---|---|---|
| `--output-base-dir`, `--no-progress`, `--log-level` | CLI flags, all optional | See Usage section above. |

### Outputs

| Output | Type | Purpose |
|---|---|---|
| Exit code | `int` | `0` success, `1` error/cancelled/nothing-to-do, `130` interrupted (`Ctrl-C`), matching standard Unix conventions. |
| `output/<timestamp>/` | Directory | This run's corrected videos. |

### Processing Steps
1. **Parse arguments, configure logging.**
2. **Launch the GUI picker** (`gui_selector.select_database_and_videos`);
   catches `ImportError` (Tkinter unavailable, with an actionable
   install-command message) and `SelectionCancelled` (user cancelled,
   logged as a warning, not an error) separately.
3. **Set `config.DATABASE_PATH`** from the selection, and compute this
   run's unique output directory (`_make_unique_output_dir`): a
   timestamped subfolder under `--output-base-dir`, with a numeric suffix
   appended if that exact folder somehow already exists (guaranteeing
   output is never silently overwritten).
4. **`config.validate_config()`** checks the database exists and creates
   the output directory.
5. **`video_processor.discover_videos_from_paths()`** probes every selected
   video file.
6. **Sanity-check frame overlap** between the selected videos' global frame
   range and the database's `DETECTION` frame range, warning (not
   aborting) if they don't overlap at all, usually indicates the wrong
   database was picked.
7. **`video_processor.process_all_videos()`** runs the actual rendering
   pass.
8. **Report a completion summary** (video count, elapsed time, output
   location).

### Key Design Decisions & Assumptions
- **Every failure path returns a specific, documented exit code** rather
  than letting an exception propagate to a raw traceback, this matters for
  both interactive use and any future scripted/scheduled invocation.
- **Cancelling a dialog is not treated as an error condition** in its log
  message (`logger.warning`, not `logger.error`), even though its exit
  code is `1` (no work was done) for consistency with other empty-result
  paths.
- **The unique-output-directory guarantee is enforced here, not left to
  chance**: `_make_unique_output_dir` actively checks for and avoids
  collisions rather than relying on timestamp granularity alone.

### Do NOT Modify
- The exit code conventions (`0`/`1`/`130`), if any external tooling
  eventually wraps this script, it will depend on these.
- The order of operations in step 3 above (`DATABASE_PATH` and
  `OUTPUT_VIDEO_DIR` must both be set on `config` *before*
  `validate_config()` runs), reordering this breaks validation.

### Open Source Notes
- **External dependencies**: none beyond what the modules it imports
  already require.
- **Standard library**: `argparse`, `logging`, `sys`, `time`, `datetime`,
  `pathlib`.
- **Configuration files / environment variables**: none read directly by
  `main.py` itself, though `config.py`'s own env var overrides
  (`LMT_LOG_LEVEL`, etc.) still apply as defaults before CLI flags are
  parsed.
- **Platform assumptions**: requires Tkinter GUI support (not
  headless-safe) for its default flow; `video_processor.discover_videos()`
  remains available for headless/scripted use if called directly instead
  of through `main.py`.

---

## Consolidated dependency summary

| Dependency | Used by | Required? |
|---|---|---|
| `numpy` | `mask_decoder.py`, `overlay_renderer.py` | Yes |
| `opencv-python` (`cv2`) | `overlay_renderer.py`, `video_processor.py` | Yes |
| `tqdm` | `video_processor.py` | Optional (degrades to no progress bar) |
| `tkinter` | `gui_selector.py` | Yes, for the default GUI flow (system package on Linux) |

No other third-party dependencies anywhere in the project. Standard-library
modules used across the codebase: `sqlite3`, `zlib`, `xml.etree.ElementTree`,
`json`, `argparse`, `logging`, `pathlib`, `dataclasses`, `datetime`, `re`,
`os`, `sys`, `time`, `contextlib`, `typing`.
