# video-harness

Headless recording → publish-ready YouTube long-form / vertical Shorts.
Pure `ffmpeg` + `Whisper` — no CapCut (it has no headless automation API), no cloud editor.

## Install / run

No install needed — run as a module, or install the `vh` console script:

```bash
# module form (what we've been using)
<python> -m vh.cli run recording.mp4 --preset screencast --lang ko

# or install to get the `vh` command (entry point = vh.cli:main)
pip install -e .
vh run recording.mp4 --preset shorts
vh presets
```

`ffmpeg` + `ffprobe` must be on PATH (or set `VH_FFMPEG` / `VH_FFPROBE`).

## CLI (vh/cli.py, argparse)

```
vh run <input> [--preset screencast|talkinghead|shorts|slides|shorts_boxed]
               [--out DIR] [--lang ko|en|...] [--no-keep]
vh presets                         # list presets
```

`run` does: clean (silence trim) → reframe (aspect) → transcribe → caption burn.
Outputs land in `out/<stem>/`: `*.final.mp4`, `.srt`, `.ass`, `.words.json`.

### Presets (vh/config.py)
| preset | aspect | captions | reframe/layout |
|--------|--------|----------|----------------|
| `screencast`   | 16:9 | line | none |
| `talkinghead`  | 16:9 | word | none |
| `shorts`       | 9:16 | word | blur-pad (aspect-aware) |
| `shorts_boxed` | 9:16 | word | **boxed** (header / video band / caption zones) |
| `slides`       | 16:9 | line | none |

## Library (vh/steps/*) — driven directly for advanced edits

The CLI covers the common path; these are composed in a script for the rest
(chapters, title cards, boxed vertical). See each module:

- `steps/clean.py` — silence removal via ffmpeg `silencedetect` (aarch64-safe;
  auto-editor 29.x is x86_64-only).
- `steps/transcribe.py` — word-level Whisper. Backend via `VH_ASR_BACKEND`:
  `gpu` shells out to `VH_GPU_PYTHON` (transformers Whisper), `cpu` uses
  in-process faster-whisper int8.
- `steps/caption.py` — `build_ass()` (word-pop / line + optional chapter title
  cards), `build_boxed_ass()` (3-zone vertical), `burn()`.
- `steps/chapters.py` — `Chapter(start, title)` list, `youtube_chapters()`
  → `0:00 Title` description block, `detect_chapters()` (LLM, optional).
- `steps/reframe.py` — aspect-aware: no blur bars / no upscale when the source
  already matches the target aspect.
- `steps/compose.py` — `compose_boxed()`: landscape → 9:16 with header/caption zones.
- `steps/titlecards.py` — `render_card()` + `build_with_interstitials()`: splice
  full-frame chapter title cards at boundaries and re-time captions.

Example (chapters + interstitial title cards):
```python
from vh.steps import titlecards as T, chapters as C
from vh.steps.transcribe import Word
words = [Word(**w) for w in json.load(open("words.json"))]
chs = [C.Chapter(0, "인트로"), C.Chapter(63, "설치"), ...]      # Claude-in-the-loop
T.build_with_interstitials("src.mp4", "final.mp4", chs, words,
                           card_dur=1.8, style="word", max_words=5)
print(C.youtube_chapters(chs))                                  # description block
```

## Config (vh/config.py) — all overridable via env

| env var | purpose |
|---------|---------|
| `VH_FFMPEG` / `VH_FFPROBE` | ffmpeg / ffprobe binaries |
| `VH_VENC` | video encoder: `h264_nvenc` (default) or `libx264` |
| `VH_ASR_BACKEND` | `gpu` (default) or `cpu` |
| `VH_WHISPER_MODEL` | `small` (default), `large-v3`, … |
| `VH_GPU_PYTHON` | interpreter for the GPU transcription worker |
| `VH_CAPTION_FONT` / `VH_CAPTION_FONTSDIR` | caption font (Noto Sans CJK KR for Korean) |
| `VH_WHISPER_THREADS`, `VH_WHISPER_DEVICE`, `VH_WHISPER_COMPUTE` | CPU-backend tuning |
| `VH_RENDER_HOST` / `VH_RENDER_PORT` | **opt-in** remote render host (ssh target + port) |
| `VH_RENDER_PYTHON` / `VH_RENDER_TMP` | remote interpreter (needs faster-whisper) + scratch dir |
| `VH_RENDER_FFMPEG` / `VH_RENDER_FONTSDIR` | remote ffmpeg + font dir (defaults: `ffmpeg`, same Noto path) |

### Remote offload (opt-in)

Set a render host and **both transcription and encoding run there automatically**
(`VH_ASR_BACKEND=auto`, the default: remote if a host is set, else local). Unset
→ everything local.

```bash
export VH_RENDER_HOST="user@host"
export VH_RENDER_PORT="22"
export VH_RENDER_PYTHON="/path/to/env/bin/python"   # faster-whisper + CUDA installed
# host also needs ffmpeg (NVENC) + Noto CJK fonts for Korean captions
```

- **Transcription**: audio is scp'd over, faster-whisper runs on the host
  (CUDA ~59x realtime on a 4090), words pulled back.
- **Encoding** (burn / boxed / interstitials / clean / reframe): each ffmpeg job
  ships its inputs (mp4, .ass, card clips), runs `h264_nvenc` on the host, and
  pulls the output back. Local paths in the command are rewritten to remote
  paths automatically (`vh/remote.py::run_ffmpeg`).
- **Session cache**: inputs are staged through a persistent cache keyed by
  size+mtime+name, so a large source uploads **once** and is reused across every
  ffmpeg call and re-render (revisions, 16:9 + 9:16 from one source). Cache dir
  defaults to `{VH_RENDER_TMP}/vh_cache` (override `VH_RENDER_CACHE`); wipe with
  `remote.clear_cache()`.

No manual scp/ssh. The address lives ONLY in your environment — never in the
repo, skills, or logs. See `vh/remote.py` and `steps/transcribe.py`.

## Notes / platform
- **NVENC** (`h264_nvenc`) for 1080p60 — far faster than CPU x264.
- **Korean captions** need Noto Sans CJK KR; libass gets it via the `fontsdir`
  option because conda ffmpeg's fontconfig may not scan system font dirs.
- **aarch64 (GB10)**: auto-editor won't run (x86_64 binary) → native silence
  removal; faster-whisper has no CUDA wheel → transformers-Whisper on GPU or CPU.
