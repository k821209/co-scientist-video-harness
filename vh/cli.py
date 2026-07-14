"""Command-line entry point.

Examples:
  vh run recording.mp4 --preset screencast
  vh run webcam.mkv --preset shorts --lang ko
  vh presets
"""
from __future__ import annotations

import argparse
import sys

from .config import PRESETS
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vh", description="headless video harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process one recording end-to-end")
    r.add_argument("input", help="source recording (mp4/mkv/mov/...)")
    r.add_argument("--preset", default="screencast", choices=list(PRESETS))
    r.add_argument("--out", default="out", help="output directory")
    r.add_argument("--lang", default=None, help="force whisper language (e.g. ko, en)")
    r.add_argument("--no-keep", action="store_true", help="delete intermediates")

    sub.add_parser("presets", help="list available presets")

    g = sub.add_parser("styles", help="build a video-style gallery from a styles/ dir")
    g.add_argument("styles_dir", help="project styles directory (catalog.json + thumbs/)")
    g.add_argument("--out", default=None, help="output html (default <styles_dir>/gallery.html)")
    g.add_argument("--title", default=None)
    g.add_argument("--no-base", action="store_true", help="project catalog only, skip vh base styles")

    args = p.parse_args(argv)

    if args.cmd == "styles":
        from .style_gallery import build_style_gallery
        path = build_style_gallery(args.styles_dir, out=args.out, title=args.title,
                                   include_base=not args.no_base)
        print(f"[done] {path}")
        return 0

    if args.cmd == "presets":
        for name, pr in PRESETS.items():
            print(f"{name:14s} aspect={pr.aspect:5s} caption={pr.caption_style:5s} "
                  f"reframe={pr.reframe_mode:4s} silence>{pr.silence_threshold}")
        return 0

    if args.cmd == "run":
        res = run_pipeline(
            args.input, args.preset, args.out,
            language=args.lang, keep_intermediates=not args.no_keep,
        )
        print("\n=== output ===")
        print("video :", res.final)
        print("srt   :", res.srt or "(none)")
        print("words :", res.n_words)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
