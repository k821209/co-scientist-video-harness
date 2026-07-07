"""Content-aware section headers / chapter titles.

A *chapter* = a semantic section boundary in the narration + a short title.
Detection is an LLM job (see `detect_chapters`); this module then:
  - hands the chapters to caption.build_ass to burn animated title cards, and
  - emits a YouTube "chapters" block (0:00 Title ...) for the video description.

The Chapter list is plain data, so it can come from an LLM, a human, or a
heuristic — the rendering doesn't care.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .transcribe import Word


@dataclass
class Chapter:
    start: float          # seconds into the (cleaned) video
    title: str


def _hhmmss(sec: float) -> str:
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def youtube_chapters(chapters: list[Chapter]) -> str:
    """YouTube description block. First entry MUST be 0:00 for chapters to work."""
    ch = sorted(chapters, key=lambda c: c.start)
    if not ch or ch[0].start > 0.5:
        ch = [Chapter(0.0, "시작")] + ch
    else:
        ch[0] = Chapter(0.0, ch[0].title)
    return "\n".join(f"{_hhmmss(c.start)} {c.title}" for c in ch)


def dump_chapters(chapters: list[Chapter], path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps([asdict(c) for c in chapters], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def detect_chapters(words: list[Word], api_key: str | None = None,
                    max_chapters: int = 8) -> list[Chapter]:
    """LLM-based section detection over the transcript.

    Sends the timestamped transcript to Claude and asks for section boundaries
    + titles. Requires ANTHROPIC_API_KEY (or api_key). Raises if unavailable so
    the caller can fall back to manual chapters.
    """
    import os
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("no ANTHROPIC_API_KEY — pass chapters manually")

    # Build a compact timestamped transcript.
    lines = []
    for w in words:
        lines.append(f"[{w.start:.1f}] {w.text}")
    transcript = " ".join(lines)

    import anthropic
    client = anthropic.Anthropic(api_key=key)
    prompt = (
        "다음은 영상 나레이션의 단어별 타임스탬프 전사입니다. 내용 흐름이 바뀌는 "
        f"지점을 최대 {max_chapters}개 섹션으로 나누고, 각 섹션에 짧은 한국어 제목"
        "(6단어 이내)을 붙이세요. 반드시 첫 섹션은 0.0초에서 시작합니다.\n"
        "JSON 배열로만 답하세요: [{\"start\": <초>, \"title\": \"...\"}, ...]\n\n"
        f"전사:\n{transcript}"
    )
    msg = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    text = text[text.find("["): text.rfind("]") + 1]
    data = json.loads(text)
    return [Chapter(float(d["start"]), str(d["title"])) for d in data]
