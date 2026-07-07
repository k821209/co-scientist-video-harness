"""Render-host offload transport (opt-in, env-configured).

If VH_RENDER_HOST is set, heavy stages ship their inputs here over ssh/scp, run
on the remote box, and pull the outputs back. If it is unset, callers fall back
to local execution. The host address lives ONLY in the user's environment — it
is never written to the repo, skills, or logs beyond the transient ssh command.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from . import config


def enabled() -> bool:
    return config.render_host_enabled()


def _ssh_port_flag() -> list[str]:
    return ["-p", config.RENDER_PORT] if config.RENDER_PORT else []


def _scp_port_flag() -> list[str]:
    return ["-P", config.RENDER_PORT] if config.RENDER_PORT else []  # scp uses -P


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"remote cmd failed ({proc.returncode}): {' '.join(cmd[:2])}…\n{tail}")
    return proc


def sh(command: str) -> str:
    """Run a shell command on the render host, return stdout."""
    return _run(["ssh", "-o", "BatchMode=yes", *_ssh_port_flag(),
                 config.RENDER_HOST, command]).stdout


def push(local: str, remote: str) -> None:
    _run(["scp", "-q", "-o", "BatchMode=yes", *_scp_port_flag(),
          str(local), f"{config.RENDER_HOST}:{remote}"])


def pull(remote: str, local: str) -> None:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    _run(["scp", "-q", "-o", "BatchMode=yes", *_scp_port_flag(),
          f"{config.RENDER_HOST}:{remote}", str(local)])


def push_text(text: str, remote: str) -> None:
    """Write text to a remote file (used to ship a worker script)."""
    import tempfile, os
    fd, tmp = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        push(tmp, remote)
    finally:
        os.unlink(tmp)


def cleanup(*remote_paths: str) -> None:
    if remote_paths:
        try:
            sh("rm -rf " + " ".join(f"'{p}'" for p in remote_paths))
        except Exception:
            pass


def _cache_dir() -> str:
    return config.RENDER_CACHE or f"{config.RENDER_TMP.rstrip('/')}/vh_cache"


def remote_exists(path: str) -> bool:
    import shlex
    try:
        return sh(f"test -e {shlex.quote(path)} && echo Y || echo N").strip().endswith("Y")
    except Exception:
        return False


def push_cached(local: str) -> tuple[str, bool]:
    """Upload `local` to the persistent cache (keyed by size+mtime+name) unless
    an identical copy is already there. Returns (remote_path, cache_hit)."""
    import os
    import shlex
    st = os.stat(local)
    key = f"{st.st_size}_{int(st.st_mtime)}_{Path(local).name}"
    cdir = _cache_dir()
    rpath = f"{cdir}/{key}"
    sh("mkdir -p " + shlex.quote(cdir))
    if remote_exists(rpath):
        return rpath, True
    push(str(local), rpath)
    return rpath, False


def clear_cache() -> None:
    import shlex
    try:
        sh("rm -rf " + shlex.quote(_cache_dir()))
    except Exception:
        pass


def run_ffmpeg(argv: list[str], reads: list[str], write: str) -> str:
    """Run one ffmpeg command on the render host.

    `reads` = every local file the command reads (inputs, .ass, card clips) —
    staged through the persistent cache so large sources upload once and are
    reused across calls / re-renders. `write` = local output path. Local paths in
    `argv` (incl. filter strings like subtitles='…') are rewritten to their
    remote paths; the output is pulled back. Font dir -> VH_RENDER_FONTSDIR.
    """
    import os
    import shlex
    jobdir = f"{config.RENDER_TMP.rstrip('/')}/vhjob_{os.getpid()}_{abs(hash((str(write),))) % 100000}"
    sh("mkdir -p " + shlex.quote(jobdir))

    pathmap: dict[str, str] = {}
    hits = 0
    for p in reads:
        rp, hit = push_cached(str(p))
        pathmap[str(p)] = rp
        hits += 1 if hit else 0
    if reads:
        print(f"[remote] {len(reads)} input(s), {hits} cached, "
              f"{len(reads) - hits} uploaded")
    rout = f"{jobdir}/out{Path(write).suffix or '.mp4'}"

    def rewrite(tok: str) -> str:
        for lp, rp in pathmap.items():
            if lp in tok:
                tok = tok.replace(lp, rp)
        if str(write) in tok:
            tok = tok.replace(str(write), rout)
        return tok

    new = [rewrite(t) for t in argv]
    if new and new[0] == config.FFMPEG:
        new[0] = config.RENDER_FFMPEG
    if config.CAPTION_FONTSDIR and config.RENDER_FONTSDIR:
        new = [t.replace(config.CAPTION_FONTSDIR, config.RENDER_FONTSDIR) for t in new]

    try:
        sh(" ".join(shlex.quote(t) for t in new))
        pull(rout, str(write))
    finally:
        cleanup(jobdir)   # job scratch only; the input cache persists
    return str(write)


def ffmpeg_run(argv: list[str], reads: list[str], write: str) -> None:
    """Offload one ffmpeg command to the render host if configured, else local."""
    if enabled():
        run_ffmpeg(argv, reads, write)
    else:
        from .probe import run
        run(argv)


def check() -> str:
    """One-line reachability + capability probe for the configured host."""
    return sh(
        "echo host=$(hostname) arch=$(uname -m); "
        f"{config.RENDER_PYTHON} -c "
        "'import faster_whisper,ctranslate2 as c;print(\"faster_whisper ok cuda=\"+str(c.get_cuda_device_count()))' "
        "2>&1 | tail -1; "
        "command -v ffmpeg >/dev/null && echo ffmpeg=yes || echo ffmpeg=NO"
    ).strip()
