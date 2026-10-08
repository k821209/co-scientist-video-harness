"""CivitAI — search models/LoRAs, read their details, download the weights.

Three calls, and each one needs something different. That asymmetry is the whole
reason this module exists: every piece of it was found by hitting the wall.

    search(...)        list/search      civitai.red   no auth
    detail(model_id)   one model        civitai.com   User-Agent **and** token
    download(...)      the weights      CDN           token in the URL, NOT a header

**A plain request with no User-Agent gets 403 on the detail endpoint**, token or
not — the token alone is not enough and the error says "Forbidden", not "set a
UA". `search` works bare, so it is easy to conclude the token is wrong when only
the UA is missing.

**Never send `Authorization` on a download.** The weights are a 307 to
Cloudflare R2, the header follows the redirect, and it collides with R2's
presigned signature: you get 403 and a 0-byte file that looks like a finished
download. Put the token in the query string instead — `download()` does.

`civitai.red` mirrors civitai.com's API and answers the list endpoints without
auth; the detail endpoint is served by civitai.com. Both hosts are overridable
if one of them is blocked or goes away.

Configuration (never hardcode a token):

    export CIVITAI_TOKEN="…"            # or pass token= / write ~/.civitai_token

Quick use:

    from vh import civitai
    for m in civitai.search("qwen", types="LORA", base_model="Qwen 2.1"):
        print(m["name"], m["downloads"], m["files"])
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LIST_HOST = os.environ.get("CIVITAI_LIST_HOST", "https://civitai.red").rstrip("/")
API_HOST = os.environ.get("CIVITAI_API_HOST", "https://civitai.com").rstrip("/")

# A browser-shaped UA. The detail endpoint 403s without one — this is not
# politeness, it is a hard requirement (measured 2026-10-06).
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/140.0 Safari/537.36")

TOKEN_FILE = Path(os.environ.get("CIVITAI_TOKEN_FILE", "~/.civitai_token")).expanduser()


def token(explicit: str | None = None) -> str | None:
    """CIVITAI_TOKEN, else ~/.civitai_token, else None (search still works)."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("CIVITAI_TOKEN")
    if env:
        return env.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip() or None
    return None


def _get(url: str, tok: str | None = None, timeout: int = 40) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _flat(m: dict) -> dict:
    """One model, flattened to the fields you actually choose on."""
    vs = m.get("modelVersions") or []
    v = vs[0] if vs else {}
    return {
        "id": m.get("id"),
        "name": m.get("name"),
        "type": m.get("type"),
        "downloads": (m.get("stats") or {}).get("downloadCount", 0),
        "nsfw": bool(m.get("nsfw")),
        "commercial": m.get("allowCommercialUse"),
        "base_models": sorted({x.get("baseModel") for x in vs if x.get("baseModel")}),
        "version": v.get("name"),
        "version_id": v.get("id"),
        "trained_words": v.get("trainedWords") or [],
        "files": [{"name": f.get("name"),
                   "gb": round((f.get("sizeKB") or 0) / 1048576, 2),
                   "url": f.get("downloadUrl")}
                  for f in (v.get("files") or [])],
        "versions": [{"name": x.get("name"), "base_model": x.get("baseModel"),
                      "id": x.get("id")} for x in vs],
    }


def search(query: str | None = None, types: str = "LORA", base_model: str | None = None,
           sort: str = "Most Downloaded", limit: int = 100, nsfw: bool | None = None,
           pages: int = 1, flat: bool = True) -> list[dict]:
    """Search models. No auth needed.

    `base_model` is filtered HERE, not server-side: passing it as a query
    parameter is unreliable, and the labels are not a controlled vocabulary —
    one family shows up as "Qwen 2.1", "Qwen 2" and "Qwen" at once, so an
    exact server-side match silently drops most of what you wanted. Pass a
    string for exact match or a tuple to accept several spellings; call with
    `base_model=None` first and read `base_models` to learn the labels in use.

    `nsfw=False` drops flagged models; None keeps everything.

    NOTE on `pages`: the API paginates by cursor, and adding `page=` to a
    sorted query returns a DIFFERENT result set rather than the next slice
    (measured: a loop over page=1..3 returned zero of the 8 rows that the
    single unpaged call had just found). Until that is understood, >1 page
    re-requests with a larger `limit` instead of walking pages."""
    want = ((base_model,) if isinstance(base_model, str) else tuple(base_model)) if base_model else None
    q = {"types": types, "sort": sort, "limit": min(limit * max(pages, 1), 200)}
    if query:
        q["query"] = query
    url = f"{LIST_HOST}/api/v1/models?" + urllib.parse.urlencode(q)
    items = _get(url).get("items", [])
    out = []
    for m in items:
        if nsfw is False and m.get("nsfw"):
            continue
        if want and not any((v.get("baseModel") in want) for v in (m.get("modelVersions") or [])):
            continue
        out.append(_flat(m) if flat else m)
    return out


def detail(model_id: int | str, tok: str | None = None) -> dict:
    """Full record for one model — description, every version, file names and sizes.

    Needs BOTH a browser User-Agent and a token. Without the UA it is 403 even
    with a valid token, which reads as an auth failure and is not one."""
    t = token(tok)
    if not t:
        raise RuntimeError("detail() needs a token: set CIVITAI_TOKEN or write ~/.civitai_token")
    try:
        return _get(f"{API_HOST}/api/v1/models/{model_id}", tok=t)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise RuntimeError(
                f"403 on model {model_id}. The token is sent and the User-Agent is set, so this is "
                f"the model's own gate (early access / paid) rather than the UA problem.") from e
        raise


def download(url: str, dest: str, tok: str | None = None, timeout: int = 3600) -> str:
    """Fetch a weights file to `dest`.

    The token goes in the QUERY STRING. Sending it as an `Authorization` header
    breaks the download: the URL 307s to Cloudflare R2, the header rides along,
    and R2 rejects it against its presigned signature — 403, and a 0-byte file
    that every size-free check calls a success. Downloads to `dest + '.part'`
    and renames, so an interrupted fetch cannot be mistaken for a complete one."""
    t = token(tok)
    if t:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={t}"
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    if not shutil.which("curl"):
        raise RuntimeError("download() shells out to curl (resume + redirects); install curl")
    r = subprocess.run(["curl", "-sL", "--fail", "-m", str(timeout), "-A", UA,
                        "-o", str(part), url], capture_output=True, text=True)
    if r.returncode or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download failed ({r.returncode}): {url.split('?')[0]}\n{r.stderr[:300]}")
    part.rename(out)
    return str(out)


def check_safetensors(path: str) -> dict:
    """Read a safetensors header and count tensors — the cheap integrity check.

    A truncated download is still a plausible-looking file of plausible size;
    the header parse is what separates "downloaded" from "usable", and it costs
    one read of the first few KB."""
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n <= 0 or n > 100_000_000:
            raise RuntimeError(f"not a safetensors file (header length {n}): {path}")
        head = json.loads(fh.read(n))
    meta = head.pop("__metadata__", None)
    return {"path": path, "tensors": len(head),
            "gb": round(os.path.getsize(path) / 1e9, 2), "metadata": meta}


def fetch(model_id: int | str, dest_dir: str, version: int | str | None = None,
          tok: str | None = None) -> list[dict]:
    """detail -> pick a version -> download its files -> verify. The usual path.

    `version` is a version id or name; default is the newest (the API lists
    versions newest first)."""
    d = detail(model_id, tok)
    vs = d.get("modelVersions") or []
    if not vs:
        raise RuntimeError(f"model {model_id} has no versions")
    v = vs[0]
    if version is not None:
        v = next((x for x in vs if str(x.get("id")) == str(version) or x.get("name") == version), None)
        if v is None:
            raise RuntimeError(f"version {version!r} not in {[x.get('name') for x in vs]}")
    got = []
    for f in v.get("files") or []:
        p = os.path.join(dest_dir, f["name"])
        if os.path.exists(p) and os.path.getsize(p) > 0:
            got.append({"path": p, "reused": True})
            continue
        download(f["downloadUrl"], p, tok)
        got.append(check_safetensors(p) if p.endswith(".safetensors") else {"path": p})
    return got
