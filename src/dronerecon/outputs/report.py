"""Hash-chained, signed quality report.

Every pipeline stage adds an entry: what it did, key figures, warnings and the
SHA-256 of each input and output file. Each entry stores the hash of the one
before it, so removing, reordering or editing any entry breaks the chain. The
final document is signed with Ed25519; the public key travels inside the report
so anyone can verify it, while only the holder of the private key can produce
it. `verify_report` checks the chain, the signature and (optionally) that the
output files on disk still match their recorded hashes.

The same content is rendered as a self-contained HTML page for people.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from dronerecon import __version__

GENESIS = "0" * 64


# --------------------------------------------------------------------------- canonical form & hashing

def _clean(v: Any) -> Any:
    """JSON-safe, deterministic: NaN/inf -> None, tuples -> lists, numpy scalars -> Python."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


json_safe = _clean  # public name: make any value strict-JSON safe (browsers reject NaN/Infinity)


def canonical(obj: Any) -> bytes:
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return hashlib.sha256(canonical(body)).hexdigest()


# --------------------------------------------------------------------------- keys

def generate_key(path: str | Path) -> Ed25519PrivateKey:
    """Create a signing key and store it (PEM, unencrypted). Keep this file private."""
    key = Ed25519PrivateKey.generate()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    return key


def load_key(path: str | Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{path} is not an Ed25519 private key")
    return key


def _public_b64(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode()


# --------------------------------------------------------------------------- building

@dataclass
class ReportBuilder:
    title: str
    run_id: str
    config: dict
    clock: Callable[[], str] = field(default=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    entries: list[dict] = field(default_factory=list)

    def add(self, stage: str, summary: str, metrics: dict | None = None, warnings: tuple[str, ...] | list = (),
            inputs: list[str | Path] = (), outputs: list[str | Path] = (), root: str | Path | None = None) -> dict:
        def files(paths):
            out = []
            for p in paths:
                pp = Path(p)
                name = str(pp.relative_to(root)) if root is not None and pp.is_absolute() else pp.as_posix()
                out.append({"path": Path(name).as_posix(), "sha256": sha256_file(pp), "bytes": pp.stat().st_size})
            return out

        entry = {"index": len(self.entries), "stage": stage, "time": self.clock(), "summary": summary,
                 "metrics": _clean(metrics or {}), "warnings": list(warnings),
                 "inputs": files(inputs), "outputs": files(outputs),
                 "prev_hash": self.entries[-1]["entry_hash"] if self.entries else GENESIS}
        entry["entry_hash"] = _entry_hash(entry)
        self.entries.append(entry)
        return entry

    def finalize(self, key: Ed25519PrivateKey, overview: dict | None = None) -> dict:
        doc = {
            "format": "dronerecon-quality-report/1",
            "title": self.title, "run_id": self.run_id, "created": self.clock(),
            "software": {"name": "dronerecon", "version": __version__},
            "config_sha256": hashlib.sha256(canonical(self.config)).hexdigest(),
            "overview": _clean(overview or {}),
            "entries": self.entries,
            "chain_head": self.entries[-1]["entry_hash"] if self.entries else GENESIS,
            "signature": {"algorithm": "Ed25519", "public_key": _public_b64(key)},
        }
        doc["signature"]["value"] = base64.b64encode(key.sign(canonical(_unsigned(doc)))).decode()
        return doc


def _unsigned(doc: dict) -> dict:
    d = dict(doc)
    d["signature"] = {k: v for k, v in doc["signature"].items() if k != "value"}
    return d


# --------------------------------------------------------------------------- verification

@dataclass(frozen=True)
class ReportCheck:
    ok: bool
    chain_ok: bool
    signature_ok: bool
    files_ok: bool | None  # None when files were not checked
    problems: tuple[str, ...]


def verify_report(doc: dict, files_root: str | Path | None = None, trusted_public_key: str | None = None) -> ReportCheck:
    problems: list[str] = []
    prev = GENESIS
    chain_ok = True
    for i, e in enumerate(doc.get("entries", [])):
        if e.get("index") != i:
            chain_ok = False
            problems.append(f"entry {i} is out of order")
        if e.get("prev_hash") != prev:
            chain_ok = False
            problems.append(f"entry {i} ({e.get('stage')}) does not follow the entry before it")
        if _entry_hash(e) != e.get("entry_hash"):
            chain_ok = False
            problems.append(f"entry {i} ({e.get('stage')}) was modified after it was recorded")
        prev = e.get("entry_hash")
    if doc.get("chain_head") != prev:
        chain_ok = False
        problems.append("the chain head does not match the last entry (entries added or removed)")

    sig = doc.get("signature", {})
    signature_ok = False
    try:
        pub_b64 = sig["public_key"]
        if trusted_public_key is not None and pub_b64 != trusted_public_key:
            problems.append("signed with a different key than the one you trust")
        else:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(pub_b64)).verify(
                base64.b64decode(sig["value"]), canonical(_unsigned(doc)))
            signature_ok = True
    except (KeyError, ValueError, InvalidSignature):
        problems.append("the signature does not match the report content")

    files_ok = None
    if files_root is not None:
        files_ok = True
        root = Path(files_root)
        for e in doc.get("entries", []):
            for f in e.get("outputs", []):
                p = root / f["path"]
                if not p.is_file():
                    files_ok = False
                    problems.append(f"{f['path']} is missing")
                elif sha256_file(p) != f["sha256"]:
                    files_ok = False
                    problems.append(f"{f['path']} has changed since the report was signed")
    ok = chain_ok and signature_ok and files_ok is not False
    return ReportCheck(ok, chain_ok, signature_ok, files_ok, tuple(problems))


def write_report(folder: str | Path, doc: dict, reliability_svg: str | None = None) -> tuple[Path, Path]:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "quality_report.json"
    js.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    page = out / "quality_report.html"
    page.write_text(render_html(doc, reliability_svg), encoding="utf-8")
    return js, page


# --------------------------------------------------------------------------- HTML

def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, (list, dict)):
        return html.escape(json.dumps(v))
    return html.escape(str(v))


def render_html(doc: dict, reliability_svg: str | None = None) -> str:
    e = html.escape
    ov = doc.get("overview", {})
    trust = ov.get("trust", {})
    n_warn = sum(len(x["warnings"]) for x in doc["entries"])
    verdict = ov.get("verdict", "Completed with notes" if n_warn else "Completed")

    bar = ""
    if trust:
        parts = [("measured", "var(--ok)"), ("inferred", "var(--warn)"), ("unknown", "var(--bad)")]
        bar = '<div class="bar">' + "".join(
            f'<span style="width:{100 * float(trust.get(k, 0) or 0):.2f}%;background:{c}" '
            f'title="{k.capitalize()} {100 * float(trust.get(k, 0) or 0):.0f}%"></span>' for k, c in parts) + "</div>"
        bar += '<div class="legend">' + "".join(
            f'<span><i style="background:{c}"></i>{k.capitalize()} {100 * float(trust.get(k, 0) or 0):.0f}%</span>'
            for k, c in parts) + "</div>"

    key_rows = "".join(f"<tr><th>{e(str(k).replace('_', ' ').capitalize())}</th><td>{_fmt(v)}</td></tr>"
                       for k, v in ov.items() if k not in ("trust", "verdict"))
    stages = []
    for x in doc["entries"]:
        warn = "".join(f"<li>{e(w[0].upper() + w[1:])}</li>" for w in x["warnings"])
        metrics = "".join(f"<tr><th>{e(k.replace('_', ' '))}</th><td>{_fmt(v)}</td></tr>" for k, v in x["metrics"].items())
        files = "".join(f"<tr><td>{e(f['path'])}</td><td class='mono'>{f['sha256'][:16]}…</td></tr>" for f in x["outputs"])
        stages.append(
            f'<section class="stage"><header><span class="n">{x["index"] + 1}</span><h3>{e(x["stage"])}</h3>'
            f'<time>{e(x["time"])}</time></header><p>{e(x["summary"])}</p>'
            + (f'<ul class="warn">{warn}</ul>' if warn else "")
            + (f'<table class="kv">{metrics}</table>' if metrics else "")
            + (f'<details><summary>Output files ({len(x["outputs"])})</summary><table class="files">{files}</table>'
               f'</details>' if files else "")
            + f'<div class="hash mono">entry {x["entry_hash"][:16]}… ← {x["prev_hash"][:16]}…</div></section>')

    rel = f'<section class="card"><h2>Is the stated uncertainty honest?</h2>{reliability_svg}' \
          f'<p class="muted">If the blue line follows the diagonal, a stated ±1σ can be taken at face value.</p>' \
          f'</section>' if reliability_svg else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quality Report</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--text:#1d2330;--muted:#5b6475;--line:#e3e6eb;--ok:#2ea05a;--warn:#e6a028;--bad:#c83c3c;--accent:#2f6fdf}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11141a;--card:#181c24;--text:#e6e9ef;--muted:#a2aab8;--line:#272c36}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:16px;margin:0 0 12px}}h3{{font-size:15px;margin:0}}
.muted{{color:var(--muted)}}.mono{{font-family:ui-monospace,Consolas,monospace;font-size:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}}
.verdict{{display:inline-block;padding:3px 10px;border-radius:999px;font-size:13px;font-weight:600;background:#e8f0fd;color:var(--accent)}}
.bar{{display:flex;height:14px;border-radius:7px;overflow:hidden;margin:8px 0}}
.legend{{display:flex;gap:16px;font-size:13px;color:var(--muted);flex-wrap:wrap}}.legend i{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:4px 8px 4px 0;vertical-align:top}}
.kv th{{color:var(--muted);font-weight:500;width:45%}}
.stage{{border-left:2px solid var(--line);padding:4px 0 16px 20px;margin-left:12px;position:relative}}
.stage header{{display:flex;align-items:baseline;gap:10px}}.stage time{{margin-left:auto;color:var(--muted);font-size:12px}}
.stage .n{{position:absolute;left:-13px;top:2px;width:24px;height:24px;border-radius:50%;background:var(--card);border:2px solid var(--line);font-size:12px;display:flex;align-items:center;justify-content:center}}
.warn{{margin:6px 0;padding-left:18px;color:var(--warn)}}.warn li{{color:var(--text)}}
.hash{{color:var(--muted);margin-top:6px}}details{{margin-top:6px}}summary{{cursor:pointer;color:var(--muted);font-size:13px}}
svg{{max-width:100%;height:auto}}
</style></head><body><main>
<p class="muted">Quality report · {e(doc["run_id"])}</p>
<h1>{e(doc["title"])}</h1>
<p><span class="verdict">{e(verdict)}</span> <span class="muted">· {len(doc["entries"])} stages · {n_warn} notes · created {e(doc["created"])}</span></p>
<section class="card"><h2>How much of the model can be measured</h2>{bar or '<p class="muted">No trust summary recorded.</p>'}
<table class="kv">{key_rows}</table></section>
{rel}
<section class="card"><h2>What happened, step by step</h2>{''.join(stages)}</section>
<section class="card"><h2>Integrity</h2>
<p>Each step above is chained to the one before it by its SHA-256 hash, and the whole report is signed ({e(doc["signature"]["algorithm"])}).
Editing, removing or reordering anything, or changing an output file, is detected by <span class="mono">verify_report</span>.</p>
<table class="kv"><tr><th>Chain head</th><td class="mono">{e(doc["chain_head"])}</td></tr>
<tr><th>Public key</th><td class="mono">{e(doc["signature"]["public_key"])}</td></tr>
<tr><th>Configuration hash</th><td class="mono">{e(doc["config_sha256"])}</td></tr>
<tr><th>Software</th><td>{e(doc["software"]["name"])} {e(doc["software"]["version"])}</td></tr></table></section>
</main></body></html>"""
