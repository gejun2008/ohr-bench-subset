#!/usr/bin/env python3
"""
Run Azure Document Intelligence (prebuilt-layout, markdown output) over the
OHR-Bench subset and write results in OHR-Bench's ground-truth format.

RUN THIS ON THE COMPANY MACHINE.

Output format matches data/retrieval_base/gt/<domain>/<doc>.json exactly:
    [{"page_idx": 0, "text": "..."}, {"page_idx": 1, "text": "..."}, ...]

so the result folder is a drop-in replacement for `gt` in the official
OHR-Bench evaluation framework, and works with score_ocr.py here.

Design notes
------------
- prebuilt-layout with output_content_format=markdown is the correct mode:
  OHR-Bench's gt is markdown + LaTeX, so any other mode makes the edit
  distance meaningless (you would be measuring format, not accuracy).
- Results are written per document and skipped if already present, so an
  interrupted run resumes without re-billing pages.
- A page ledger is written to _run_log.json: pages billed, wall time, errors.
  Check it before anyone asks what this cost.

Usage
-----
    pip install azure-ai-documentintelligence>=1.0.0
    export AZURE_DI_ENDPOINT="https://<resource>.cognitiveservices.azure.com/"
    export AZURE_DI_KEY="..."
    python run_azure_di.py --root . --out data/retrieval_base/azure
    python run_azure_di.py --root . --out data/retrieval_base/azure --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

MODEL = "prebuilt-layout"


def make_client(endpoint: str, key: str):
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def analyze(client, pdf_bytes: bytes, page_range: str | None = None):
    """
    Call Azure DI. Handles the 1.x SDK signature drift (body= vs analyze_request=).

    page_range is Azure's 1-based inclusive form, e.g. "6-30". Passing it means
    only those pages are analysed AND only those pages are billed, which is what
    makes windows out of 380-page 10-Ks affordable.
    """
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

    req = AnalyzeDocumentRequest(bytes_source=pdf_bytes)
    kwargs = {"output_content_format": "markdown"}
    if page_range:
        kwargs["pages"] = page_range
    try:
        poller = client.begin_analyze_document(MODEL, body=req, **kwargs)
    except TypeError:
        poller = client.begin_analyze_document(MODEL, analyze_request=req, **kwargs)
    return poller.result()


def split_pages(result, page_offset: int = 0) -> list[dict]:
    """
    Slice the single markdown `content` string into per-page chunks using the
    span offsets Azure reports for each page. Falls back to one page holding
    everything if spans are absent.

    page_offset re-indexes the output back onto the source document's own
    numbering: when pages="6-30" is used, Azure returns pages 0..24 but the gt
    shipped in this subset keeps the original indices 5..29.
    """
    content = result.content or ""
    pages = getattr(result, "pages", None) or []
    out = []
    for i, page in enumerate(pages):
        spans = getattr(page, "spans", None) or []
        if not spans:
            out.append({"page_idx": i + page_offset, "text": ""})
            continue
        chunks = []
        for sp in spans:
            off = sp.get("offset") if isinstance(sp, dict) else sp.offset
            ln = sp.get("length") if isinstance(sp, dict) else sp.length
            chunks.append(content[off:off + ln])
        out.append({"page_idx": i + page_offset, "text": "".join(chunks)})
    if not out:
        out = [{"page_idx": page_offset, "text": content}]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="subset root (holds manifest.json and pdfs/)")
    ap.add_argument("--out", default="data/retrieval_base/azure")
    ap.add_argument("--pdf-dir", default=None, help="defaults to <root>/pdfs")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="process at most N docs (smoke test)")
    ap.add_argument("--domain", default=None, help="restrict to one domain, e.g. finance")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify inputs, page counts and credentials; make no billed calls")
    args = ap.parse_args()

    root = Path(args.root)
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else root / "pdfs"
    out_dir = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    docs = manifest["documents"]
    if args.domain:
        docs = [d for d in docs if d["domain"] == args.domain]

    todo, done, missing = [], [], []
    for d in docs:
        name = d["doc_name"]
        stem = name.split("/", 1)[1]
        pdf = pdf_dir / d["domain"] / f"{stem}.pdf"
        tgt = out_dir / d["domain"] / f"{stem}.json"
        start, end = d.get("page_start", 0), d.get("page_end")
        n_pages = d.get("pages", d.get("gt_pages", 0))
        prange = f"{start + 1}-{end + 1}" if d.get("is_window") else None
        if not pdf.exists():
            missing.append(name)
        elif tgt.exists():
            done.append(name)
        else:
            todo.append((name, pdf, tgt, n_pages, start, prange))

    if args.limit:
        todo = todo[:args.limit]

    pages_todo = sum(t[3] for t in todo)
    print(f"documents: {len(docs)}  already done: {len(done)}  to process: {len(todo)}  "
          f"missing pdf: {len(missing)}")
    print(f"pages to bill this run (gt page count): {pages_todo}")
    if missing:
        print(f"  missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    endpoint = os.environ.get("AZURE_DI_ENDPOINT")
    key = os.environ.get("AZURE_DI_KEY")
    if not endpoint or not key:
        sys.exit("ERROR: set AZURE_DI_ENDPOINT and AZURE_DI_KEY")

    if args.dry_run:
        print("dry-run: inputs look fine, no calls made.")
        return
    if not todo:
        print("nothing to do.")
        return

    client = make_client(endpoint, key)
    log = {"model": MODEL, "content_format": "markdown",
           "endpoint_host": endpoint.split("//")[-1].split("/")[0],
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "docs": [], "errors": []}
    lock = Lock()
    t0 = time.time()

    def work(item):
        name, pdf, tgt, gt_pages, offset, prange = item
        started = time.time()
        data = pdf.read_bytes()
        last_err = None
        for attempt in range(4):
            try:
                result = analyze(client, data, prange)
                pages = split_pages(result, offset)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
                return {"doc_name": name, "azure_pages": len(pages), "gt_pages": gt_pages,
                        "page_range": prange or "all",
                        "chars": sum(len(p["text"]) for p in pages),
                        "seconds": round(time.time() - started, 1)}
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt * 5)
        return {"doc_name": name, "error": last_err}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it[0] for it in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                if "error" in r:
                    log["errors"].append(r)
                    print(f"  [{i}/{len(todo)}] FAIL {r['doc_name']}: {r['error'][:120]}")
                else:
                    log["docs"].append(r)
                    flag = "" if r["azure_pages"] == r["gt_pages"] else \
                        f"  !! page mismatch gt={r['gt_pages']} azure={r['azure_pages']}"
                    print(f"  [{i}/{len(todo)}] ok {r['doc_name']} "
                          f"{r['azure_pages']}p {r['chars']}ch {r['seconds']}s{flag}")

    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    log["wall_seconds"] = round(time.time() - t0, 1)
    log["pages_billed_estimate"] = sum(d["azure_pages"] for d in log["docs"])
    log_path = out_dir / "_run_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                existing = [existing]
        except Exception:  # noqa: BLE001
            existing = []
    log_path.write_text(json.dumps(existing + [log], ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\nok={len(log['docs'])} fail={len(log['errors'])} "
          f"pages={log['pages_billed_estimate']} wall={log['wall_seconds']}s")
    print(f"results -> {out_dir}")
    print(f"log     -> {log_path}")
    mism = [d for d in log["docs"] if d["azure_pages"] != d["gt_pages"]]
    if mism:
        print(f"\nWARNING: {len(mism)} documents have a page-count mismatch vs gt. "
              f"Per-page scoring will be misaligned for those. Investigate before "
              f"trusting their numbers.")


if __name__ == "__main__":
    main()
