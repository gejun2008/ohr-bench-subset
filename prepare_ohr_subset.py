#!/usr/bin/env python3
"""
OHR-Bench subset builder.

RUN THIS ON A MACHINE THAT CAN REACH huggingface.co (i.e. NOT the company laptop).

What it does
------------
1. Downloads only what is needed from opendatalab/OHR-Bench:
     - data/qas_v2.json           (QA + evidence annotations)
     - data/retrieval_base/gt/**  (human-verified ground truth, per document)
     - pdfs.zip                   (the source PDFs)
   The perturbed-noise folders (formatting_noise_*, semantic_noise_*) are NOT
   downloaded. They are for the noise-transfer study, not for OCR benchmarking.

2. Selects a DETERMINISTIC subset under a page budget, because Azure DI is
   billed and rate-limited per page and the full set is ~8,500 pages.
   Selection is budgeted-maximum-coverage greedy over evidence-source types,
   weighted so that tables / charts / formulas / reading-order are not crowded
   out by plain text. Finance is over-weighted on purpose.

3. Emits a self-contained tree plus a manifest with sha256 per PDF, so the
   company-side run is reproducible and auditable.

IMPORTANT: only pdfs.zip actually requires Hugging Face.
The GitHub repo opendatalab/OHR-Bench already ships qas_v2.json, the full
ground truth, and MinerU's parsed output (~103 MB total). Clone it into the
cache directory and this script will use it without touching HF:

    git clone --depth 1 https://github.com/opendatalab/OHR-Bench.git hf_cache
    python prepare_ohr_subset.py --budget 150 --skip-pdfs   # zero HF access

Usage
-----
    pip install huggingface_hub
    python prepare_ohr_subset.py --list                 # inspect HF repo, download nothing
    python prepare_ohr_subset.py --budget 150           # build a 150-page subset
    python prepare_ohr_subset.py --budget 150 --finance-share 0.45

Output
------
    out/
      manifest.json                      <- commit
      README.md                          <- commit
      data/qas_subset.json               <- commit
      data/retrieval_base/gt/<domain>/*  <- commit
      pdfs/<domain>/<doc>.pdf            <- see packaging note printed at the end
      ohr-subset-pdfs.zip                <- GitHub Release asset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO_ID = "opendatalab/OHR-Bench"
REPO_TYPE = "dataset"

# Evidence-source weights. Verified against the real qas_v2.json, whose values
# are exactly: text, table, formula, chart, reading_order, multi.
# (The README example writes "reading order" with a space; the data does not.)
# Plain text is abundant and cheap to get right; the discriminating signal for
# a layout engine lives in the other five.
SOURCE_WEIGHT = {
    "table": 3.0,
    "chart": 3.0,
    "multi": 2.5,
    "formula": 2.0,
    "reading_order": 2.0,
    "text": 1.0,
}
DEFAULT_WEIGHT = 1.0


def norm_source(s: str) -> str:
    return (s or "unknown").strip().lower().replace(" ", "_")


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------

def hf_list(token: str | None):
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True, token=token)
    rows = []
    for s in info.siblings:
        rows.append((s.rfilename, s.size or 0))
    rows.sort(key=lambda r: -r[1])
    return rows


def fetch(patterns, cache_dir: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download
    print(f"  downloading patterns={patterns}")
    p = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=patterns,
        local_dir=str(cache_dir),
        token=token,
    )
    return Path(p)


def ensure_gt(cache_dir: Path, token: str | None) -> Path:
    """gt may ship as loose files or inside retrieval.zip. Handle both."""
    loose = cache_dir / "data" / "retrieval_base" / "gt"
    if loose.is_dir() and any(loose.rglob("*.json")):
        print("  gt: found loose files")
        return loose

    files = [f for f, _ in hf_list(token)]
    gt_loose = [f for f in files if f.startswith("data/retrieval_base/gt/") and f.endswith(".json")]
    if gt_loose:
        fetch(["data/retrieval_base/gt/**"], cache_dir, token)
        return loose

    zips = [f for f in files if f.endswith(".zip") and "retrieval" in f.lower()]
    if not zips:
        sys.exit("ERROR: could not locate gt data. Run with --list and inspect the repo layout.")
    zname = zips[0]
    print(f"  gt: not loose; extracting from {zname} (only gt/* members)")
    fetch([zname], cache_dir, token)
    zpath = cache_dir / zname
    dest = cache_dir / "data" / "retrieval_base"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        members = [m for m in zf.namelist()
                   if "/gt/" in m.replace("\\", "/") and m.endswith(".json")]
        if not members:
            sys.exit(f"ERROR: no gt/*.json inside {zname}; inspect it manually.")
        for m in members:
            rel = m.replace("\\", "/")
            idx = rel.index("/gt/")
            target = dest / rel[idx + 1:]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"  gt: extracted {len(members)} files")
    return loose


# ----------------------------------------------------------------------------
# indexing
# ----------------------------------------------------------------------------

def build_index(gt_root: Path, qas: list) -> dict:
    """doc_name -> {domain, pages, gt_path, qa: [...], sources: Counter, numeric: int}"""
    docs: dict[str, dict] = {}

    for gt_file in sorted(gt_root.rglob("*.json")):
        domain = gt_file.parent.name
        doc_name = f"{domain}/{gt_file.stem}"
        try:
            pages = json.loads(gt_file.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: unreadable gt {gt_file}: {e}")
            continue
        if not isinstance(pages, list):
            continue
        docs[doc_name] = {
            "domain": domain,
            "pages": len(pages),
            "gt_path": gt_file,
            "qa": [],
            "sources": Counter(),
            "numeric": 0,
            "by_page": defaultdict(list),   # page_idx -> [qa, ...]
            "gt_chars": sum(len(p.get("text", "") or "") for p in pages),
        }

    missing = Counter()
    for q in qas:
        dn = q.get("doc_name")
        if dn not in docs:
            missing[dn] += 1
            continue
        d = docs[dn]
        d["qa"].append(q)
        d["sources"][norm_source(q.get("evidence_source"))] += 1
        if (q.get("answer_form") or "").strip().lower() == "numeric":
            d["numeric"] += 1
        p = q.get("evidence_page_no")
        if p is not None:
            try:
                d["by_page"][int(p)].append(q)
            except (TypeError, ValueError):
                pass

    if missing:
        print(f"  note: {sum(missing.values())} QAs reference {len(missing)} docs with no gt file")
    return docs


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------

def allocate_budget(domains: list[str], budget: int, finance_share: float) -> dict[str, int]:
    """Domains with no QAs get nothing — they cannot be scored, so pages spent
    on them are wasted. In the real data that removes `notes` and `paper`."""
    alloc = {d: 0 for d in domains}
    usable = [d for d in domains if d != "finance"]
    has_finance = "finance" in domains
    if has_finance:
        alloc["finance"] = int(budget * finance_share)
        remaining = budget - alloc["finance"]
    else:
        remaining = budget
    if usable:
        per = remaining // len(usable)
        for d in usable:
            alloc[d] = per
        leftover = remaining - per * len(usable)
        alloc["finance" if has_finance else usable[0]] += leftover
    return alloc


def best_window(d: dict, max_pages: int) -> tuple[int, int, float, int]:
    """
    Pick the contiguous page window of at most max_pages that carries the most
    weighted evidence. Returns (start, end_inclusive, value, qa_count).

    This exists because the finance domain is bimodal: ten 10-K/10-Q documents
    of 83-382 pages hold the real financial-report content, and a single one of
    them exceeds a sensible page budget. Selecting whole documents therefore
    silently biases the subset toward short scanned forms. Azure DI accepts a
    page range, so a window costs only the pages it contains.
    """
    n = d["pages"]
    if n <= max_pages:
        val = sum(SOURCE_WEIGHT.get(s, DEFAULT_WEIGHT) * c for s, c in d["sources"].items())
        return 0, n - 1, val, len(d["qa"])

    page_val = [0.0] * n
    page_qa = [0] * n
    for p, qs in d["by_page"].items():
        if 0 <= p < n:
            page_val[p] = sum(SOURCE_WEIGHT.get(norm_source(q.get("evidence_source")),
                                                DEFAULT_WEIGHT) for q in qs)
            page_qa[p] = len(qs)

    best = (0, max_pages - 1, -1.0, 0)
    run_val = sum(page_val[:max_pages])
    run_qa = sum(page_qa[:max_pages])
    if run_val > best[2]:
        best = (0, max_pages - 1, run_val, run_qa)
    for start in range(1, n - max_pages + 1):
        run_val += page_val[start + max_pages - 1] - page_val[start - 1]
        run_qa += page_qa[start + max_pages - 1] - page_qa[start - 1]
        if run_val > best[2] + 1e-12:      # strict -> earliest window wins ties
            best = (start, start + max_pages - 1, run_val, run_qa)
    return best


def window_sources(d: dict, start: int, end: int) -> Counter:
    c = Counter()
    for p in range(start, end + 1):
        for q in d["by_page"].get(p, []):
            c[norm_source(q.get("evidence_source"))] += 1
    return c


def select_domain(docs_in_domain: list[tuple[str, dict]], page_budget: int,
                  max_doc_pages: int, longform_share: float = 0.5
                  ) -> list[tuple[str, int, int]]:
    """
    Split the domain budget between long-form documents (total pages >
    max_doc_pages) and short ones, then select within each pool.

    Without this split, density-based selection is structurally biased toward
    tiny documents: a 1-page scanned form with 3 QAs scores 3.0, while a
    25-page window of a 10-Q with 20 QAs scores 0.8. On the real data that
    produced a finance subset of 19 scanned forms and zero 10-K/10-Q, which is
    the opposite of what a financial-document evaluation needs. Unused budget
    in either pool rolls over to the other.
    """
    if page_budget <= 0 or not docs_in_domain:
        return []
    longform = [(n, d) for n, d in docs_in_domain if d["pages"] > max_doc_pages]
    short = [(n, d) for n, d in docs_in_domain if d["pages"] <= max_doc_pages]
    if not longform:
        return _select_pool(short, page_budget, max_doc_pages)
    if not short:
        return _select_pool(longform, page_budget, max_doc_pages)

    lf_budget = int(page_budget * longform_share)
    lf = _select_pool(longform, lf_budget, max_doc_pages)
    lf_used = sum(e - s0 + 1 for _, s0, e in lf)
    sh = _select_pool(short, page_budget - lf_used, max_doc_pages)
    sh_used = sum(e - s0 + 1 for _, s0, e in sh)
    leftover = page_budget - lf_used - sh_used
    if leftover > 0:
        taken = {n for n, _, _ in lf + sh}
        extra = _select_pool([(n, d) for n, d in longform if n not in taken],
                             leftover, max_doc_pages)
        lf += extra
    return lf + sh


def _select_pool(docs_in_domain: list[tuple[str, dict]], page_budget: int,
                 max_doc_pages: int) -> list[tuple[str, int, int]]:
    """
    Two phases, both greedy by value density, both deterministic
    (ties broken by doc_name via the sorted() iteration order).

    Phase 1 - budgeted maximum coverage with per-evidence-type saturation caps,
              so no single table-heavy window eats the domain budget.
    Phase 2 - once coverage saturates, spend whatever budget is left on the
              QA-densest remaining windows. Without this the caps leave large
              parts of the budget unused (on the real data, 86 of 150 pages).

    Returns [(doc_name, page_start, page_end_inclusive), ...]
    """
    if page_budget <= 0 or not docs_in_domain:
        return []

    cand = {}
    for name, d in docs_in_domain:
        if not d["qa"]:
            continue
        s, e, val, nqa = best_window(d, max_doc_pages)
        if nqa == 0:
            continue
        cand[name] = {"d": d, "start": s, "end": e, "pages": e - s + 1,
                      "sources": window_sources(d, s, e), "qa": nqa}

    total_sources = Counter()
    for c in cand.values():
        total_sources.update(c["sources"])
    total_pages = sum(c["pages"] for c in cand.values()) or 1
    ratio = min(1.0, page_budget / total_pages)
    cap = {s: max(4, int(round(n * ratio * 2.0))) for s, n in total_sources.items()}

    chosen: list[tuple[str, int, int]] = []
    used = 0
    covered: Counter = Counter()

    # phase 1: coverage
    while cand:
        best_name, best_density = None, 0.0
        for name in sorted(cand):
            c = cand[name]
            if used + c["pages"] > page_budget:
                continue
            val = 0.0
            for s, n in c["sources"].items():
                room = max(0, cap.get(s, 0) - covered[s])
                if room:
                    val += SOURCE_WEIGHT.get(s, DEFAULT_WEIGHT) * min(n, room)
            if val <= 0:
                continue
            density = val / c["pages"]
            if density > best_density + 1e-12:
                best_name, best_density = name, density
        if best_name is None:
            break
        c = cand.pop(best_name)
        chosen.append((best_name, c["start"], c["end"]))
        used += c["pages"]
        covered.update(c["sources"])

    # phase 2: fill remaining budget by QA density
    while cand and used < page_budget:
        best_name, best_density = None, 0.0
        for name in sorted(cand):
            c = cand[name]
            if used + c["pages"] > page_budget:
                continue
            density = c["qa"] / c["pages"]
            if density > best_density + 1e-12:
                best_name, best_density = name, density
        if best_name is None:
            break
        c = cand.pop(best_name)
        chosen.append((best_name, c["start"], c["end"]))
        used += c["pages"]
        covered.update(c["sources"])

    return chosen


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150, help="total PDF pages to select")
    ap.add_argument("--finance-share", type=float, default=0.40)
    ap.add_argument("--longform-share", type=float, default=0.5,
                    help="fraction of each domain budget reserved for documents\n"
                         "longer than --max-doc-pages")
    ap.add_argument("--max-doc-pages", type=int, default=25,
                    help="cap pages taken from any single document; longer "
                         "documents contribute their best contiguous window")
    ap.add_argument("--cache", default="hf_cache")
    ap.add_argument("--out", default="out")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--list", action="store_true", help="list repo files and exit")
    ap.add_argument("--skip-pdfs", action="store_true", help="build metadata only")
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)

    if args.list:
        print(f"{'size (MB)':>12}  file")
        for name, size in hf_list(args.token):
            print(f"{size / 1e6:12.1f}  {name}")
        return

    cache.mkdir(parents=True, exist_ok=True)

    print("[1/5] downloading metadata")
    qas_path = cache / "data" / "qas_v2.json"
    if qas_path.exists():
        print("  qas_v2.json already cached, skipping download")
    else:
        fetch(["data/qas_v2.json"], cache, args.token)
    if not qas_path.exists():
        cands = list(cache.rglob("qas_v2.json"))
        if not cands:
            sys.exit("ERROR: qas_v2.json not found. Run --list and check the path.")
        qas_path = cands[0]
    qas = json.loads(qas_path.read_text(encoding="utf-8"))
    print(f"  qas_v2.json: {len(qas)} QAs")

    gt_root = ensure_gt(cache, args.token)

    print("[2/5] indexing")
    docs = build_index(gt_root, qas)
    by_domain: dict[str, list] = defaultdict(list)
    for name, d in docs.items():
        by_domain[d["domain"]].append((name, d))
    print(f"  {len(docs)} docs, {sum(d['pages'] for d in docs.values())} pages, "
          f"{len(by_domain)} domains")
    for dom in sorted(by_domain):
        items = by_domain[dom]
        print(f"    {dom:<16} docs={len(items):>5}  pages={sum(d['pages'] for _, d in items):>6}  "
              f"qas={sum(len(d['qa']) for _, d in items):>5}")

    print("[3/5] selecting subset")
    alloc = allocate_budget(sorted(by_domain), args.budget, args.finance_share)
    picks: list[tuple[str, int, int]] = []
    for dom in sorted(by_domain):
        got = select_domain(by_domain[dom], alloc.get(dom, 0), args.max_doc_pages,
                            args.longform_share)
        picks.extend(got)
        pg = sum(e - s0 + 1 for _, s0, e in got)
        note = ""
        if not any(d["qa"] for _, d in by_domain[dom]):
            note = "   (no QAs in this domain - skipped)"
        print(f"    {dom:<16} budget={alloc.get(dom, 0):>4}  picked={len(got):>3} windows  "
              f"pages={pg:>4}{note}")
    picks.sort()

    win = {n: (s0, e) for n, s0, e in picks}
    selected = [n for n, _, _ in picks]
    sel_pages = sum(e - s0 + 1 for _, s0, e in picks)

    # only QAs whose evidence page falls inside the selected window are scoreable
    sel_qas = []
    src_mix = Counter()
    n_numeric = 0
    for n in selected:
        s0, e = win[n]
        for p_idx in range(s0, e + 1):
            for q in docs[n]["by_page"].get(p_idx, []):
                sel_qas.append(q)
                src_mix[norm_source(q.get("evidence_source"))] += 1
                if (q.get("answer_form") or "").strip().lower() == "numeric":
                    n_numeric += 1

    n_windowed = sum(1 for n in selected if win[n] != (0, docs[n]["pages"] - 1))
    print(f"  TOTAL: {len(selected)} windows / {sel_pages} pages / {len(sel_qas)} QAs "
          f"({args.budget} page budget, {100*sel_pages/max(1,args.budget):.0f}% used)")
    print(f"  {n_windowed} of them are page windows out of a larger document")
    print(f"  evidence mix: {dict(src_mix)}")
    print(f"  numeric answers: {n_numeric}")

    print("[4/5] writing tree")
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "retrieval_base" / "gt").mkdir(parents=True, exist_ok=True)
    for n in selected:
        d = docs[n]
        s0, e = win[n]
        pages = json.loads(d["gt_path"].read_text(encoding="utf-8"))
        # keep original page_idx values so the window stays aligned with the PDF
        kept = [pg for i, pg in enumerate(pages) if s0 <= int(pg.get("page_idx", i)) <= e]
        tgt = out / "data" / "retrieval_base" / "gt" / d["domain"] / (n.split("/", 1)[1] + ".json")
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "data" / "qas_subset.json").write_text(
        json.dumps(sel_qas, ensure_ascii=False, indent=1), encoding="utf-8")

    # MinerU's parsed output ships in the GitHub repo. Carrying it along costs
    # nothing and gives a second engine to compare Azure against on the exact
    # same pages, without running MinerU anywhere.
    mineru_src = gt_root.parent / "MinerU"
    n_mineru = 0
    if mineru_src.is_dir():
        for n in selected:
            d = docs[n]
            stem = n.split("/", 1)[1]
            src = mineru_src / d["domain"] / f"{stem}.json"
            if src.exists():
                s0, e = win[n]
                mp = json.loads(src.read_text(encoding="utf-8"))
                kept = [pg for i, pg in enumerate(mp)
                        if s0 <= int(pg.get("page_idx", i)) <= e]
                tgt = out / "data" / "retrieval_base" / "MinerU" / d["domain"] / f"{stem}.json"
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
                n_mineru += 1
        print(f"  MinerU baseline carried along for {n_mineru}/{len(selected)} docs")

    pdf_entries = {}
    if not args.skip_pdfs:
        print("[5/5] downloading pdfs.zip and extracting selected documents")
        zpath = cache / "pdfs.zip"
        if zpath.exists():
            print("  pdfs.zip already cached, skipping download")
        else:
            fetch(["pdfs.zip"], cache, args.token)
        if not zpath.exists():
            cands = list(cache.rglob("pdfs.zip"))
            if not cands:
                sys.exit("ERROR: pdfs.zip not found. Run --list.")
            zpath = cands[0]
        want = {n.split("/", 1)[1].lower(): n for n in selected}
        with zipfile.ZipFile(zpath) as zf:
            for m in zf.namelist():
                if not m.lower().endswith(".pdf"):
                    continue
                stem = Path(m).stem.lower()
                if stem not in want:
                    continue
                n = want[stem]
                tgt = out / "pdfs" / docs[n]["domain"] / (n.split("/", 1)[1] + ".pdf")
                tgt.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(tgt, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        for n in selected:
            p = out / "pdfs" / docs[n]["domain"] / (n.split("/", 1)[1] + ".pdf")
            if p.exists():
                pdf_entries[n] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
            else:
                print(f"  WARN: no PDF found in pdfs.zip for {n}")
    else:
        print("[5/5] --skip-pdfs set, not downloading pdfs.zip")

    manifest = {
        "source": {"repo": REPO_ID, "repo_type": REPO_TYPE},
        "license_note": "OHR-Bench is released for research purposes only, not for "
                        "commercial use. Confirm internal legal sign-off before use.",
        "selection": {
            "algorithm": "budgeted-max-coverage greedy over evidence_source, "
                         "deterministic tie-break by doc_name",
            "page_budget": args.budget,
            "max_doc_pages": args.max_doc_pages,
            "note": "Selection unit is a contiguous PAGE WINDOW, not a whole "
                    "document. Submit only page_start..page_end to the OCR "
                    "engine; the gt shipped here is trimmed to match.",
            "finance_share": args.finance_share,
            "source_weight": SOURCE_WEIGHT,
            "per_domain_page_budget": alloc,
        },
        "totals": {
            "windows": len(selected),
            "pages": sel_pages,
            "qas": len(sel_qas),
            "evidence_mix": dict(src_mix),
            "numeric_answers": n_numeric,
        },
        "documents": [
            {
                "doc_name": n,
                "domain": docs[n]["domain"],
                "doc_total_pages": docs[n]["pages"],
                # 0-based, inclusive. Azure DI wants 1-based: pages=f"{start+1}-{end+1}"
                "page_start": win[n][0],
                "page_end": win[n][1],
                "pages": win[n][1] - win[n][0] + 1,
                "is_window": win[n] != (0, docs[n]["pages"] - 1),
                "qa_count": sum(len(docs[n]["by_page"].get(p, []))
                                for p in range(win[n][0], win[n][1] + 1)),
                "evidence_sources": dict(window_sources(docs[n], *win[n])),
                "pdf": pdf_entries.get(n),
            }
            for n in selected
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # package PDFs for a GitHub Release asset
    if pdf_entries:
        zip_out = out / "ohr-subset-pdfs.zip"
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted((out / "pdfs").rglob("*.pdf")):
                zf.write(p, p.relative_to(out))
        mb = zip_out.stat().st_size / 1e6
        print(f"\n  ohr-subset-pdfs.zip = {mb:.1f} MB")
        if mb < 90:
            print("  -> under the 100 MB GitHub file limit: you can commit it directly.")
        elif mb < 1900:
            print("  -> too big to commit. Attach it as a GitHub RELEASE ASSET "
                  "(2 GB/asset, no LFS bandwidth).")
        else:
            print("  -> exceeds the 2 GB release-asset limit. Lower --budget or split the zip.")
        print("  -> either way, do NOT `git add out/pdfs/`; keep the repo text-only.")

    text_bytes = sum(p.stat().st_size for p in out.rglob("*")
                     if p.is_file() and p.suffix == ".json")
    print(f"  committed text (gt + qas + manifest) = {text_bytes / 1e6:.1f} MB")
    print(f"\nDone -> {out.resolve()}")


if __name__ == "__main__":
    main()
