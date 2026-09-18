#!/usr/bin/env python3
"""
Score parsed output against OHR-Bench ground truth. No LLM, no retriever,
no GPU — runs anywhere, and is deterministic.

Three metrics, in increasing order of how much they actually matter to you:

1. Normalised edit distance (E.D.) per page, lower is better.
   Comparable in kind to the E.D. column in the OHR-Bench results table.
   Coarse: punishes harmless formatting drift alongside real errors.

2. Evidence recall. For every QA, check whether the annotated
   `evidence_context` survives in the parsed text of `evidence_page_no`.
   Broken out by evidence_source (text / table / formula / chart / reading
   order). This is the metric that tells you WHERE the engine fails.

3. Numeric answer recall. For QAs with answer_form == "Numeric", check
   whether the answer value survives in the parsed page. This is the closest
   proxy available for "did any number get mangled" — treat a miss here as
   far more serious than a few points of edit distance.

Usage
-----
    pip install rapidfuzz
    python score_ocr.py --root . --pred data/retrieval_base/azure
    python score_ocr.py --root . --pred data/retrieval_base/azure \
                        --baseline data/retrieval_base/gt      # sanity check: should be perfect
    python score_ocr.py --root . --pred ... --dump misses.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz import fuzz
    HAVE_RF = True
except ImportError:  # pragma: no cover
    HAVE_RF = False

FUZZ_THRESHOLD = 90  # partial_ratio above this counts as "survived"


def as_list(x) -> list:
    """evidence_context and evidence_page_no are sometimes scalars and
    sometimes lists in qas_v2.json (43% and 9% of rows respectively)."""
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def norm(s) -> str:
    """Whitespace/unicode normalisation only. Does NOT strip markdown — the gt
    is markdown too, so structure is part of what we are scoring."""
    if not s:
        return ""
    if not isinstance(s, str):
        s = " ".join(str(x) for x in as_list(s))
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def norm_loose(s) -> str:
    """For evidence matching: also collapse markdown/LaTeX noise that carries
    no information, so we measure content survival rather than syntax."""
    s = norm(s).lower()
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\(mathrm|text|textbf|mathbf|left|right|;|,|!)\b", "", s)
    s = re.sub(r"[{}|*_#\\]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ed(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if HAVE_RF:
        return Levenshtein.normalized_distance(a, b)
    import difflib
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def norm_source(x) -> str:
    return (x or "unknown").strip().lower().replace(" ", "_")


SHORT_EVIDENCE = 25  # below this length, fuzzy matching is not trustworthy


def contains(haystack: str, needle: str) -> bool:
    """Exact substring for short needles; fuzzy only when there is enough text
    for partial_ratio to mean something. A lenient matcher here would silently
    turn a real OCR error into a pass, which is the failure mode that matters."""
    if not needle:
        return False
    if needle in haystack:
        return True
    if len(needle) < SHORT_EVIDENCE:
        return False
    if HAVE_RF:
        return fuzz.partial_ratio(needle, haystack) >= FUZZ_THRESHOLD
    return False


# Token-boundary aware: "1234O" must NOT count as a hit for 1234.
NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")


def _val(s: str):
    """Numeric value, so '1,234.00' == '1234' but '1234.5' != '1234'."""
    try:
        return float(s.replace(",", "").rstrip(".").lstrip("+"))
    except ValueError:
        return None


def numeric_present(page_text: str, answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return False
    m = NUM_RE.search(a)
    if not m:
        return contains(norm_loose(page_text), norm_loose(answer))
    target = _val(m.group(0))
    if target is None:
        return False
    for cand in NUM_RE.findall(page_text):
        v = _val(cand)
        if v is not None and v == target:
            return True
    return False


def load_pages(path: Path) -> dict[int, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for i, p in enumerate(data):
        idx = p.get("page_idx", i)
        out[int(idx)] = p.get("text", "") or ""
    return out


def evidence_pass(qas, cache, collect_misses=False):
    """One scoring pass over the QA set against a page cache. Used twice:
    once for the prediction, once for gt (to obtain the ceiling)."""
    by_source = defaultdict(lambda: [0, 0])
    by_domain_src = defaultdict(lambda: [0, 0])
    numeric = [0, 0]
    numeric_by_domain = defaultdict(lambda: [0, 0])
    misses = []

    for q in qas:
        dn = q.get("doc_name")
        if dn not in cache:
            continue
        pnos = as_list(q.get("evidence_page_no"))
        if not pnos:
            continue
        chunks = []
        for pn in pnos:
            try:
                pn = int(pn)
            except (TypeError, ValueError):
                continue
            t = cache[dn].get(pn)
            if t is None:
                t = cache[dn].get(pn - 1)      # tolerate 1-based annotation
            if t:
                chunks.append(t)
        if not chunks:
            continue
        text = "\n".join(chunks)
        ptext = norm_loose(text)

        src = norm_source(q.get("evidence_source"))
        dom = q.get("doc_type") or dn.split("/")[0]
        evs = [norm_loose(e) for e in as_list(q.get("evidence_context"))]
        evs = [e for e in evs if e]
        # Strict: every snippet must survive. A half-recovered table still
        # yields a wrong answer, so partial credit would flatter the engine.
        hit = bool(evs) and all(contains(ptext, e) for e in evs)
        by_source[src][1] += 1
        by_source[src][0] += int(hit)
        by_domain_src[(dom, src)][1] += 1
        by_domain_src[(dom, src)][0] += int(hit)

        is_num = (q.get("answer_form") or "").strip().lower() == "numeric"
        nhit = None
        if is_num:
            nhit = numeric_present(text, q.get("answers") or "")
            numeric[1] += 1
            numeric[0] += int(nhit)
            numeric_by_domain[dom][1] += 1
            numeric_by_domain[dom][0] += int(nhit)

        if collect_misses and (not hit or (is_num and not nhit)):
            misses.append({
                "ID": q.get("ID"), "doc_name": dn,
                "page": q.get("evidence_page_no"),
                "evidence_source": src, "answer_form": q.get("answer_form"),
                "evidence_hit": hit, "numeric_hit": nhit,
                "answer": q.get("answers"),
                "evidence_context": norm(q.get("evidence_context"))[:300],
            })

    return {"by_source": by_source, "by_domain_src": by_domain_src,
            "numeric": numeric, "numeric_by_domain": numeric_by_domain,
            "misses": misses}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--pred", required=True, help="parsed-output dir, e.g. data/retrieval_base/azure")
    ap.add_argument("--gt", default="data/retrieval_base/gt")
    ap.add_argument("--qas", default="data/qas_subset.json")
    ap.add_argument("--dump", default=None, help="write per-QA misses to this JSON")
    args = ap.parse_args()

    if not HAVE_RF:
        print("WARNING: rapidfuzz not installed; falling back to difflib. Numbers are "
              "still usable but not comparable to published E.D. values.\n")

    root = Path(args.root)
    pred_root = Path(args.pred) if Path(args.pred).is_absolute() else root / args.pred
    gt_root = Path(args.gt) if Path(args.gt).is_absolute() else root / args.gt
    qas = json.loads((root / args.qas).read_text(encoding="utf-8"))

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    docs = [d["doc_name"] for d in manifest["documents"]]

    # ---- metric 1: edit distance -------------------------------------------
    per_domain_ed = defaultdict(list)
    pred_cache: dict[str, dict[int, str]] = {}
    gt_cache: dict[str, dict[int, str]] = {}
    skipped = []

    for dn in docs:
        domain, stem = dn.split("/", 1)
        pf = pred_root / domain / f"{stem}.json"
        gf = gt_root / domain / f"{stem}.json"
        if not pf.exists() or not gf.exists():
            skipped.append(dn)
            continue
        p, g = load_pages(pf), load_pages(gf)
        pred_cache[dn], gt_cache[dn] = p, g
        for idx, gtext in g.items():
            per_domain_ed[domain].append(ed(norm(p.get(idx, "")), norm(gtext)))

    print("=" * 74)
    print("1. NORMALISED EDIT DISTANCE vs ground truth  (0.0 = identical, lower better)")
    print("=" * 74)
    print(f"{'domain':<18}{'pages':>8}{'mean E.D.':>12}{'median':>10}{'p90':>10}")
    all_ed = []
    for dom in sorted(per_domain_ed):
        v = sorted(per_domain_ed[dom])
        all_ed.extend(v)
        n = len(v)
        print(f"{dom:<18}{n:>8}{sum(v)/n:>12.3f}{v[n//2]:>10.3f}{v[int(n*0.9)-1 if n>1 else 0]:>10.3f}")
    if all_ed:
        v = sorted(all_ed)
        n = len(v)
        print(f"{'ALL':<18}{n:>8}{sum(v)/n:>12.3f}{v[n//2]:>10.3f}{v[int(n*0.9)-1]:>10.3f}")
    if skipped:
        print(f"\nskipped {len(skipped)} docs with no prediction: {skipped[:5]}"
              f"{' ...' if len(skipped) > 5 else ''}")

    # ---- metrics 2 & 3: evidence + numeric survival ------------------------
    # Both are measured against a CEILING computed by running the identical
    # pass with gt as the prediction. The ceiling is well below 100%: ~17% of
    # evidence_context strings and ~33% of numeric answers never appear
    # verbatim in the ground truth itself (annotations were written from a
    # different rendering; chart values were read off figures). Raw recall is
    # therefore uninterpretable - only recall/ceiling is.
    pred_res = evidence_pass(qas, pred_cache, collect_misses=True)
    ceil_res = evidence_pass(qas, gt_cache, collect_misses=False)
    misses = pred_res["misses"]

    print()
    print("=" * 74)
    print("2. EVIDENCE RECALL  (did the annotated evidence survive parsing?)")
    print("=" * 74)
    print(f"{'evidence_source':<18}{'hit':>6}{'total':>7}{'raw':>9}"
          f"{'ceiling':>10}{'vs ceiling':>12}")
    order = ["text", "table", "formula", "chart", "reading_order", "multi"]
    keys = [k for k in order if k in pred_res["by_source"]] + \
           [k for k in sorted(pred_res["by_source"]) if k not in order]
    for k in keys:
        h, t = pred_res["by_source"][k]
        ch, ct = ceil_res["by_source"].get(k, (0, 0))
        raw = h / t if t else 0
        ceil = ch / ct if ct else 0
        print(f"{k:<18}{h:>6}{t:>7}{raw:>8.1%}{ceil:>10.1%}"
              f"{(raw / ceil if ceil else 0):>11.1%}")
    th = sum(v[0] for v in pred_res["by_source"].values())
    tt = sum(v[1] for v in pred_res["by_source"].values())
    cth = sum(v[0] for v in ceil_res["by_source"].values())
    ctt = sum(v[1] for v in ceil_res["by_source"].values())
    raw, ceil = (th / tt if tt else 0), (cth / ctt if ctt else 0)
    print(f"{'ALL':<18}{th:>6}{tt:>7}{raw:>8.1%}{ceil:>10.1%}"
          f"{(raw / ceil if ceil else 0):>11.1%}")

    print()
    print("=" * 74)
    print("3. NUMERIC ANSWER RECALL  (the one that maps to your hard requirement)")
    print("=" * 74)
    print(f"{'domain':<18}{'hit':>6}{'total':>7}{'raw':>9}{'ceiling':>10}{'vs ceiling':>12}")
    for dom in sorted(pred_res["numeric_by_domain"]):
        h, t = pred_res["numeric_by_domain"][dom]
        ch, ct = ceil_res["numeric_by_domain"].get(dom, (0, 0))
        raw, ceil = (h / t if t else 0), (ch / ct if ct else 0)
        print(f"{dom:<18}{h:>6}{t:>7}{raw:>8.1%}{ceil:>10.1%}"
              f"{(raw / ceil if ceil else 0):>11.1%}")
    h, t = pred_res["numeric"]
    ch, ct = ceil_res["numeric"]
    raw, ceil = (h / t if t else 0), (ch / ct if ct else 0)
    print(f"{'ALL':<18}{h:>6}{t:>7}{raw:>8.1%}{ceil:>10.1%}"
          f"{(raw / ceil if ceil else 0):>11.1%}")

    print()
    print("worst domains by table-evidence recall (vs ceiling):")
    rows = []
    for (d, src), v in pred_res["by_domain_src"].items():
        if src != "table" or v[1] < 5:
            continue
        cv = ceil_res["by_domain_src"].get((d, src), (0, 0))
        ceil = cv[0] / cv[1] if cv[1] else 0
        raw = v[0] / v[1]
        rows.append((d, v, raw / ceil if ceil else 0))
    rows.sort(key=lambda r: r[2])
    for d, (h, t), norm_r in rows[:5]:
        print(f"  {d:<16}{h:>5}/{t:<5}{norm_r:>8.1%}")

    if args.dump:
        Path(args.dump).write_text(json.dumps(misses, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"\n{len(misses)} misses written to {args.dump}")


if __name__ == "__main__":
    main()
