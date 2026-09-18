# OHR-Bench subset — Azure Document Intelligence evaluation

A deterministic, page-budgeted subset of [opendatalab/OHR-Bench](https://github.com/opendatalab/OHR-Bench)
for benchmarking Azure Document Intelligence on a machine without Hugging Face access.

**Licence: OHR-Bench is released for research purposes only, not for commercial use.**
Confirm internal legal sign-off before using this for product selection.
Source PDFs are distributed as a GitHub Release asset, not committed to this repo.

| file | runs on | purpose |
|---|---|---|
| `prepare_ohr_subset.py` | personal machine (HF access) | download, select subset, emit manifest + release zip |
| `run_azure_di.py` | company machine | Azure DI prebuilt-layout → OHR-Bench gt format, resumable |
| `score_ocr.py` | anywhere | edit distance + evidence recall + numeric recall, no LLM needed |
| `EVAL_GUIDE.md` | — | full operating procedure |

Start with `EVAL_GUIDE.md` section 0. Do not skip the proxy pre-flight.

## Quick start

```bash
# personal machine
pip install huggingface_hub
git clone --depth 1 https://github.com/opendatalab/OHR-Bench.git hf_cache
python prepare_ohr_subset.py --budget 250 --max-doc-pages 15 --longform-share 0.6 --skip-pdfs

# company machine
pip install azure-ai-documentintelligence rapidfuzz
export AZURE_DI_ENDPOINT=... AZURE_DI_KEY=...
python run_azure_di.py --root . --dry-run
python run_azure_di.py --root . --limit 3 --domain finance
python run_azure_di.py --root .
python score_ocr.py --root . --pred data/retrieval_base/gt          # self-check, must be perfect
python score_ocr.py --root . --pred data/retrieval_base/azure --dump misses.json
```

## Subset selection

Budgeted maximum coverage, greedy by value density over `evidence_source`
(table 3.0 / chart 3.0 / formula 2.0 / reading order 2.0 / text 1.0), with
per-type saturation caps and deterministic tie-break by `doc_name`. Finance is
over-weighted. Same arguments always produce the same subset; `manifest.json`
records the parameters and a sha256 per PDF.
