# Experiments (opt-template-title branch)

This branch is the "template title stripping" optimization for bib-dedupe.
It strips editorial-style template prefixes **only for records without DOI**.
All other prep/blocking/matching logic stays the same.

## Run from repo root

All commands below assume you are at the repo root.

## Dataset layout (required)

Place datasets under `experiments/` so others can run after cloning. Example:

```
experiments/
  exp_mis/
    mis-quarterly/
      data/
        search/
          CROSSREF.bib
          DBLP.bib
          pdfs.bib
      baseline/
        records_pre_merged.csv
        merged_record_ids.csv
      mixed/
        records_pre_merged.csv
        merged_record_ids.csv
      pdf_only/
        records_pre_merged.csv
        merged_record_ids.csv
```

Minimum required files per use case:
- Multi-tool evaluation: `records_pre_merged.csv` and `merged_record_ids.csv` in the target subset directory.
- Single-tool evaluation: `data/search/` with `CROSSREF.bib`, `DBLP.bib`, and `pdfs.bib`.

## Single-tool evaluation (bib-dedupe only)

Use `experiments/low_quality_ds_evaluation.py` to evaluate only bib-dedupe.
It can also generate `records_pre_merged.csv` / `merged_record_ids.csv` for subsets.

Example:
```
python experiments/low_quality_ds_evaluation.py exp_mis/mis-quarterly
```

Optional subset:
```
python experiments/low_quality_ds_evaluation.py exp_mis/mis-quarterly pdf_only
```

Outputs:
- `experiments_output/output_<dataset>/evaluation.csv`
- `experiments_output/output_<dataset>/current_results.md`

## Multi-tool evaluation (ASReview vs Bib-dedupe vs Buhos)

Run one of the subset-specific scripts:

Baseline:
```
python experiments/baseline_multitool_evaluation.py exp_mis/mis-quarterly
```

Mixed:
```
python experiments/mixed_multitool_evaluation.py exp_mis/mis-quarterly
```

PDF-only:
```
python experiments/pdf_multitool_evaluation.py exp_mis/mis-quarterly
```

Outputs (per subset):
- `experiments_output/output_<dataset>_multiTool/<subset>/evaluation.csv`
- `experiments_output/output_<dataset>_multiTool/<subset>/current_results.md`
- `experiments_output/output_<dataset>_multiTool/<subset>/false_positive_multitools.csv`

Notes:
- `<dataset>` is derived from the `exp_*` prefix (e.g., `exp_mis` -> `mis`).