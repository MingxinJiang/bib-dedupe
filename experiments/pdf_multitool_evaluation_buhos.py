#!/usr/bin/env python3
"""
PDF-only dataset multi-tool evaluation (ASReview vs Bib_dedupe vs Buhos)

Usage:
    python pdf_multitool_evaluation_buhos.py [dataset_root]

Example:
    python pdf_multitool_evaluation_buhos.py           # default exp_mis
    python pdf_multitool_evaluation_buhos.py exp_jmis  # specify exp_jmis

Output:
- evaluation.csv, current_results.md, false_positive_multitools.csv
- Output dir: /Users/jiangmingxin/Desktop/bib-dedupe/experiments_output/output_<dataset>_multiTool/pdf_only/
"""

import sys
import json
import subprocess
import shutil
import os
from pathlib import Path
from datetime import datetime
import warnings
import itertools
import pandas as pd
import bibtexparser

warnings.filterwarnings("ignore")

# Project root directory
REPO_ROOT = Path("/Users/jiangmingxin/Desktop/bib-dedupe").resolve()
BASE = REPO_ROOT / "experiments"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bib_dedupe.bib_dedupe import prep, block, match, cluster, merge
from bib_dedupe.dedupe_benchmark import DedupeBenchmarker
from bib_dedupe.constants.fields import ID, DUPLICATE, DUPLICATE_LABEL
from bib_dedupe.constants.fields import TITLE, AUTHOR, YEAR, DOI, PAGES, NUMBER, VOLUME, CONTAINER_TITLE, ABSTRACT


def _ensure_output_dir(dataset_root: str) -> Path:
    out_root = REPO_ROOT / "experiments_output"
    out_root.mkdir(exist_ok=True)
    ds_name = dataset_root.replace("exp_", "") if dataset_root.startswith("exp_") else dataset_root
    out_parent = out_root / f"output_{ds_name}_multiTool"
    out_parent.mkdir(exist_ok=True)
    out_dir = out_parent / "pdf_only"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def _resolve_subset_dir(dataset_root: str) -> Path:
    return (BASE / dataset_root / "pdf_only").resolve()


def _load_dataset(ds_dir: Path) -> pd.DataFrame:
    records_path = ds_dir / "records_pre_merged.csv"
    merged_ids_path = ds_dir / "merged_record_ids.csv"
    if not records_path.exists() or not merged_ids_path.exists():
        raise FileNotFoundError(f"Missing PDF-only dataset files: {records_path} or {merged_ids_path}")
    bench = DedupeBenchmarker(benchmark_path=ds_dir)
    records_df = bench.get_records_for_dedupe()
    return records_df


def _coerce_text_fields(records_df: pd.DataFrame) -> pd.DataFrame:
    text_cols = [TITLE, AUTHOR, DOI, PAGES, NUMBER, VOLUME, YEAR, CONTAINER_TITLE, ABSTRACT]
    for col in text_cols:
        if col in records_df.columns:
            records_df.loc[:, col] = records_df[col].fillna("").astype(str)
    return records_df


def _load_pdfs_entries(dataset_root: str) -> dict:
    base_dir = (BASE / dataset_root).resolve()
    candidates = list(base_dir.rglob("pdfs.bib"))
    pdf_path = None
    for c in candidates:
        try:
            if c.parent.name == "search" and c.parent.parent.name == "data":
                pdf_path = c
                break
        except Exception:
            continue
    if pdf_path is None or not pdf_path.exists():
        return {}
    with pdf_path.open(encoding="utf-8") as f:
        db = bibtexparser.load(f)
    return {str(e.get("ID", "")).strip(): e for e in db.entries}


def _export_false_positive(out_dir: Path, dataset_root: str, records_df: pd.DataFrame, dup_sets: list[list[str]], true_merged_ids: list[list[str]]):
    pred_pairs = set(itertools.chain.from_iterable(itertools.combinations(s, 2) for s in dup_sets))
    true_pairs = set(itertools.chain.from_iterable(itertools.combinations(s, 2) for s in true_merged_ids))
    fp_pairs = pred_pairs - true_pairs
    if not fp_pairs:
        return

    df_idx = records_df.set_index(ID, drop=False)
    pdf_entries = _load_pdfs_entries(dataset_root)

    def _from_pdfs_bib(rid: str) -> dict:
        if rid.startswith("P_"):
            oid = rid[2:]
            entry = pdf_entries.get(oid)
            if entry:
                def get_any(d, keys):
                    for k in keys:
                        if k in d and d[k] is not None:
                            return d[k]
                    return ""
                return {
                    "title": get_any(entry, ["title", "Title"]),
                    "author": get_any(entry, ["author", "Author"]),
                    "doi": get_any(entry, ["doi", "DOI"]),
                    "file": get_any(entry, ["file", "File", "FILE"]),
                }
        r = df_idx.loc[rid]
        return {"title": r.get(TITLE, ""), "author": r.get(AUTHOR, ""), "doi": r.get(DOI, ""), "file": r.get("file", "")}

    # Group into clusters
    fp_clusters: dict[int, set[str]] = {}
    for pair in fp_pairs:
        id1, id2 = sorted(pair)
        found = None
        for cid, ids in fp_clusters.items():
            if id1 in ids or id2 in ids:
                found = cid
                break
        if found is not None:
            fp_clusters[found].update([id1, id2])
        else:
            new_cid = len(fp_clusters)
            fp_clusters[new_cid] = {id1, id2}

    rows = []
    for cid, ids in fp_clusters.items():
        for rid in sorted(ids):
            orig = _from_pdfs_bib(rid)
            row = {
                "cluster_id": cid,
                "id": rid,
                "title": orig.get("title", ""),
                "author": orig.get("author", ""),
                "doi": orig.get("doi", ""),
            }
            row["file"] = orig.get("file", "")
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "false_positive_multitools.csv", index=False, encoding="utf-8")


def _evaluate_bib_dedupe(records_df: pd.DataFrame, ds_dir: Path, out_dir: Path, dataset_root: str) -> dict:
    timestamp = datetime.now()
    records_df = _coerce_text_fields(records_df.copy())
    prepared_df = prep(records_df)
    blocked_df = block(prepared_df)
    matched_df = match(blocked_df)
    dup_sets = cluster(matched_df)
    merged_df = merge(records_df, duplicate_id_sets=dup_sets)

    bench = DedupeBenchmarker(benchmark_path=ds_dir)
    result = bench.compare_dedupe_id(records_df=records_df, merged_df=merged_df, timestamp=timestamp)

    _export_false_positive(out_dir=out_dir, dataset_root=dataset_root, records_df=records_df, dup_sets=dup_sets, true_merged_ids=bench.true_merged_ids)
    return result


def _evaluate_asreview(records_df: pd.DataFrame, ds_dir: Path) -> dict:
    try:
        from asreview.data import ASReviewData
    except Exception:
        return {"error": "ASReview not installed"}
    timestamp = datetime.now()
    records_df = _coerce_text_fields(records_df.copy())
    asdata = ASReviewData(records_df)
    merged_df = asdata.drop_duplicates()
    bench = DedupeBenchmarker(benchmark_path=ds_dir)
    result = bench.compare_dedupe_id(records_df=records_df, merged_df=merged_df, timestamp=timestamp)
    return result


def _evaluate_buhos(records_df: pd.DataFrame, ds_dir: Path) -> dict:
    records_df = _coerce_text_fields(records_df.copy())
    buhos_df = records_df.copy()
    if "id" not in buhos_df.columns:
        buhos_df.insert(0, "id", range(len(buhos_df)))
    buhos_csv_path = REPO_ROOT / "notebooks/buhos/records.csv"
    buhos_csv_path.parent.mkdir(parents=True, exist_ok=True)
    buhos_df.to_csv(buhos_csv_path, index=False)
    timestamp = datetime.now()
    method_name = "by_metadata"
    ruby_script_path = REPO_ROOT / "notebooks/buhos/handle_cli.rb"

    ruby_cmd = ["ruby", str(ruby_script_path), method_name, "records.csv"]
    if shutil.which("rbenv"):
        ruby_cmd = ["rbenv", "exec", "ruby", str(ruby_script_path), method_name, "records.csv"]
    env = dict(os.environ)
    env["PATH"] = f"{Path.home()}/.rbenv/bin:{Path.home()}/.rbenv/shims:" + env.get("PATH", "")
    try:
        result = subprocess.run(
            ruby_cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return {"error": "Ruby not installed"}

    if result.returncode != 0:
        return {"error": f"Ruby script error: {result.stderr.strip()}"}

    try:
        duplicates = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "Buhos output not valid JSON"}

    matched_df = pd.DataFrame(columns=[f"{ID}_1", f"{ID}_2", DUPLICATE_LABEL])
    rows_to_add = []
    id_map = dict(zip(buhos_df["id"].astype(int), buhos_df[ID]))
    for pair in duplicates:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            id_1 = id_map.get(int(pair[0]))
            id_2 = id_map.get(int(pair[1]))
        except Exception:
            continue
        if id_1 is None or id_2 is None:
            continue
        rows_to_add.append(
            {
                f"{ID}_1": id_1,
                f"{ID}_2": id_2,
                "search_set_1": "",
                "search_set_2": "",
                DUPLICATE_LABEL: DUPLICATE,
            }
        )

    if rows_to_add:
        matched_df = pd.concat([matched_df, pd.DataFrame(rows_to_add)], ignore_index=True)

    duplicate_id_sets = cluster(matched_df)
    merged_df = merge(records_df, duplicate_id_sets=duplicate_id_sets)
    bench = DedupeBenchmarker(benchmark_path=ds_dir)
    result = bench.compare_dedupe_id(records_df=records_df, merged_df=merged_df, timestamp=timestamp)
    return result


def _append_evaluation_csv(out_dir: Path, package: str, dataset: str, result: dict):
    row = {
        "package": package,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset,
        "TP": result.get("TP", 0),
        "FP": result.get("FP", 0),
        "FN": result.get("FN", 0),
        "TN": result.get("TN", 0),
        "false_positive_rate": result.get("false_positive_rate", 0.0),
        "specificity": result.get("specificity", 0.0),
        "sensitivity": result.get("sensitivity", 0.0),
        "precision": result.get("precision", 0.0),
        "f1": result.get("f1", 0.0),
        "runtime": result.get("runtime", ""),
    }
    eval_csv = out_dir / "evaluation.csv"
    if eval_csv.exists():
        df = pd.read_csv(eval_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(eval_csv, index=False)


def _write_current_results(out_dir: Path, dataset_label: str, bib_result: dict, asr_result: dict, buhos_result: dict):
    def row_of(tool: str, r: dict) -> dict:
        if not r or "error" in r:
            return {"tool": tool, "FP": "-", "TP": "-", "FN": "-", "TN": "-", "false_positive_rate": "-", "sensitivity": "-", "precision": "-", "f1": "-"}
        return {
            "tool": tool,
            "FP": r.get("FP", 0),
            "TP": r.get("TP", 0),
            "FN": r.get("FN", 0),
            "TN": r.get("TN", 0),
            "false_positive_rate": round(float(r.get("false_positive_rate", 0.0)), 4),
            "sensitivity": round(float(r.get("sensitivity", 0.0)), 4),
            "precision": round(float(r.get("precision", 0.0)), 4),
            "f1": round(float(r.get("f1", 0.0)), 4),
        }

    summary_df = pd.DataFrame([
        row_of("bib-dedupe", bib_result),
        row_of("asreview", asr_result),
        row_of("buhos", buhos_result),
    ], columns=["tool", "FP", "TP", "FN", "TN", "false_positive_rate", "sensitivity", "precision", "f1"])

    def detail_row(tool: str, r: dict) -> dict:
        if not r or "error" in r:
            return {"TP": "-", "FP": "-", "FN": "-", "TN": "-", "runtime": "-", "false_positive_rate": "-", "specificity": "-", "sensitivity": "-", "precision": "-", "f1": "-", "dataset": dataset_label, "tool": tool}
        return {
            "TP": r.get("TP", 0),
            "FP": r.get("FP", 0),
            "FN": r.get("FN", 0),
            "TN": r.get("TN", 0),
            "runtime": r.get("runtime", ""),
            "false_positive_rate": round(float(r.get("false_positive_rate", 0.0)), 6),
            "specificity": round(float(r.get("specificity", 0.0)), 6),
            "sensitivity": round(float(r.get("sensitivity", 0.0)), 6),
            "precision": round(float(r.get("precision", 0.0)), 6),
            "f1": round(float(r.get("f1", 0.0)), 6),
            "dataset": dataset_label,
            "tool": tool,
        }

    detail_df = pd.DataFrame([
        detail_row("bib-dedupe", bib_result),
        detail_row("asreview", asr_result),
        detail_row("buhos", buhos_result),
    ], columns=["TP", "FP", "FN", "TN", "runtime", "false_positive_rate", "specificity", "sensitivity", "precision", "f1", "dataset", "tool"])

    md_lines = []
    md_lines.append(f"## {dataset_label} PDF Dataset Evaluation Report")
    md_lines.append("")
    md_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("## Tool Performance Comparison")
    md_lines.append("")
    md_lines.append(summary_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Detailed Results")
    md_lines.append("")
    md_lines.append(detail_df.to_markdown(index=False))

    (out_dir / "current_results.md").write_text("\n".join(md_lines), encoding="utf-8")


def main():
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "exp_mis"
    out_dir = _ensure_output_dir(dataset_root)
    ds_dir = _resolve_subset_dir(dataset_root)

    records_df = _load_dataset(ds_dir)

    bib_result = _evaluate_bib_dedupe(records_df, ds_dir, out_dir, dataset_root)
    _append_evaluation_csv(out_dir, "bib-dedupe", "pdf_only", bib_result)

    asr_result = _evaluate_asreview(records_df, ds_dir)
    if "error" not in asr_result:
        _append_evaluation_csv(out_dir, "asreview", "pdf_only", asr_result)

    buhos_result = _evaluate_buhos(records_df, ds_dir)
    if "error" not in buhos_result:
        _append_evaluation_csv(out_dir, "buhos", "pdf_only", buhos_result)
    else:
        print(f"Buhos skipped: {buhos_result.get('error')}")

    _write_current_results(out_dir, f"{dataset_root}/pdf_only", bib_result, asr_result, buhos_result)

    print(f"Complete. Results written to: {out_dir}")


if __name__ == "__main__":
    main()
