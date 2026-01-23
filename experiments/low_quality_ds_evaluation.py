#!/usr/bin/env python3
"""
Low-quality dataset evaluation script

Usage:
python experiments/low_quality_ds_evaluation.py <dataset_root> [subset]

Examples (run from repo root):
python experiments/low_quality_ds_evaluation.py exp_mis/mis-quarterly

"""

import warnings
import csv
import re
import sys
from pathlib import Path
from datetime import datetime
import itertools

BASE = Path(__file__).parent
REPO_ROOT = BASE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import bibtexparser
from colrev.loader import load_utils
from bib_dedupe.bib_dedupe import prep, block, match, cluster, merge
from bib_dedupe.dedupe_benchmark import DedupeBenchmarker
from bib_dedupe.constants.fields import ID as FIELD_ID, AUTHOR, TITLE, DOI

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Output helpers (match experiments_output/output_<dataset>)
def _ensure_output_dir(dataset_root: Path) -> Path:
    out_root = REPO_ROOT / "experiments_output"
    out_root.mkdir(exist_ok=True)
    exp_name = dataset_root.parent.name
    ds_name = exp_name.replace("exp_", "") if exp_name.startswith("exp_") else exp_name
    out_dir = out_root / f"output_{ds_name}"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def _append_evaluation_csv(out_dir: Path, subset: str, result: dict) -> None:
    row = {
        "dataset": subset,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
    columns = [
        "dataset",
        "time",
        "TP",
        "FP",
        "FN",
        "TN",
        "false_positive_rate",
        "specificity",
        "sensitivity",
        "precision",
        "f1",
        "runtime",
    ]
    eval_csv = out_dir / "evaluation.csv"
    if eval_csv.exists():
        df = pd.read_csv(eval_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.reindex(columns=columns)
    df.to_csv(eval_csv, index=False)


def _write_current_results(out_dir: Path, dataset_root: Path, results_by_subset: dict) -> None:
    summary_rows = []
    detail_rows = []
    for subset, result in results_by_subset.items():
        summary_rows.append({
            "dataset": subset,
            "FP": result.get("FP", 0),
            "TP": result.get("TP", 0),
            "FN": result.get("FN", 0),
            "TN": result.get("TN", 0),
            "false_positive_rate": round(float(result.get("false_positive_rate", 0.0)), 4),
            "sensitivity": round(float(result.get("sensitivity", 0.0)), 4),
            "precision": round(float(result.get("precision", 0.0)), 4),
            "f1": round(float(result.get("f1", 0.0)), 4),
        })

        detail_rows.append({
            "dataset": subset,
            "TP": result.get("TP", 0),
            "FP": result.get("FP", 0),
            "FN": result.get("FN", 0),
            "TN": result.get("TN", 0),
            "runtime": result.get("runtime", ""),
            "false_positive_rate": round(float(result.get("false_positive_rate", 0.0)), 6),
            "specificity": round(float(result.get("specificity", 0.0)), 6),
            "sensitivity": round(float(result.get("sensitivity", 0.0)), 6),
            "precision": round(float(result.get("precision", 0.0)), 6),
            "f1": round(float(result.get("f1", 0.0)), 6),
            "tool": "bib-dedupe",
        })

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["dataset", "FP", "TP", "FN", "TN", "false_positive_rate", "sensitivity", "precision", "f1"],
    )
    detail_df = pd.DataFrame(
        detail_rows,
        columns=["dataset", "TP", "FP", "FN", "TN", "runtime", "false_positive_rate", "specificity", "sensitivity", "precision", "f1", "tool"],
    )

    md_lines = []
    md_lines.append(f"## {dataset_root.parent.name}/{dataset_root.name} Evaluation Report")
    md_lines.append("")
    md_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append(summary_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Detailed Results")
    md_lines.append("")
    md_lines.append(detail_df.to_markdown(index=False))

    (out_dir / "current_results.md").write_text("\n".join(md_lines), encoding="utf-8")

# Source file mapping - adjust for different datasets
DEFAULT_SRC_MAP = {
    "CROSSREF.bib": ("C", "crossref"),
    "DBLP.bib": ("D", "dblp"), 
    "pdfs.bib": ("P", "pdf"),
}

DEFAULT_PREFIX = {
    "CROSSREF.bib": "C",
    "DBLP.bib": "D",
    "pdfs.bib": "P",
}

class UniversalEvaluator:
    def __init__(self, dataset_path: str, src_map=None, prefix_map=None):
        raw_path = Path(dataset_path)
        if raw_path.is_absolute():
            resolved_root = raw_path
        elif raw_path.parts and raw_path.parts[0] == "experiments":
            resolved_root = (BASE.parent / raw_path).resolve()
        else:
            resolved_root = (BASE / raw_path).resolve()

        # Accept dataset root, data/ directory, or data/search directory.
        if resolved_root.name == "search" and resolved_root.parent.name == "data":
            self.data_dir = resolved_root.parent
            self.search_dir = resolved_root
        elif (resolved_root / "data" / "search").exists():
            self.data_dir = resolved_root / "data"
            self.search_dir = self.data_dir / "search"
        elif (resolved_root / "search").exists() and resolved_root.name == "data":
            self.data_dir = resolved_root
            self.search_dir = resolved_root / "search"
        else:
            self.data_dir = resolved_root / "data"
            self.search_dir = self.data_dir / "search"

        self.dataset_path = resolved_root
        self.dataset_root = self.data_dir.parent
        self.output_dir = _ensure_output_dir(self.dataset_root)
        
        self.src_map = src_map or DEFAULT_SRC_MAP
        self.prefix_map = prefix_map or DEFAULT_PREFIX
        
        # Ensure directory exists
        if not self.search_dir.exists():
            raise FileNotFoundError(f"Search directory not found: {self.search_dir}")
        
        print(f"Evaluator initialized. Dataset: {self.dataset_root}")
        print(f"Search directory: {self.search_dir}")
        print(f"Available .bib files: {list(self.search_dir.glob('*.bib'))}")

        # Source .bib entry cache (by source name -> {orig_ID: entry})
        self.source_entries = {}

    def _load_source_bibs(self):
        """Load .bib files from search/ as dictionary indexed by source name, with normalized string IDs."""
        if self.source_entries:
            return
        # e.g.: {'pdf': 'pdfs.bib', 'crossref': 'CROSSREF.bib', 'dblp': 'DBLP.bib'}
        src_to_file = {v[1]: k for k, v in self.src_map.items()}
        for src_name, fname in src_to_file.items():
            path = self.search_dir / fname
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                db = bibtexparser.load(f)
            # Normalize to string keys to avoid misses from numeric/case differences
            self.source_entries[src_name] = {
                str(e.get("ID", "")).strip(): e for e in db.entries
            }

    def _get_original_fields(self, rec_row: pd.Series, subset: str) -> dict:
        """Get original fields preferentially from source bib files (pdfs.bib / CROSSREF.bib / DBLP.bib);
           If orig_* columns are missing, infer source and original ID from ID prefix P_/C_/D_."""
        self._load_source_bibs()

        # 1) Try using orig_* first; otherwise infer from ID prefix
        orig_id = str(rec_row.get("orig_ID", "") or "").strip()
        orig_src = str(rec_row.get("orig_source", "") or "").strip()
        if not orig_id or not orig_src:
            cur_id = str(
                rec_row.get(FIELD_ID) or rec_row.get("ID") or getattr(rec_row, "ID", "") or ""
            ).strip()

            m = re.match(r"^([CDP])_(.+)$", cur_id)
            if m:
                pref, tail = m.group(1), m.group(2)
                orig_id = tail
                orig_src = {"C": "crossref", "D": "dblp", "P": "pdf"}.get(pref, "")

        entry = self.source_entries.get(orig_src, {}).get(orig_id)

        def get_any(d: dict, keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return ""

        if entry:  # Hit source bib - return "original values"
            title = get_any(entry, ["title", "Title"])
            author = get_any(entry, ["author", "Author"])
            doi = get_any(entry, ["doi", "DOI"])
            file_field = get_any(entry, ["file", "File", "FILE"]) if subset == "pdf_only" else ""
        else:      # Fallback (preserve current df as much as possible)
            title = rec_row.get(TITLE, "")
            author = rec_row.get(AUTHOR, "")
            doi = rec_row.get(DOI, "")
            file_field = ""

        return {"title": title, "author": author, "doi": doi, "file": file_field}

    def bib_to_df(self, bib_path: Path) -> pd.DataFrame:
        """bib to DataFrame"""
        if not bib_path.is_file():
            raise FileNotFoundError(f"File not found: {bib_path}")
        
        records = load_utils.load(filename=bib_path, unique_id_field="ID")
        df = (pd.DataFrame.from_dict(records, orient="index")
              .reset_index()
              .rename(columns={"index": "ID"}))
        df = df.reset_index(drop=True)
        
        # Remove duplicate columns (keep only the first)
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df

    def create_csv_datasets(self):
        print("=== create csv ===")
        
        # Get all bib files
        bib_files = list(self.search_dir.glob("*.bib"))
        if not bib_files:
            raise FileNotFoundError(f"No bib files found in {self.search_dir}.")
        
        # Read all bib files
        dataframes = {}
        for bib_file in bib_files:
            df = self.bib_to_df(bib_file)
            dataframes[bib_file.name] = df
            print(f"read {bib_file.name}: {len(df)} records")
        
        # Create different dataset combinations
        datasets = {}
        
        # If CROSSREF and DBLP exist, create baseline
        if "CROSSREF.bib" in dataframes and "DBLP.bib" in dataframes:
            df_baseline = pd.concat([dataframes["CROSSREF.bib"], dataframes["DBLP.bib"]], ignore_index=True)
            datasets["baseline"] = df_baseline
            print(f"created baseline: {len(df_baseline)} records")
        
        # If pdfs exist, create pdf_only
        if "pdfs.bib" in dataframes:
            datasets["pdf_only"] = dataframes["pdfs.bib"]
            print(f"created pdf_only: {len(dataframes['pdfs.bib'])} records")
        
        # Create mixed (all files)
        all_dfs = list(dataframes.values())
        if len(all_dfs) > 1:
            df_mixed = pd.concat(all_dfs, ignore_index=True)
            datasets["mixed"] = df_mixed
            print(f"created mixed: {len(df_mixed)} records")
        
        # Save CSV files
        for name, df in datasets.items():
            csv_path = self.dataset_root / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"saved {csv_path}")
        
        return datasets

    def filter_records(self):
        print("=== Filtering record ===")
        
        # Read original records
        src_path = self.data_dir / "records.bib"
        if not src_path.exists():
            print(f"Warning: {src_path} does not exist, skipping filtering step")
            return
        
        with src_path.open(encoding="utf-8") as f:
            db = bibtexparser.load(f)

        def write_subset(name, keep_fn):
            out = bibtexparser.bibdatabase.BibDatabase()
            out.entries = [e for e in db.entries if keep_fn(e.get("colrev_origin", ""))]
            path = self.dataset_root / Path(name) / "records.bib"
            path.parent.mkdir(exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                bibtexparser.dump(out, f)
            print(f"created {name}/records.bib: {len(out.entries)} entries")

        # Create subsets
        write_subset("baseline",
                     lambda origin: "CROSSREF.bib" in origin or "DBLP.bib" in origin)
        write_subset("pdf_only",
                     lambda origin: "pdfs.bib" in origin)
        write_subset("mixed",
                     lambda origin: True)

    def generate_merged_record_ids(self, subset):
        """Generate merged record IDs"""
        print(f"=== Generating merged record IDs for {subset} ===")
        
        subset_dir = self.dataset_root / subset
        rec_path = subset_dir / "records.bib"
        out_path = subset_dir / "merged_record_ids.csv"
        
        if not rec_path.exists():
            print(f"Warning: {rec_path} does not exist, skipping merged record ID generation")
            return

        with rec_path.open(encoding="utf-8") as bibfile:
            db = bibtexparser.load(bibfile)

        pattern = re.compile(r"([^;{}\s]+\.bib)/([^;{}\s]+)")
        clusters = {}  # map: original cluster key -> set of prefixed IDs

        # Determine allowed prefixes for current subset
        if subset == "baseline":
            allowed_prefixes = {"C", "D"}
        elif subset == "pdf_only":
            allowed_prefixes = {"P"}
        elif subset == "mixed":
            allowed_prefixes = {"C", "D", "P"}
        else:
            allowed_prefixes = {"C", "D", "P"}

        for entry in db.entries:
            orig = entry.get("colrev_origin", "").replace("\n", "").replace(" ", "")
            if not orig:
                continue

            parts = pattern.findall(orig)
            if not parts:
                continue

            prefixed_ids = []
            for fname, rid in parts:
                prefix = self.prefix_map.get(fname)
                if prefix and prefix in allowed_prefixes:
                    prefixed_ids.append(f"{prefix}_{rid}")
            if not prefixed_ids:
                continue

            key = orig
            clusters.setdefault(key, set()).update(prefixed_ids)

        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["merged_ids"])
            for prefset in clusters.values():
                row = ";".join(sorted(prefset))
                writer.writerow([row])

        print(f"created {out_path}: {len(clusters)} clusters")

    def generate_records_pre_merged(self, subset):
        """Generate pre-merged records"""
        print(f"=== Generating pre-merged records for {subset} ===")
        
        subset_dir = self.dataset_root / subset
        out_path = subset_dir / "records_pre_merged.csv"

        if subset == "baseline":
            bib_files = ["CROSSREF.bib", "DBLP.bib"]
        elif subset == "mixed":
            bib_files = list(self.src_map.keys())
        elif subset == "pdf_only":
            bib_files = ["pdfs.bib"]
        else:
            raise ValueError(f"Unknown subset: {subset}")

        all_dfs = []
        for bib_file in bib_files:
            path = self.search_dir / bib_file
            if not path.exists():
                print(f"Warning: {path} does not exist, skipping")
                continue
                
            prefix, src = self.src_map[bib_file]
            
            with path.open(encoding="utf-8") as fh:
                bib = bibtexparser.load(fh)
            
            df = pd.DataFrame(bib.entries)
            if df.empty:
                continue
                
            df["orig_ID"] = df["ID"]
            df["ID"] = df["orig_ID"].apply(lambda rid: f"{prefix}_{rid}")
            df["orig_source"] = src
            all_dfs.append(df)

        if not all_dfs:
            print(f"Warning: No valid records found for {subset}")
            return

        full = pd.concat(all_dfs, ignore_index=True)
        cols = ["ID", "orig_ID", "orig_source"] + [c for c in full.columns if c not in ("ID", "orig_ID", "orig_source")]
        full = full[cols]

        full.to_csv(out_path, index=False)
        print(f"created {out_path}: {len(full)} records")

    def evaluate_subset(self, subset):
        """Evaluate specified subset"""
        print(f"\n=== Evaluating {subset} ===")
        subset_dir = self.dataset_root / subset

        try:
            bench = DedupeBenchmarker(benchmark_path=subset_dir)
            records_df = bench.get_records_for_dedupe()
            print(f"{datetime.now()} Loaded {len(records_df)} record")
        except Exception as e:
            print(f"Error: unable to load records: {e}")
            return

        if AUTHOR in records_df.columns:
            records_df[AUTHOR] = records_df[AUTHOR].fillna("").astype(str)

        timestamp = datetime.now()
        try:
            df1 = prep(records_df)
            df2 = block(df1)
            df3 = match(df2)
            dup_sets = cluster(df3)   
            merged_df = merge(records_df, duplicate_id_sets=dup_sets)
        except Exception as e:
            print(f"Error: exception during deduplication: {e}")
            return

        try:
            result = bench.compare_dedupe_id(
                records_df=records_df,
                merged_df=merged_df,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"Error: exception during comparison: {e}")
            return

        print(f"\nResults for {subset}:")
        print(f"  TP  = {result['TP']}")
        print(f"  FP  = {result['FP']}")
        print(f"  FN  = {result['FN']}")
        print(f"  TN  = {result.get('TN', 'n/a')}")
        print(f"  Precision     = {result['precision']:.4f}")
        print(f"  Recall        = {result['sensitivity']:.4f}")
        print(f"  Specificity   = {result['specificity']:.4f}")
        print(f"  F1 score      = {result['f1']:.4f}")

        _append_evaluation_csv(self.output_dir, subset, result)

        # Generate false positive and true positive pairs
        pred_pairs = set(itertools.chain.from_iterable(
            itertools.combinations(cluster, 2) for cluster in dup_sets
        ))
        true_pairs = set(itertools.chain.from_iterable(
            itertools.combinations(tc, 2) for tc in bench.true_merged_ids
        ))
        fp_pairs = pred_pairs - true_pairs
        tp_pairs = pred_pairs & true_pairs
        

        df_idx = records_df.set_index(FIELD_ID, drop=False)

        # Read "original field snapshot" from subset directory
        pre_path = subset_dir / "records_pre_merged.csv"
        pre_df = pd.read_csv(pre_path, dtype=str)  # Read all as strings to avoid 001 -> 1
        pre_df.columns = [c.lower() for c in pre_df.columns]  # Normalize column names to lowercase
        # Ensure 'id' column exists (records_pre_merged.csv contains generated prefixed IDs)
        if "id" not in pre_df.columns:
            raise RuntimeError(f"{pre_path} missing 'id' column")
        pre_df = pre_df.set_index("id", drop=False)

        def _orig_from_premerged(rid: str):
            """Get original fields preferentially from records_pre_merged.csv; fallback to _get_original_fields if not found."""
            if rid in pre_df.index:
                row = pre_df.loc[rid]
                return {
                    "title": row.get("title", "") or "",
                    "author": row.get("author", "") or "",
                    "doi": row.get("doi", "") or "",
                    "file": (row.get("file", "") or "") if subset == "pdf_only" else "",
                }
            # Fallback (rare misses): use existing backtracking logic
            return self._get_original_fields(df_idx.loc[rid], subset)


        if subset == "pdf_only":
            probe = records_df[records_df[FIELD_ID].astype(str).str.startswith("P_")].head(1)
            if not probe.empty:
                test = self._get_original_fields(probe.iloc[0], subset)
                assert test["title"] or test["author"], (
                    f"Failed to get original fields from source bib; "
                    f"id={probe.iloc[0][FIELD_ID]}, "
                    f"orig_ID={probe.iloc[0].get('orig_ID','')}, "
                    f"orig_source={probe.iloc[0].get('orig_source','')}"
                )
        
        # Export false positives (output by cluster with original fields)
        if fp_pairs:
            # Aggregate pairs into clusters (connected components approximation: simple merge rules)
            fp_clusters = {}
            for pair in fp_pairs:
                id1, id2 = sorted(pair)
                found_cluster = None
                for cid, ids in fp_clusters.items():
                    if id1 in ids or id2 in ids:
                        found_cluster = cid
                        break
                if found_cluster is not None:
                    fp_clusters[found_cluster].update([id1, id2])
                else:
                    new_cid = len(fp_clusters)
                    fp_clusters[new_cid] = {id1, id2}

            rows = []
            for cid, ids in fp_clusters.items():
                for rid in sorted(ids):
                    r = df_idx.loc[rid]
                    orig = _orig_from_premerged(rid)
                    row = {
                        "cluster_id": cid,
                        "id": rid,
                        "title": orig.get("title", ""),
                        "author": orig.get("author", ""),
                        "doi": orig.get("doi", ""),
                    }
                    if subset == "pdf_only":
                        row["file"] = orig.get("file", "")
                    rows.append(row)
            fp_df = pd.DataFrame(rows)
            out_fp = subset_dir / "false_positives.csv"
            fp_df.to_csv(out_fp, index=False, encoding="utf-8")
            print(f"{datetime.now()} False positives saved to {out_fp}")

        if tp_pairs:
            tp_clusters = {}
            for pair in tp_pairs:
                id1, id2 = sorted(pair)
                found_cluster = None
                for cid, ids in tp_clusters.items():
                    if id1 in ids or id2 in ids:
                        found_cluster = cid
                        break
                if found_cluster is not None:
                    tp_clusters[found_cluster].update([id1, id2])
                else:
                    new_cid = len(tp_clusters)
                    tp_clusters[new_cid] = {id1, id2}

            rows_tp = []
            for cid, ids in tp_clusters.items():
                for rid in sorted(ids):
                    r = df_idx.loc[rid]
                    orig = self._get_original_fields(r, subset)
                    row = {
                        "cluster_id": cid,
                        "id": rid,
                        "title": orig.get("title", ""),
                        "author": orig.get("author", ""),
                        "doi": orig.get("doi", ""),
                    }
                    if subset == "pdf_only":
                        row["file"] = orig.get("file", "")
                    rows_tp.append(row)
            tp_df = pd.DataFrame(rows_tp)
            out_tp = subset_dir / "true_positives.csv"
            tp_df.to_csv(out_tp, index=False, encoding="utf-8")
            print(f"{datetime.now()} True positives saved to {out_tp}")

        return result

    def run_evaluation(self, subsets=None):
        """Run complete evaluation pipeline"""
        if subsets is None:
            # Determine subsets based on available bib files
            available_files = [f.name for f in self.search_dir.glob("*.bib")]
            subsets = []
            
            if "CROSSREF.bib" in available_files and "DBLP.bib" in available_files:
                subsets.append("baseline")
            if "pdfs.bib" in available_files:
                subsets.append("pdf_only")
            if len(available_files) > 1:
                subsets.append("mixed")
        
        print(f"Subsets to be processed: {', '.join(subsets)}")
        
        # Step 1: Create CSV datasets
        self.create_csv_datasets()
        
        # Step 2: Filter records
        self.filter_records()
        
        # Steps 3-5: Generate necessary files and evaluate for each subset
        results_by_subset = {}
        for subset in subsets:
            print(f"\n{'='*50}")
            print(f"Processing subset: {subset}")
            print(f"{'='*50}")
            
            # Step 3: Generate merged record IDs
            self.generate_merged_record_ids(subset)
            
            # Step 4: Generate pre-merged records
            self.generate_records_pre_merged(subset)
            
            # Step 5: Evaluate
            result = self.evaluate_subset(subset)
            if result:
                results_by_subset[subset] = result

        if results_by_subset:
            _write_current_results(self.output_dir, self.dataset_root, results_by_subset)
        
        print(f"\n{'='*50}")
        print("All evaluations finished!")
        print(f"{'='*50}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("python low_quality_ds_evaluation.py <dataset_path> [subset]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    subset = sys.argv[2] if len(sys.argv) > 2 else None
    
    if subset and subset not in ["baseline", "pdf_only", "mixed"]:
        print("Error: subset must be baseline, pdf_only, or mixed")
        sys.exit(1)
    
    try:
        evaluator = UniversalEvaluator(dataset_path)
        evaluator.run_evaluation([subset] if subset else None)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
