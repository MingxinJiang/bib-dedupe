#!/usr/bin/env python3
"""
整合的评估脚本 - 从原始bib文件到最终评估的完整流程

这个脚本将以下步骤合并：
1. 将bib文件转换为CSV格式
2. 根据来源过滤记录
3. 生成合并记录ID
4. 生成预处理记录
5. 执行去重评估

使用方法：
python integrated_evaluation.py [subset]
其中subset可以是: baseline, pdf_only, mixed (默认: 全部)
"""

import warnings
import csv
import re
import sys
from pathlib import Path
from datetime import datetime
import itertools

import pandas as pd
import bibtexparser
from colrev.loader import load_utils
from bib_dedupe.bib_dedupe import prep, block, match, cluster, merge
from bib_dedupe.dedupe_benchmark import DedupeBenchmarker
from bib_dedupe.constants.fields import ID as FIELD_ID, AUTHOR, TITLE, DOI

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

BASE = Path(__file__).parent
DATA_DIR = BASE / "mis-quarterly" / "data"
SEARCH_DIR = DATA_DIR / "search"

# 源文件映射
SRC_MAP = {
    "CROSSREF.bib": ("C", "crossref"),
    "DBLP.bib": ("D", "dblp"), 
    "pdfs.bib": ("P", "pdf"),
}

PREFIX = {
    "CROSSREF.bib": "C",
    "DBLP.bib": "D",
    "pdfs.bib": "P",
}

def bib_to_df(rel_path: str) -> pd.DataFrame:
    """将bib文件转换为DataFrame"""
    bib_path = BASE / rel_path
    if not bib_path.is_file():
        raise FileNotFoundError(f"{bib_path} not found")
    records = load_utils.load(filename=bib_path, unique_id_field="ID")
    df = (pd.DataFrame.from_dict(records, orient="index")
          .reset_index()
          .rename(columns={"index": "ID"}))
    df = df.reset_index(drop=True)
    
    # 去除重复列（只保留第一个）
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def create_csv_datasets():
    """创建CSV数据集"""
    print("=== 创建CSV数据集 ===")
    
    # 读取bib文件
    df_cr = bib_to_df("mis-quarterly/data/search/CROSSREF.bib")
    df_db = bib_to_df("mis-quarterly/data/search/DBLP.bib")
    df_pdf = bib_to_df("mis-quarterly/data/search/pdfs.bib")

    # 创建baseline数据集 (CROSSREF + DBLP)
    df_baseline = pd.concat([df_cr, df_db], ignore_index=True)
    df_baseline.to_csv(BASE / "baseline.csv", index=False)
    print(f"创建 baseline.csv: {len(df_baseline)} 条记录")

    # 创建pdf_only数据集
    df_pdf.to_csv(BASE / "pdf_only.csv", index=False)
    print(f"创建 pdf_only.csv: {len(df_pdf)} 条记录")

    # 创建mixed数据集 (baseline + pdf)
    df_mixed = pd.concat([df_baseline, df_pdf], ignore_index=True)
    df_mixed.to_csv(BASE / "mixed.csv", index=False)
    print(f"创建 mixed.csv: {len(df_mixed)} 条记录")

def filter_records():
    """根据来源过滤记录"""
    print("=== 过滤记录 ===")
    
    # 读取原始记录
    src_path = DATA_DIR / "records.bib"
    with src_path.open(encoding="utf-8") as f:
        db = bibtexparser.load(f)

    def write_subset(name, keep_fn):
        out = bibtexparser.bibdatabase.BibDatabase()
        out.entries = [e for e in db.entries if keep_fn(e.get("colrev_origin", ""))]
        path = BASE / Path(name) / "records.bib"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            bibtexparser.dump(out, f)
        print(f"创建 {name}/records.bib: {len(out.entries)} 条记录")

    # 创建各个子集
    write_subset("baseline",
                 lambda origin: "CROSSREF.bib" in origin or "DBLP.bib" in origin)
    write_subset("pdf_only",
                 lambda origin: "pdfs.bib" in origin)
    write_subset("mixed",
                 lambda origin: True)

def generate_merged_record_ids(subset):
    """生成合并记录ID"""
    print(f"=== 为 {subset} 生成合并记录ID ===")
    
    subset_dir = BASE / subset
    rec_path = subset_dir / "records.bib"
    out_path = subset_dir / "merged_record_ids.csv"

    with rec_path.open(encoding="utf-8") as bibfile:
        db = bibtexparser.load(bibfile)

    pattern = re.compile(r"([^;{}\s]+\.bib)/([^;{}\s]+)")
    clusters = {}  # map: 原簇 key -> set of prefixed IDs

    # 确定当前子集允许的前缀
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
            prefix = PREFIX.get(fname)
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

    print(f"创建 {out_path}: {len(clusters)} 个簇")

def generate_records_pre_merged(subset):
    """生成预处理记录"""
    print(f"=== 为 {subset} 生成预处理记录 ===")
    
    subset_dir = BASE / subset
    out_path = subset_dir / "records_pre_merged.csv"

    # 确定要处理的bib文件
    if subset == "baseline":
        bib_files = ["CROSSREF.bib", "DBLP.bib"]
    elif subset == "mixed":
        bib_files = ["CROSSREF.bib", "DBLP.bib", "pdfs.bib"]
    elif subset == "pdf_only":
        bib_files = ["pdfs.bib"]
    else:
        raise ValueError("Unknown subset")

    all_dfs = []
    for bib_file in bib_files:
        path = SEARCH_DIR / bib_file
        prefix, src = SRC_MAP[bib_file]
        
        with path.open(encoding="utf-8") as fh:
            bib = bibtexparser.load(fh)
        
        df = pd.DataFrame(bib.entries)
        if df.empty:
            continue
            
        df["orig_ID"] = df["ID"]
        df["ID"] = df["orig_ID"].apply(lambda rid: f"{prefix}_{rid}")
        df["orig_source"] = src
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    cols = ["ID", "orig_ID", "orig_source"] + [c for c in full.columns if c not in ("ID", "orig_ID", "orig_source")]
    full = full[cols]

    full.to_csv(out_path, index=False)
    print(f"创建 {out_path}: {len(full)} 条记录")

def evaluate_subset(subset):
    """评估指定子集"""
    print(f"\n=== 评估 {subset} ===")
    subset_dir = BASE / subset

    try:
        bench = DedupeBenchmarker(benchmark_path=subset_dir)
        records_df = bench.get_records_for_dedupe()
        print(f"{datetime.now()} 加载了 {len(records_df)} 条记录")
    except Exception as e:
        print(f"错误: 无法加载记录: {e}")
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
        print(f"错误: 去重过程中出错: {e}")
        return

    try:
        result = bench.compare_dedupe_id(
            records_df=records_df,
            merged_df=merged_df,
            timestamp=timestamp
        )
    except Exception as e:
        print(f"错误: 比较结果时出错: {e}")
        return

    print(f"\n{subset} 的结果:")
    print(f"  TP  = {result['TP']}")
    print(f"  FP  = {result['FP']}")
    print(f"  FN  = {result['FN']}")
    print(f"  TN  = {result.get('TN', 'n/a')}")
    print(f"  精确率            = {result['precision']:.4f}")
    print(f"  召回率 (敏感度)   = {result['sensitivity']:.4f}")
    print(f"  特异度           = {result['specificity']:.4f}")
    print(f"  F1 分数          = {result['f1']:.4f}")

    # 生成假阳性和真阳性对
    pred_pairs = set(itertools.chain.from_iterable(
        itertools.combinations(cluster, 2) for cluster in dup_sets
    ))
    true_pairs = set(itertools.chain.from_iterable(
        itertools.combinations(tc, 2) for tc in bench.true_merged_ids
    ))
    fp_pairs = pred_pairs - true_pairs
    tp_pairs = pred_pairs & true_pairs
    
    print(f"{datetime.now()} 发现 {len(fp_pairs)} 个假阳性对，正在导出...")

    df_idx = records_df.set_index(FIELD_ID)
    
    # 导出假阳性
    if fp_pairs:
        # 将pairs转换为clusters
        fp_clusters = {}
        for pair in fp_pairs:
            id1, id2 = sorted(pair)
            # 查找是否已经存在包含这些ID的cluster
            found_cluster = None
            for cluster_id, cluster_ids in fp_clusters.items():
                if id1 in cluster_ids or id2 in cluster_ids:
                    found_cluster = cluster_id
                    break
            
            if found_cluster is not None:
                fp_clusters[found_cluster].update([id1, id2])
            else:
                new_cluster_id = len(fp_clusters)
                fp_clusters[new_cluster_id] = {id1, id2}
        
                    # 按组输出
            rows = []
            for cluster_id, cluster_ids in fp_clusters.items():
                for i, record_id in enumerate(sorted(cluster_ids)):
                    record = df_idx.loc[record_id]
                    rows.append({
                        "cluster_id": cluster_id,
                        "id": record_id,
                        "title": record.get(TITLE, ""),
                        "author": record.get(AUTHOR, ""),
                        "doi": record.get(DOI, ""),
                    })
        
        fp_df = pd.DataFrame(rows)
        out_fp = subset_dir / "false_positives.csv"
        fp_df.to_csv(out_fp, index=False, encoding="utf-8")
        print(f"{datetime.now()} 假阳性已保存到 {out_fp}")

    # 导出真阳性
    if tp_pairs:
        # 将pairs转换为clusters
        tp_clusters = {}
        for pair in tp_pairs:
            id1, id2 = sorted(pair)
            # 查找是否已经存在包含这些ID的cluster
            found_cluster = None
            for cluster_id, cluster_ids in tp_clusters.items():
                if id1 in cluster_ids or id2 in cluster_ids:
                    found_cluster = cluster_id
                    break
            
            if found_cluster is not None:
                tp_clusters[found_cluster].update([id1, id2])
            else:
                new_cluster_id = len(tp_clusters)
                tp_clusters[new_cluster_id] = {id1, id2}
        
                    # 按组输出
            rows_tp = []
            for cluster_id, cluster_ids in tp_clusters.items():
                for i, record_id in enumerate(sorted(cluster_ids)):
                    record = df_idx.loc[record_id]
                    rows_tp.append({
                        "cluster_id": cluster_id,
                        "id": record_id,
                        "title": record.get(TITLE, ""),
                        "author": record.get(AUTHOR, ""),
                        "doi": record.get(DOI, ""),
                    })
        
        tp_df = pd.DataFrame(rows_tp)
        out_tp = subset_dir / "true_positives.csv"
        tp_df.to_csv(out_tp, index=False, encoding="utf-8")
        print(f"{datetime.now()} 真阳性已保存到 {out_tp}")

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        subset = sys.argv[1]
        if subset not in ["baseline", "pdf_only", "mixed"]:
            print("错误: subset 必须是 baseline, pdf_only, 或 mixed")
            sys.exit(1)
        subsets = [subset]
    else:
        subsets = ["pdf_only", "baseline", "mixed"]

    print("开始整合评估流程...")
    print(f"将处理以下子集: {', '.join(subsets)}")
    
    # 步骤1: 创建CSV数据集
    create_csv_datasets()
    
    # 步骤2: 过滤记录
    filter_records()
    
    # 步骤3-5: 为每个子集生成必要文件并评估
    for subset in subsets:
        print(f"\n{'='*50}")
        print(f"处理子集: {subset}")
        print(f"{'='*50}")
        
        # 步骤3: 生成合并记录ID
        generate_merged_record_ids(subset)
        
        # 步骤4: 生成预处理记录
        generate_records_pre_merged(subset)
        
        # 步骤5: 评估
        evaluate_subset(subset)
    
    print(f"\n{'='*50}")
    print("所有评估完成！")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 