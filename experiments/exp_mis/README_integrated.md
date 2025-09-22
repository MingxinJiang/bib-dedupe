# 整合评估脚本使用说明

## 概述

我们创建了两个整合的评估脚本，可以将整个数据处理流程（从原始bib文件到最终评估）合并到一个脚本中，大大简化了操作流程。

## 脚本说明

### 1. `integrated_evaluation.py` - 针对MIS Quarterly数据集的整合脚本

这个脚本专门为MIS Quarterly数据集设计，将以下步骤合并：
- bib2csv.py - 将bib文件转换为CSV格式
- filter_records.py - 根据来源过滤记录
- gen_merged_record_ids.py - 生成合并记录ID
- gen_records_pre_merged.py - 生成预处理记录
- evaluation_pdf.py - 执行去重评估

**使用方法：**
```bash
# 处理所有子集
python integrated_evaluation.py

# 只处理特定子集
python integrated_evaluation.py baseline
python integrated_evaluation.py pdf_only
python integrated_evaluation.py mixed
```

### 2. `universal_evaluation.py` - 通用评估脚本

这个脚本可以处理任何数据集，只需要指定数据集路径即可。

**使用方法：**
```bash
# 处理指定数据集的所有子集
python universal_evaluation.py mis-quarterly

# 只处理特定子集
python universal_evaluation.py mis-quarterly pdf_only
python universal_evaluation.py mis-quarterly baseline
python universal_evaluation.py mis-quarterly mixed
```

## 优势

1. **简化流程**：从原来的5个独立脚本简化为1个脚本
2. **减少错误**：避免手动运行多个脚本时可能出现的错误
3. **提高效率**：一次性完成整个评估流程
4. **易于扩展**：通用脚本可以轻松应用到其他数据集

## 输出文件

脚本会自动生成以下文件：
- `baseline.csv`, `pdf_only.csv`, `mixed.csv` - CSV格式的数据集
- `baseline/`, `pdf_only/`, `mixed/` 目录下的：
  - `records.bib` - 过滤后的记录
  - `merged_record_ids.csv` - 合并记录ID
  - `records_pre_merged.csv` - 预处理记录
  - `false_positives.csv` - 假阳性结果
  - `true_positives.csv` - 真阳性结果

## 注意事项

1. 确保数据集目录结构正确：
   ```
   dataset_name/
   └── data/
       ├── records.bib
       └── search/
           ├── CROSSREF.bib
           ├── DBLP.bib
           └── pdfs.bib
   ```

2. 如果某些文件不存在，脚本会给出警告并跳过相应步骤

3. 通用脚本会自动检测可用的bib文件并创建相应的子集

## 示例

### 处理MIS Quarterly数据集
```bash
cd experiments/exp_mis
python universal_evaluation.py mis-quarterly
```

### 只处理PDF数据
```bash
python universal_evaluation.py mis-quarterly pdf_only
```

这样，您就可以用一个命令完成整个评估流程，大大简化了操作！ 