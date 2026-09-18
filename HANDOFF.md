# HANDOFF — OHR-Bench 子集交接给公司机

个人机建好的评测子集，交给公司机跑 Azure DI 并打分。这是两台机器之间唯一的接口文档。

> ⚠️ **本仓库 `data/` 是裁剪过的子集，与 opendatalab/OHR-Bench 官方仓库的全量 `data/` 不是一回事。**
> **不要混用，也不要用官方全量覆盖它** —— gt 已按页窗口裁剪，与 `manifest.json` 的
> `page_start` / `page_end` 严格对应。覆盖后 90 个文档里有 4 个的页号会对不上，
> 打分结果全部作废，而且不会报错，只会悄悄给出错误的数字。

> ⚠️ **License：OHR-Bench 是 research use only，不得商用。** 本次只做内部调研选型，
> 不上线、不进产品。对外引用数字前先过内部法务。

---

## 1. 怎么拿到文件

**交付方式：U 盘 / 内网共享盘等公司批准的传输渠道，不走 GitHub 下载。**

原因：备份仓库是 private（数据集 research-only，不能公开），private 仓库的浏览器下载
同样需要在公司机上登录 GitHub 账号。这一条不做假设，所以走物理/内网传输。

需要搬的就**一个文件**：

| 文件 | 字节数 | SHA-256 |
|---|---|---|
| `ohr-handoff-v1.zip` | `67128516` (64.0 MiB) | `fbb790dcc770f6dec3660a510681b9d8e944019c5567262b674a30f7a725ef01` |

个人机上的位置：`ohr-bench-subset/out/ohr-handoff-v1.zip`

这个 zip 是自包含的，解压即得下面第 2 节的完整目录，公司机不需要再从任何地方下载东西。
里面含 302 个条目：90 个 PDF、90 个 gt、90 个 MinerU 基线、QA、manifest、两个脚本、操作指南。

### 备份（仅在公司机浏览器能登录 GitHub 时可用）

private 仓库：https://github.com/gejun2008/ohr-bench-subset

仓库 zip（只有文本，**不含 PDF**）：
`https://github.com/gejun2008/ohr-bench-subset/archive/refs/heads/main.zip`

- 未登录访问返回 404，这是正常的
- **没有创建 Release**，PDF 不在 GitHub 上，只在上面那个交付 zip 里
- GitHub 动态生成的 archive zip **每次字节数和 sha256 都可能变**，不要拿它做完整性校验，
  以 `ohr-handoff-v1.zip` 的 sha256 为准

---

## 2. 解压后应有的目录树

```
ohr/
├── manifest.json
├── data/qas_subset.json
├── data/retrieval_base/gt/<domain>/*.json
├── data/retrieval_base/MinerU/<domain>/*.json
├── pdfs/<domain>/*.pdf
├── run_azure_di.py
├── score_ocr.py
└── EVAL_GUIDE.md
```

`<domain>` 实际有 7 个：`academic` `administration` `finance` `law` `manual` `news` `textbook`。
（官方 gt 里另有 `notes` 和 `paper`，这两个域 QA 数为 0，无法打分，已排除。）

解压命令：

```bash
unzip ohr-handoff-v1.zip && cd ohr
```

---

## 3. 自检（解压后在 `ohr/` 目录里跑）

```bash
python -c "
import json,glob
m=json.load(open('manifest.json'))
print('manifest windows:', m['totals']['windows'], 'pages:', m['totals']['pages'])
print('gt files:', len(glob.glob('data/retrieval_base/gt/*/*.json')))
print('pdf files:', len(glob.glob('pdfs/*/*.pdf')))
"
```

**必须输出**（已在个人机上解压验证过）：

```
manifest windows: 90 pages: 214
gt files: 90
pdf files: 90
```

任何一行对不上就是传输出了问题，重新校验 sha256，不要继续往下跑。

传输前后各算一次校验和：

```bash
shasum -a 256 ohr-handoff-v1.zip      # macOS / Linux
certutil -hashfile ohr-handoff-v1.zip SHA256   # Windows
```

---

## 4. 子集内容（用来判断结果有没有代表性）

```
TOTAL: 90 windows / 214 pages / 713 QAs (250 page budget, 86% used)
4 of them are page windows out of a larger document
evidence mix: {table: 210, chart: 163, formula: 132, text: 117, reading_order: 86, multi: 5}
numeric answers: 342
```

其中 4 个是从长文档里截出来的页窗口，全部是真实 10-K（页号 0-based）：

| 文档 | 总页数 | 窗口 | QA 数 |
|---|---|---|---|
| finance/AES_2022_10K | 257 | 83–97 | 57 |
| finance/AMAZON_2017_10K | 85 | 50–64 | 24 |
| finance/AMAZON_2019_10K | 83 | 50–64 | 27 |
| finance/VERIZON_2021_10K | 120 | 15–29 | 46 |

**这 4 个必须只提交窗口内的页。** `run_azure_di.py` 读 manifest 的 `is_window` 自动处理，
转成 Azure 的 1-based `pages` 参数；输出的 `page_idx` 会偏移回原文档编号，和 gt 对齐。
不做窗口的话光 AES 一篇就是 257 页，直接把计费预算炸掉。

**已知缺陷：** `multi`（跨多页证据）只有 5 条。建子集的脚本在
`prepare_ohr_subset.py:192` 处对 list 型的 `evidence_page_no` 会 `int()` 失败并跳过，
这类 QA 大部分没进子集。不影响其余五类，但 `multi` 那一行的数字样本太少，不要单独解读。

---

## 5. 公司机要做的事

完整步骤见 `EVAL_GUIDE.md` 的 Phase B 和 Phase C，这里只列顺序和卡点。

### 5.1 改 API adapter（这块是公司机这边做）

公司环境不直连 Azure SDK，走内部封装的对外 API，要替换 `run_azure_di.py` 里的
`analyze()` 函数。三个约束不能破：

1. **必须 layout 类模型 + markdown 输出**，表格要是 markdown 表格。不是 markdown 的话
   编辑距离测的是格式不是准确率，整个评测作废。
2. **必须支持传页范围**，否则第 4 节那 4 个 10-K 会整篇提交。
3. **返回要能切成** `[{"page_idx": N, "text": "..."}]`，且 `page_idx` 偏移回原文档编号。

### 5.2 跑 OCR

```bash
pip install azure-ai-documentintelligence rapidfuzz
python run_azure_di.py --root . --dry-run                    # 确认页数 = 214
python run_azure_di.py --root . --limit 3 --domain finance   # 冒烟
```

**冒烟必须确认两件事，缺一整个评测作废：**

1. 输出是 markdown，表格是 markdown 表格
2. 没有 `!! page mismatch` 告警

```bash
python run_azure_di.py --root . --workers 3   # 全量，中断可续跑，不重复计费
```

计费规模：**214 页**。跑完看 `data/retrieval_base/azure/_run_log.json` 里的
`pages_billed_estimate` 核对。

### 5.3 打分

```bash
python score_ocr.py --root . --pred data/retrieval_base/gt      # 自检，E.D. 必须 0.000
python score_ocr.py --root . --pred data/retrieval_base/MinerU  # 免费对照组
python score_ocr.py --root . --pred data/retrieval_base/azure --dump misses.json
```

- 自检那条 E.D. 不是 0.000 就是文件错位了，先查这个，别看后面的数
- **只看 `vs ceiling` 那列。** 约 17% 的 evidence 和 33% 的数字答案在 gt 里就不逐字出现，
  raw 绝对值没有意义
- MinerU 是免费对照组，个人机实测 mean E.D. 0.233（官方全集报 0.24，子集代表性 OK）
- `misses.json` 每类 evidence 抽 3 条回原 PDF 肉眼核对。假阴性超过 30% 说明匹配规则太严。
  **这步没做之前 recall 数字不要往外说。**

---

## 6. 对标与边界

官方跑过 Azure DI（2025-06-30），**只能引用 retrieval 侧**：
TXT 78.0 / TAB 59.4 / FOR 55.2 / CHA 45.2 / RO 5.8 / ALL 60.6。
generation 和 overall 别用 —— 官方注明那批用 Llama3.1-8B 当 generator，正在重测。

那是全量 8,561 页的结果，这里是 214 页子集，**绝对值不可直接比，看相对形状。**

这套评测**回答不了**：签名识别（OHR-Bench 无此标注）、你们自己单据的版面特征、
chunking / embedding / retrieval 下游效果。

正确用途是**帮你决定内部标注集该标什么** —— 先看 Azure 在哪类元素掉点，
再把人工标注预算集中砸过去。

---

## 7. 复现这个子集（个人机，一般用不到）

```bash
git clone --depth 1 https://github.com/opendatalab/OHR-Bench.git hf_cache
python prepare_ohr_subset.py --budget 250 --max-doc-pages 15 --longform-share 0.6 --skip-pdfs
```

参数相同结果确定可复现。注意 `manifest.json` 的 `selection` 段**没有记录 `longform_share`**，
只记了 `max_doc_pages` 和 `finance_share`，光看 manifest 复现不出来，以本节命令为准。

加 PDF 要去掉 `--skip-pdfs`，那一步需要访问 Hugging Face 下 `pdfs.zip`（1.5 GB）。
