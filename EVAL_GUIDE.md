# OHR-Bench × Azure Document Intelligence — 操作指南

只做调研,不上线。两台机器:**个人机**(能上 HF)建子集,**公司机**跑 Azure DI 并打分。

---

## 0. 先搞清楚什么在哪(这一步决定了后面简单多少)

三个不同的位置,别混:

| 位置 | 有什么 | 公司机能不能直接拿 |
|---|---|---|
| **github.com/opendatalab/OHR-Bench** | `qas_v2.json`、**全量人工核验 gt**(9 个域)、**MinerU 解析输出**、评测框架代码。共 103 MB | **能,`git clone` 就行** |
| **huggingface.co/datasets/opendatalab/OHR-Bench** | `pdfs.zip` —— PDF 原件 | 不能 |
| **你自己的 private repo** | 中转 PDF 子集用 | — |

**结论:只有 PDF 需要中转。** gt / QA / MinerU 基线公司机直接 clone。

公司机先验证下载链路(Release 会从 github.com 302 跳到下面第二个域,很多代理只放行第一个):

```bash
curl -sI  https://github.com                    | head -1
curl -sIL https://objects.githubusercontent.com | head -1
```

第二条不通 → 改用分卷 commit(见 S6),全程只依赖 github.com。

---

## Phase A — 个人机,交给 CC

**S1. 骨架**

```bash
mkdir ohr-bench-subset && cd ohr-bench-subset && git init
# 放入 prepare_ohr_subset.py / run_azure_di.py / score_ocr.py / EVAL_GUIDE.md / README.md
printf 'pdfs/\nhf_cache/\n*.zip\n__pycache__/\nout/\n' > .gitignore
git clone --depth 1 https://github.com/opendatalab/OHR-Bench.git hf_cache
```

clone 完 `hf_cache/` 的目录结构正好就是脚本期望的布局,**建子集这一步零 HF 访问**。

**S2. 建子集(不下 PDF)** ← 人工闸门

```bash
python prepare_ohr_subset.py --budget 250 --max-doc-pages 15 --longform-share 0.6 --skip-pdfs
```

我在真实数据上验证过这组参数,应该得到约:

```
TOTAL: 90 windows / 214 pages / 713 QAs (250 page budget, 86% used)
4 of them are page windows out of a larger document
evidence mix: {table:210, chart:163, formula:132, text:117, reading_order:86, multi:5}
numeric answers: 342
MinerU baseline carried along for 90/90 docs
```

**验收四条**:总页数 200–220;六类 evidence 都非零;`is_window` 的文档 ≥3(必须有真实 10-K/10-Q,不能全是小表单);numeric ≥ 300。

任一条不满足就调参重跑。finance 不够就加 `--finance-share 0.5`;长文档不够就降 `--max-doc-pages`。

**S3. 下 PDF**

```bash
python prepare_ohr_subset.py --budget 250 --max-doc-pages 15 --longform-share 0.6
```

这一步才需要 HF,只下 `pdfs.zip`。扰动数据集(`formatting_noise_*` / `semantic_noise_*`)不下——那是噪声传导研究用的,跟评测 Azure 无关。

验收:没有 `WARN: no PDF found in pdfs.zip for ...`。

**S4. commit + 发 PDF**

```bash
cp -r out/manifest.json out/data . && git add -A && git commit -m "OHR-Bench subset (research use only)"
git ls-files | grep -c '\.pdf$'          # 必须是 0
gh release create v1 out/ohr-subset-pdfs.zip -n "research use only"
```

仓库设 private。文本部分约 1.3 MB。

---

## Phase B — 公司机

```bash
git clone <private-repo> ohr && cd ohr && unzip ohr-subset-pdfs.zip
pip install azure-ai-documentintelligence rapidfuzz
export AZURE_DI_ENDPOINT="https://<resource>.cognitiveservices.azure.com/"
export AZURE_DI_KEY="..."

python run_azure_di.py --root . --dry-run                 # 确认计费页数 ≈214
python run_azure_di.py --root . --limit 3 --domain finance  # 冒烟
```

**冒烟必须确认两件事**,缺一整个评测作废:

1. 输出是 markdown,表格是 markdown 表格。不是的话 `output_content_format` 没生效,编辑距离测的是格式不是准确率。
2. 没有 `!! page mismatch` 告警。

顺手把三个 key 的事解决:换 `AZURE_DI_KEY` 各跑同样 3 篇,`md5sum` 比对,byte-identical 就是同一个服务。

```bash
python run_azure_di.py --root . --workers 3   # 全量,中断可续跑,不重复计费
```

**关于页窗口**:manifest 里 `is_window: true` 的文档只提交 `page_start..page_end`(脚本自动转成 Azure 的 1-based `pages` 参数),只计费窗口内的页,输出的 `page_idx` 会偏移回原文档编号,和 gt 对齐。这是为什么 382 页的 JPMORGAN 10-K 能进子集而不炸预算。

---

## Phase C — 打分

```bash
python score_ocr.py --root . --pred data/retrieval_base/gt      # 自检,E.D. 必须 0.000
python score_ocr.py --root . --pred data/retrieval_base/MinerU  # 免费对照组
python score_ocr.py --root . --pred data/retrieval_base/azure --dump misses.json
```

### 三个指标

**1. 归一化编辑距离**,越低越好,跟官方结果表的 E.D. 列同类。我用 repo 自带的 MinerU 输出在这个子集上实测 0.233,官方全集报 0.24 —— 子集代表性没问题。

**2. Evidence recall**:标注的证据片段有没有在解析结果里活下来,按 evidence_source 分类。多片段证据要求**全部**存活(半张表照样答错)。

**3. Numeric answer recall**:`answer_form == Numeric` 的答案数值有没有留存。数值按值比较,`1,234.00` 命中 `1234`,但 `1234O`、`51234`、`1234.5` 都不算。

### 必须看"vs ceiling"那一列,不要看 raw

指标 2 和 3 有**数据集固有天花板**:约 17% 的 `evidence_context` 和 33% 的数字答案在 gt 里根本不逐字出现(标注是从另一种渲染写的,图表数值是人眼读的)。脚本会自动拿 gt 当预测跑一遍算出天花板,输出 `raw / ceiling / vs ceiling` 三列。**只有 vs ceiling 可读**,raw 绝对值没有意义。

自检那条(gt 对 gt)E.D. 必须是 0.000;它的 recall 不是 100%,那正是天花板本身。

### 为什么必须看第 3 项

MinerU 实测:mean E.D. 0.233 看着还行,但 chart 证据留存只有天花板的 15.1%、formula 48.3%、table 56.5%。**编辑距离会把结构性错误稀释掉。**几千字符里错几个数字,E.D. 几乎不动,但那一页对你已经废了。

### 人工核对 ← 不能省

`misses.json` 按 `evidence_source` 分组,每组抽 3 条回原 PDF 肉眼看,区分真错 vs 我的匹配规则太严。假阴性超过 30% 就把样例发我调规则。**没做这步之前,recall 数字不要往外说。**

---

## 对标与边界

官方已跑过 Azure DI(2025-06-30),**只能引用 retrieval 侧**:TXT 78.0 / TAB 59.4 / FOR 55.2 / CHA 45.2 / RO 5.8 / ALL 60.6。generation 和 overall 两组别用——官方注明那批用了 Llama3.1-8B 当 generator,正在重测。

那是全量 8,561 页的结果,你跑 214 页子集,**绝对值不可直接比,看相对形状**。

这套评测**回答不了**:签名(OHR-Bench 无此标注)、你们自己单据的版面特征、chunking/embedding/retrieval 下游。它的正确用途是**帮你决定内部标注集该标什么**——先看 Azure 在哪类元素掉点(我押 table 和 chart),再把人工标注预算集中砸过去。

## 数据集实况(和论文说法有出入,已实测)

- 论文说 7 个域,实际 gt 里有 9 个:`notes`(288 文档)和 `paper`(10 文档)**QA 数为 0**,脚本会自动跳过
- `evidence_source` 实际值是 `reading_order`(下划线)和多一个 `multi`,README 示例里写的是空格,别照抄
- `evidence_context` 43% 是 list,`evidence_page_no` 9% 是 list
- finance 双峰:10 个 10-K/10-Q(83–382 页)+ 55 个几页的扫描表单。不做长文档配额的话,密度选择会 100% 选中小表单
