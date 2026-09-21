# NOTICE — 数据出处与授权

本仓库包含 [OHR-Bench](https://github.com/opendatalab/OHR-Bench)（OpenDataLab）的一个
**子集**，用于内部技术调研（OCR / 文档解析引擎选型评测）。

## 授权

**OHR-Bench 是 research use only，不得用于商业用途。** 本仓库内的所有数据沿用原始授权，
不附加任何额外许可。

原始数据来源：

- 标注与 ground truth：https://github.com/opendatalab/OHR-Bench
- 源 PDF：https://huggingface.co/datasets/opendatalab/OHR-Bench (`pdfs.zip`)
- 论文：https://arxiv.org/abs/2412.02592

原始 README 声明：PDF 收集自公开渠道与社区贡献，不允许分发的内容已被移除；数据集仅供
研究使用，不得商用。版权问题请联系 OpenDataLab@pjlab.org.cn。

## 本仓库包含什么

- 90 个文档 / 214 页的确定性子集（选择参数见 `manifest.json` 与 `HANDOFF.md` 第 7 节）
- 按页窗口裁剪的 gt、对应的 713 条 QA、MinerU 解析基线
- 评测脚本，以及自包含的 `ohr-handoff-v1.zip`

**不包含**任何 HSBC 或其他机构的内部文档、客户数据或专有信息。全部内容均来自上述公开
研究数据集。

## 如果你是版权方

如需移除本仓库或其中任何文档，请通过 GitHub issue 联系，会立即处理。
