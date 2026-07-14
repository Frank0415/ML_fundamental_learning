# Tianwei Zhang 论文准备材料

入口：

- [index.html](index.html): 16 篇非安全系统论文的中文导读，范围是 inference、serving、decoding、scheduling、GPU cluster、LLM systems 和 edge-cloud execution。
- [publication_inference_marked.html](publication_inference_marked.html): 279 条官网记录加 42 条 arXiv/Scholar 补充。绿色是非安全 inference/系统，黄色是 inference security，红色是不收。每条都有来源、arXiv 和判断理由。
- [publication_original.html](publication_original.html): 2026-07-14 下载的官网原始页面。
- [arxiv_match_report.json](arxiv_match_report.json) / [arxiv_exact_query_report.json](arxiv_exact_query_report.json): 2026-07-14 的 arXiv 匹配和 Scholar 交叉检查记录。
- [quality_marker_report.json](quality_marker_report.json): Oral/高引标签清单。高引数据改用 Tianwei Zhang 的 Google Scholar 个人主页，阈值为引用数不低于 50。

本地文件：

- `papers/`: 16 篇非安全核心论文 PDF。
- `extracted_text/`: 16 份用 `mutool draw -F txt` 从 PDF 抽出的文本。

筛选口径：

- 收：推理服务、解码、KV/cache、batching/scheduling、GPU 集群、跨集群训练、长序列训练执行、边云协同推理、inference-time steering、非安全输入侧执行优化。
- 不收进核心导读：安全、隐私、攻击、防御、水印、泄漏、red-teaming、secure/private/verifiable inference。
