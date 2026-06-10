# Sora Technical Report（OpenAI）

Sora 没有公开发表的 arXiv 论文。技术说明发布在 OpenAI 官方页面：

- 官方技术说明 URL：https://openai.com/index/video-generation-models-as-world-simulators/
- "Turning visual data into patches" 部分定义了 spacetime patch
- "Video compression network" 部分定义了 3D VAE
- "Variable duration, resolution, aspect ratio" 部分定义了 token 化的灵活性

请阅读此网页并参考以下第三方分析：
- [Open-Sora](https://github.com/hpcaitech/Open-Sora) 开源复现
- [Sora 架构分析博文](https://jamesg.blog/sora/)

如需本地存档，请使用 `webfetch` 工具或 `curl https://openai.com/index/video-generation-models-as-world-simulators/ -o 04_sora_tech_report.html`。
