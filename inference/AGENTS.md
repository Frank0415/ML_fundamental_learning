# Inference 资料站扩展指南

这份文件给后续维护者和 Agent 使用。内容规范源位于 `/Users/franksair/Documents/learning_ML/inference`，发布仓库 `/Users/franksair/Documents/Research/papers` 是导出结果。先改规范源，再导出、构建和审查；不要把发布副本当成长期编辑源。

## 完成标准

一次扩展只有同时满足以下条件才算完成：

- 新内容有明确入口，用户能从首页、主线或参考资料目录到达。
- 每个需要“看完 HTML 就不用读原文”的对象都有独立完整解释，不以摘要或导读代替论文内容。
- 事实、版本、数字、图和结论能回到论文、模型卡、官方仓库或官方技术报告；推断与未公开信息明确标出。
- 原有页面、链接、标题、图像和资料不因导出丢失，新目录已进入显式复制与校验范围。
- 规范源校验、Quarto 构建、静态链接校验，以及桌面/移动端的浅色和深色截图检查全部通过。
- 仍有证据缺口、来源冲突或未验证页面时，结果保持本地，不推送生产分支。

## 先判断变更类型

| 需求 | 默认做法 | 不应做的事 |
| --- | --- | --- |
| 在现有方向加入几篇论文 | 更新已有概念页；需要逐篇完整解释时，在所属资料集增加一篇一页的 HTML，并把入口并入参考资料目录 | 为了增加数量把每篇论文都塞成新的 Mainline 概念章节 |
| 加一整条新线路 | 在 `inference/<track-slug>/` 建独立资料集，再从站点首页和相关主线章节连接 | 把 PostTrain RL、求职信息等硬塞进现有“模型到数据中心”因果主线 |
| 加一套参考资料 | 为新来源增加结构化快照和 loader，由 `build_reference_catalog.py` 生成目录 | 手改生成后的 `mainline/references.html`，或把无来源的大段正文复制进目录 |

如果一个新增主题会改变阅读顺序、拥有至少数个相互依赖的章节，或需要独立的更新频率和证据规则，就把它当作新线路。只有一两篇论文补充现有机制时，留在现有方向。

## 仓库与生成物边界

- `inference/mainline/`：Inference 概念主线、主题页、总来源表、参考目录生成器和校验器。
- `inference/prepare_sg/`：现有待读论文存档和逐篇完整解释。它是一个有自身来源清单的资料集，不是所有新论文的默认垃圾桶。
- `inference/multimodal/`、`inference/diffusion/`：独立资料线的现有例子；它们说明独立线路可以保留自己的页面序列和资源，但新线路不必复制其旧版式。
- `inference/mainline/quarto_papers_site/`：首页、Quarto 配置、全站主题初始化和 Cloudflare 静态构建模板。
- `Research/papers/`：由 `mainline/scripts/export_to_research_papers_site.py` 生成的发布副本。除紧急发布修复外，不在这里单独维护内容。
- `_site/`、`mainline/references.html`、资料索引页：生成物。应修改它们的源数据或生成器，再重新生成。

导出器会重建发布仓库中的 `mainline/`、`prepare_sg/`、`multimodal/` 和 `diffusion/` 子树。只有 `Research` 克隆而没有 `learning_ML/inference` 作者源时，停止内容修改并先取得作者源；不要从发布镜像反向拼出规范源。

已知状态（2026-07-16）：`learning_ML` 远端尚未跟踪 `inference/mainline/`，当前作者源只存在于本机；`Research/papers/AGENTS.md` 只是这份指南的可见副本，不会让发布镜像变成作者源。需要跨机器或远端 Agent 协作前，先把完整作者源作为独立、可审计的提交纳入版本控制；不要在普通内容任务中顺带发布整棵未跟踪目录。

导出器采用白名单复制。新增顶层资料线后，必须同时扩展 `export_to_research_papers_site.py` 的复制白名单、`classify_page()` 主题分类和 `inject_shared_theme()` 扫描根目录，扩展 `_quarto.yml` 的 `resources` 与 `validate-static-site.mjs` 的 `scopes`；少一个就不算完成。

## 证据与内容规则

1. 当前模型、API、产品、职位和基础设施状态在写作当天重新核验。记录版本或发布日期，不把旧页面记忆当作现状。
2. 技术结论优先使用论文、官方模型卡、官方代码仓、标准正文和官方技术报告。二手文章可以作为线索和目录来源，不能替代一手证据。
3. 论文指标必须同时写清任务、模型、硬件、基线和统计口径。论文里的倍数只属于该实验，不外推成普遍收益。
4. 架构图只保留能解释真实组件、数据路径或实验结果的一手图。每张图本地打包，带 `alt`、原始链接、图号和边界说明。没有合适图时记录例外，不生成空泛模板图充数。
5. 不转载受版权保护的第三方全文。公众号目录只保存标题、日期、原始链接、本站分类和本站独立解释；不要保存登录态、Cookie、个人信息或未授权正文。
6. 若只能获得摘要或不完整材料，把页面标为“证据受限”，缩小结论范围。不要用常识补齐论文未公开的方法、参数或实验。
7. `Find Jobs in LLM` 一类高时效线路应保存官方职位链接、地点、团队、发布日期或抓取日期和失效状态；不收集候选人数据，不把历史职位写成仍在招聘。

## 数学公式与 LaTeX 规则

1. 只要表达包含数学推导或运算结构，就必须写成 LaTeX，不能用 ASCII 伪公式代替。求和、积分、分式、矩阵、概率、期望、范数、上下标、上下限、渐近复杂度和多步等式都属于强数学公式。例如 MoE 聚合应写成 `\(y_t=\sum_{e\in\mathcal I_t}g_{t,e}\,\operatorname{Expert}_e(x_t)\)`，不能写成 `y = Σ[e in TopK(router(x))] gate_e(x) * Expert_e(x)`。
2. 行内公式使用 `\(...\)`；需要单独成行、包含推导或矩阵的公式使用 `\[...\]`。不要使用裸 `$...$` 或 `$$...$$`，避免与价格、代码和普通文本冲突。
3. 简单配置或名称不必数学化。例如 `3 × KDA + 1 × Gated MLA`、`16-of-896 experts`、`TP=8`、命令、JSON、状态机和伪代码可以保留普通文本或 `<code>/<pre>`。判断标准是读者是否需要按数学对象解析其运算关系，而不是文本中是否出现数字或 `×`。反之，`steps × patches` 若表示两个变量的实际乘积，就必须写成 `\(\mathrm{steps}\times\mathrm{patches}\)`。
4. 公式中的说明文字使用 `\text{...}`，多字符标识符使用 `\mathrm{...}` 或 `\operatorname{...}`，向量、矩阵、集合、条件概率和单位保持一致的语义记号。不要把中文段落整段塞入数学环境。
5. 公式前后必须解释变量、单位、适用条件和近似边界。论文中的经验公式或本站抽象必须标清来源/性质，不能把教学近似写成论文原式。
6. 原文存档中的 LaTeXML/MathML 属于上游文档，不批量反向改写；本站手写的 `guide.html`、Mainline、Multimodal 和 Diffusion 解释页必须遵守本规则。
7. 数学渲染使用仓库内打包的 MathJax 静态资源，只在页面含 TeX 定界符时由导出器注入。不得接入 CDN，也不得把公式预渲染成不可搜索的截图。
8. 修改公式后必须检查：TeX 定界符成对、页面只加载 1 份数学渲染器、`code/pre` 不被误渲染、桌面与移动端没有横向溢出，并在浅色/深色模式下截图验证至少 1 个行内公式、1 个独立公式和 1 个矩阵或多行公式。

## 路径 A：加入几篇相关论文

### 预期结果

新增论文应加强一个已有论点，或形成可独立阅读的完整论文页。用户从相关主题页和参考资料目录都能找到它，且不会出现只有题名没有解释的孤立条目。

### 执行规则

1. 先搜索 `mainline/topics/`、`mainline/references.html` 和各资料集 manifest，确认论文是否已有条目、是否只是缺少更新。
2. 论文只是某个机制的新证据时，更新对应概念页的机制、实验、局限和一手来源，不另建概念章节。
3. 用户需要逐篇完整解释时，一篇论文由一个 Agent 负责一个固定输出路径。页面至少讲清前置背景、问题、方法、公式或算法、端到端例子、实验设置与结果、局限、实现/复现和一手来源。
4. 属于现有 `prepare_sg` 选择集的论文，沿用它的存档、manifest、索引和 `guide.html` 约定，并运行 `prepare_sg/scripts/validate_complete_guides.py`。该校验当前要求至少 12 个二级章节和 7000 个可见正文字符（中文、英文技术名、公式变量与阿拉伯数字均计入），证据受限页只有显式登记后才能降低门槛。
5. `prepare_sg/scripts/archive_green_papers.py` 目前只认识 `publication_inference_marked.html` 中原有的 32 条绿色记录，并会重写 manifest 和索引。它不是任意论文的新增 API；扩大该选择集前先把归档器改成 manifest/input-driven，并为动态数量补测试，否则不要运行它覆盖手工新增记录。
6. 不属于现有选择集的论文，放入所属资料线的 `docs/papers/`，并在该资料线 manifest 中登记稳定 slug、标题、角色、一手来源和最后核验日期。不要伪造 `prepare_sg` 的旧编号。
7. 更新相关主题页、资料线索引、`mainline/sources.html` 和参考资料 loader。若新论文改变跨章节结论，再更新 Mainline 卡片描述或横向比较页。

### 并行方式

独立论文可以并行，但每个 worker 只拥有自己的论文来源、HTML 路径和资源目录。一个协调者负责共享 manifest、索引、参考目录和导航，避免多个 worker 同时改同一生成物。跨论文比较、共同结论和版本冲突由协调者在单篇完成后统一处理。

## 路径 B：加入一整条新线路

### 预期结果

新线路有独立入口、清楚的起止范围、按依赖排序的页面和自己的资料目录。例如 PostTrain RL 可以从目标与数据进入奖励、训练算法、rollout、系统和评测；Agentic Systems 可以从程序模型进入 runtime、状态、工具、调度和评测；求职线路则按岗位地图、能力证据、团队与职位快照组织。

若所有新增章节都服务于现有 Inference 受众，并能自然落在现有因果主线的一段中，优先在该 stage 下新增一个 subgroup 和若干 topic。若它有独立受众、阅读顺序、更新周期或证据规则，再建立顶层资料线。

### 最小资料集

默认放在 `inference/<track-slug>/`：

```text
<track-slug>/
  README.md                 范围、受众、更新规则和证据边界
  manifest.json             有序页面、语义角色、一手来源、最后核验日期
  docs/index.html           线路入口和阅读顺序
  docs/01_<slug>.html       独立完整章节
  docs/assets/              本地图片、数据和线路样式
  scripts/validate_<slug>.* 线路特有的结构与链接校验
```

manifest 的具体字段可以按线路调整，但必须能回答：有哪些页面、为什么按这个顺序、每页由什么一手来源支撑、何时核验、哪些页面证据受限。不要只列文件名。

### 集成点

1. 若线路属于 Mainline，新增 `mainline/topics/<NN_slug>.html`，并同步更新 `mainline/manifest.json`、`mainline/index.html` 的 stage/subgroup/card、`mainline/sources.html`、`mainline/visual-audit.json` 和相关页面的反向链接。Topic 必须恰好一个 `h1`、至少 10 个 `h2`、至少 2200 个可见字符，并链接共享样式、主线首页和来源页。
2. `generate_frontier_model_pages.py` 拥有 33–37，`generate_deployment_pages.py` 拥有 38–41，`generate_lingyun_catalog.py` 拥有 42。修改这些页面时改生成器并重新生成，不要只改生成后的 HTML。
3. 若线路独立，在 `mainline/quarto_papers_site/index.qmd` 增加顶层入口。只有用户频繁跨线路导航时才加入全站 navbar，普通资料线用首页卡片即可。
4. 在 `mainline/scripts/export_to_research_papers_site.py` 增加显式静态条目、复制调用、`classify_page()` 规则和 `inject_shared_theme()` 扫描根目录，不复制环境、模型权重、缓存或实验临时文件。
5. 在 `mainline/quarto_papers_site/_quarto.yml` 增加资源路径，在 `validate-static-site.mjs` 增加 HTML scope。构建脚本仍只产出纯静态文件。
6. 若新线路与 Inference 主线的某一阶段直接相关，在对应主题页加上下文链接；否则只从首页和参考资料进入，不改现有因果顺序。
7. 若线路包含论文或报告目录，为 `build_reference_catalog.py` 增加 loader，使它们进入统一参考资料页。
8. 更新本文件旁的 README 项目结构和线路说明。

### 线路拆分

先由一个协调者锁定范围、manifest 和跨页依赖，再按互不依赖的章节组并行。共享首页、CSS、manifest、编号和导航只由协调者修改。若执行中发现新的文档类型或依赖关系，只调整受影响的章节和 manifest，不重写已经成立的整条线路。

## 路径 C：加入一套参考资料

### 预期结果

新资料集在 `mainline/references.html` 中拥有紧凑分类、搜索、原始链接和明确来源边界；目录保持可扫描，不把每条资料渲染成大卡片。

### 数据与 loader 合同

1. 先保存结构化来源快照或本地 manifest，并记录上游 URL、commit/版本、抓取日期、许可证和覆盖限制。对动态来源记录“当前可见集合”，不要声称全量。
2. 在 `mainline/scripts/build_reference_catalog.py` 增加独立 loader。每条记录规范化为 `kind`、`group`、`title`、`meta`、`marker`、`href`、`actions`、`search`、`sort` 和稳定的 `reading_id`。`reading_id` 是浏览器本机阅读档案的键，不能来自会随排序变化的行号。
3. 为新 `kind/group` 增加侧栏分类、数量和稳定 hash 路由；路由必须由生成器同时产生目标 `id`，不能只有 `href="#..."`。
4. 在 `mainline/sources.html` 写清来源归属、快照时间、许可证和不可证明的范围。目录页面只保留题名、最少元数据和动作链接。
5. 重新生成 `references.html`，不要手改生成结果。若上游不可用，使用已声明版本的本地快照；没有可信快照时停止生成，不静默输出空目录。

新资料集若包含本地完整解释，还应提供“完整解释 / 原文 / 代码或项目”入口；只有索引信息时不要伪造完整解释按钮。

## 生成与验证顺序

在规范源完成语义修改后执行：

```bash
cd /Users/franksair/Documents/learning_ML/inference

# 只在参考资料输入或 loader 改动时重建目录；upstream 应是已核验快照。
expected_commit="$(PYTHONDONTWRITEBYTECODE=1 python3 -c 'import runpy; print(runpy.run_path("mainline/scripts/generate_lingyun_catalog.py")["COMMIT"])')"
test "$(git -C /tmp/lingyun-awesome-papers rev-parse HEAD)" = "$expected_commit"
PYTHONDONTWRITEBYTECODE=1 python3 mainline/scripts/build_reference_catalog.py \
  --upstream /tmp/lingyun-awesome-papers

PYTHONDONTWRITEBYTECODE=1 python3 mainline/scripts/validate_mainline.py
PYTHONDONTWRITEBYTECODE=1 python3 mainline/scripts/validate_math_markup.py
PYTHONDONTWRITEBYTECODE=1 python3 prepare_sg/scripts/validate_complete_guides.py  # 仅改逐篇解释时
PYTHONDONTWRITEBYTECODE=1 python3 mainline/scripts/export_to_research_papers_site.py --force

cd /Users/franksair/Documents/Research/papers
./scripts/cloudflare-build.sh
node ./scripts/validate-static-site.mjs _site
./scripts/serve-local.sh
```

若要更新 Systems for ML 快照，先更新 `generate_lingyun_catalog.py` 的 commit/date，再同步 `mainline/sources.html` 和生成的 topic 42；不能拿另一个 `/tmp` revision 配上旧的归属说明。

新线路的 validator 应在导出前运行。最终至少检查：新入口和返回链接、所有本地链接与 hash、图像存在且有 `alt`、桌面与移动端布局、浅色与深色模式、长标题和表格不溢出、参考资料筛选数量正确。文本抽取或脚本通过不能替代浏览器截图。

## 发布边界

- 站点保持纯静态：不增加 Pages Functions、Workers、数据库、KV、R2、队列或付费后端。
- 操作 Cloudflare 控制台时只使用 `Continue with Google` 登录。禁止使用浏览器保存的邮箱密码、Apple、GitHub 或 SSO 登录；登录后必须先核对当前 Cloudflare account 与目标 zone，确认无误后才能修改 DNS、自定义域名、Pages 项目或部署。账号不匹配时立即停止，不尝试在该账号下继续操作。
- 本机阅读档案只用 `localStorage` 保存收藏、阅读状态、最后阅读位置和最多 20 条用户手动标记历史；统一悬浮面板按“最近、收藏、正在读、已读”展示这份本机数据。记录不按时间自动失效，Cookie 只保存长期版本标记；不要自动上传滚动轨迹。除非用户另行批准，不增加账号、密码、身份 Cookie、跨设备同步或服务端副本。
- 全局搜索由 `mainline/scripts/build_global_search_index.py` 在导出时生成内容哈希索引，浏览器只在打开搜索时加载。新增资料线必须进入索引扫描范围；论文原文存档已有完整解释时不重复入库，目录中只有外部来源的条目可以保留外链结果。不要接入托管搜索、Worker 或搜索数据库。
- `_site/` 不提交；提交规范源、生成器、必要快照和导出的 `Research/papers` 内容。
- 提交前按语义拆分内容、线路集成和构建工具，精确暂存路径，不带入其他工作区改动。
- 推送 `Research/main` 可能触发 Cloudflare Pages。先给用户本地预览和变更清单，得到发布确认后再推送。
- 校验失败时保留失败报告和未解决假设，不删减内容来换取绿灯。
