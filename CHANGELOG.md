# 更新记录 / Changelog

## 1.10.2 — 2026-09-21（预发布 / Prerelease）

- Agent 独立输出配额：默认关闭思考 4096 / 开启思考 16384 Token，覆盖每次专项调用。只有预算及其他条件一致时复用常规数据。
- 添加无隐私 PNG 样本与生成器，方便图片输入性能验证。
- 报告先结论、图先表后；压缩重复分析，保留失败、截断及原始明细。
- 首页双模式核心指标卡、A/B/C 异常卡、图表到数据行定位、打印重点标记及模型自评样式。
- 报告样式规范 1.0、CSS/SVG 审核快照与自动回归；无需 JavaScript 或新增运行依赖。

Python 3.9/3.13 各 131 项回归通过。使用既有完整运行证据离线核对 19 张图、256 个图形值及 273 个页内跳转；原始证据和自评不变。完整新版 A4 分页视觉验收仍待完成，因此延续预发布状态。

真实验证边界：一个 Qwen/vLLM 部署中，较高 Agent 配额的图片请求 72/72 正常结束，工具 Loop 36/36 会话完成；长上下文仍有 2/72 次截断，常规测试 283/324 次截断。更高配额、流程完成和协议成功均不保证业务答案质量。该部署拒绝音频输入，没有音频性能结果。历史失败不能被本轮通过覆盖。

Existing HTML/PDF/image examples remain historical synthetic design references, not current performance results. The current layout is defined by the report style baseline. This release preserves all earlier releases and introduces no desktop-product source, real credentials, private endpoints or raw model payloads.
