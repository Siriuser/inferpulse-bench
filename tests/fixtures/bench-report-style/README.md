# 报告样式审核快照

对应 [报告样式规范 1.0](../../../docs/design/bench-report-style/README.md)。本目录不参与报告运行或用户安装包加载。

- `base.css`：普通报告完整 CSS，含屏幕和 A4 打印规则。
- `agent-print.css`：启用 Agent 时追加的打印规则。
- `matrix.html`、`line.html`：`ReportFormatTests.style_chart_samples()` 使用人工布局数据生成的组件快照，覆盖零值、缺失、空心点、编号与本地链接。只用于组件比较，不是独立可点击的完整报告。

`test_report_css_matches_accepted_style_baseline` 和 `test_svg_matches_accepted_style_baseline` 自动比较输出。不要盲目重新生成快照；有意调整时同步样式版本、规范及变更说明。快照不包含真实服务配置、凭据、模型回答或测评证据，也不能代替浏览器/A4 视觉验收。
