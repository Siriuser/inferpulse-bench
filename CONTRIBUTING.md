# 反馈与贡献 / Contributing

欢迎文档纠错、可复现的缺陷报告和围绕本地模型性能测试的改进。

## 报告问题

在 [Issues](https://gitee.com/xum1983/inferpulse-bench/issues) 中提供：

- 脚本版本（`python3 llm_benchmark.py --version`）、Python 版本和操作系统。
- 模型系列、推理框架及版本（如果知道）、是否经过兼容网关。
- 所选模式、字符档、并发、输出预算、轮数、预热与超时。
- 最小复现步骤、期望与实际行为、HTTP 状态或脚本错误类型。
- 经检查的 dry-run 计划或必要报告片段，注明能否稳定复现。

不要上传 API Key、填过凭据的配置、真实内网地址、请求/响应正文或未经检查的证据包。模型名称也可能包含内部标识。地址请替换为 localhost 等占位值。

涉及凭据泄漏等不适合公开描述的问题，请先联系 **xum1983@gmail.com**，仅提供必要的脱敏信息。项目没有承诺固定响应时限。

## 提交改进

1. 创建 Fork 和分支，描述改动解决的具体问题。涉及指标定义、适配范围或配置兼容性时，先在 Issue 中明确预期。
2. 保持 Python 3.9+、运行时零第三方依赖，不混入完整 InferPulse GUI 的代码或依赖。
3. 同步两份 README 中受影响的用法、默认值与限制；示例使用空 Key 和占位地址。
4. 在 Pull Request 中说明验证结果。以下基础检查不请求模型：

```bash
python3 -m py_compile llm_benchmark.py
python3 llm_benchmark.py --config llm_benchmark.deepseek.example.jsonc --dry-run
python3 llm_benchmark.py --config llm_benchmark.qwen.example.jsonc --dry-run
```

编译和 dry-run 不能代替行为验证。修改流式解析、指标、取消或证据时，请附最小模拟响应/测试，覆盖失败路径，并说明是否做过真实服务验证。不要通过删参数重试、估算 Token 或忽略失败让结果看起来成功。

代码与文档贡献使用本仓库的 MIT 许可证，请保留版权与许可声明。报告示例须确认来源与公开范围。

---

## Reporting issues

Use [Issues](https://gitee.com/xum1983/inferpulse-bench/issues). Include script/Python/OS versions, model family and serving framework if known, selected modes and test settings, minimal reproduction steps, expected and actual behavior, and the HTTP status or error category. State whether it reproduces consistently.

Share only reviewed dry-run output or necessary report excerpts. Remove keys, private endpoints, internal model identifiers and request/response bodies. Never attach a filled runtime configuration or an unchecked evidence bundle. For sensitive credential-related problems, email **xum1983@gmail.com** with minimal redacted details; no fixed response time is promised.

## Pull requests

Fork the repository and make a focused change. Discuss metric definitions, model support or configuration compatibility in an Issue first. Keep Python 3.9+, the standard-library-only runtime and both READMEs in sync.

Run the compile and two dry-run commands above. They make no model requests and do not replace behavioral tests. For changes to parsing, metrics, cancellation or evidence, include a minimal mock response/test covering the relevant failure path, and state what real-service validation, if any, was performed. Do not hide failures, estimate token usage or drop rejected parameters to manufacture success.

Code and documentation contributions use this repository's MIT license. Keep copyright and license notices. Report examples must have a clear source and be suitable for public sharing.
