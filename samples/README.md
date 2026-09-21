# 图片性能测试样本

`agent-image-grid.png` 是不含用户数据的固定 1024×768 RGB PNG，包含网格、红色方形、绿色圆形、蓝色三角形和四根彩色柱形。进入本目录后，可使用 Python 标准库重新生成：

```bash
python3 generate_agent_image.py
```

在与 `samples` 同级的 JSONC 文件中，将以下字段合并到现有 `agent_performance` 内：

```jsonc
"scenarios": ["long_context", "loop", "image_input"],
"media_samples": [
  {"kind": "image", "path": "samples/agent-image-grid.png", "count": 1},
  {"kind": "image", "path": "samples/agent-image-grid.png", "count": 4}
]
```

不加入 `audio_input`，也无需音频或视频样本。`count: 4` 是同一请求中重复同图四份，用于比较输入负载，不是四张不同图片的理解测试。服务端可能复用媒体缓存，必须结合实际输入用量解释性能差异。PNG 文件压缩后很小并不代表视觉 Token 很少。

该场景记录接口完成情况、失败、截断、延迟、输出用量及并发吞吐；图片内容描述不进行答案评分。请求成功只验证本次图片载荷的协议路径，不能证明视觉理解正确率。图片通过配置的模型接口发送，运行证据只保存媒体元数据和哈希，不保存图片正文。
