# EmailSystem 质量与性能评测设计

## 1. 目标

评测回答两个彼此独立的问题：

1. 模型输出是否更准确。
2. 在准确率不明显下降的前提下，推理是否更快。

所有实验保存模型版本、prompt 版本、数据版本、硬件、vLLM 参数和随机种子。最终用质量-速度 Pareto 前沿比较方案，不把质量和速度随意合并成一个分数。

## 2. 数据集与任务边界

当前本地公开数据：

| 数据集 | 样本数 | 标签 |
| --- | ---: | --- |
| Enron Spam | 33,716 | 17,171 spam / 16,545 ham |
| TREC06c | 64,620 | 42,854 spam / 21,766 ham |

这两个数据集只能直接监督 `spam/ham` 二分类，不能证明 `invoice/support/meeting/sales/personal/other` 六类业务意图的准确率。

因此拆成三套测试：

- `spam_detection`：使用 Enron 和 TREC06c，判断 spam 或 ham。
- `business_intent`：使用人工标注的七分类集，标签为 `invoice/support/meeting/sales/spam/personal/other`。
- `agent_e2e`：人工构造或脱敏真实邮件，标注期望分类、路由、人工审核和工具动作。

在七分类结果上评测垃圾邮件时，只做投影：预测为 `spam` 记作 spam，其余六类统一记作 ham。不能把所有 ham 标成 `other` 后计算七分类 Accuracy。

## 3. 数据切分

### 3.1 固定测试集

- Enron 按来源邮箱或原始子语料分组切分，避免同一用户或近重复邮件跨集合泄漏。
- TREC06c 保留官方顺序，建立固定且版本化的测试索引。
- 训练、验证、测试建议为 70% / 10% / 20%。测试索引一旦生成，不因调 prompt 或微调而改变。
- 对 Subject + Body 规范化后计算内容哈希，删除跨 split 重复项。
- 报告完整集结果，同时报告固定平衡子集结果，避免类别比例掩盖模型退化。

### 3.2 分层评测

除总体指标外，按以下切片报告：

- 标签：spam、ham 或七个业务类别。
- 正文长度：0-256、257-1,024、1,025-4,096、4,096+ tokens。
- 邮件形态：纯文本、HTML、回复引用链、空正文、非英文。
- 难例：营销邮件与 spam、销售邮件与 personal、support 与 other。

## 4. 准确性测试

### 4.1 垃圾邮件检测

主指标：

- Macro F1：主要排序指标，兼顾 spam 和 ham。
- Spam recall：漏掉垃圾邮件的比例。
- Spam precision：误杀正常邮件的比例。
- Accuracy：辅助指标。
- Confusion matrix：TP、FP、FN、TN。
- JSON parse success rate：结构化输出稳定性。

同时报告 Enron、TREC06c 各自结果和跨域结果。推荐用 Enron 调参、TREC06c 做外部泛化测试，之后交换一次训练/测试方向验证稳健性。

### 4.2 七类业务意图

现有公开集不覆盖该任务，需要建立独立 gold set：

- 每类至少 200 封，首版共 1,400 封。
- 每封由两人独立标注；冲突由第三人仲裁。
- 同时标注 `category`、`priority`、`requires_human_review` 和期望路由。
- 主指标使用 Macro F1；同时报告每类 Precision、Recall、F1 和混淆矩阵。
- 使用 Cohen's kappa 或 Krippendorff's alpha 报告标注一致性。

### 4.3 摘要、待办与回复

Enron/TREC 没有参考摘要，不能用它们直接计算摘要准确率。另建 200-500 封人工标注集：

- 摘要：ROUGE-L、语义相似度、事实一致性、关键信息覆盖率。
- 待办：任务级 Precision、Recall、F1；owner/due 字段准确率。
- 回复：人工盲评正确性、相关性、语气、是否虚构事实，每项 1-5 分。
- 安全性：不应回复的 spam、钓鱼或高风险邮件是否被阻断。

自动指标只用于筛选，摘要和回复最终以人工盲评为准。比较两个方案时隐藏模型名称并随机交换答案顺序。

### 4.4 Agent 端到端

每个样本标注期望节点和最终动作，计算：

- Route accuracy：`bug_tracking`、`search_documentation` 等分支是否正确。
- Human-review precision/recall：该拦截的是否拦截，正常邮件是否过度拦截。
- Tool selection accuracy：是否选择正确 MCP 工具。
- Tool argument exact match / field F1：工具参数是否完整正确。
- Workflow completion rate：是否无异常到达 END。
- Delivery safety：未获批准时不得真实发送。

## 5. 推理速度测试

质量测试和性能测试分开运行。性能测试仍保存输出，以确认加速没有造成格式或质量异常。

### 5.1 两种服务场景

**在线延迟模式**

- batch size = 1，并发 = 1、4、8。
- 指标：TTFT、每请求生成速度、端到端延迟 p50/p95/p99。
- TTFT 必须通过 vLLM OpenAI 兼容服务的流式接口测量；当前离线 `LLM.generate()` 无法准确测 TTFT。

**离线吞吐模式**

- batch size = 1、8、16、32。
- 固定 1,000 封分层样本。
- 指标：emails/s、input tokens/s、output tokens/s、总 wall time、GPU 峰值显存。

### 5.2 冷启动与热运行

分别记录：

- Cold start：进程启动到模型可服务的时间和模型加载峰值显存。
- Warm latency：先预热 20 个请求，再正式计时。
- 每个配置独立运行 3 次，报告中位数和波动范围。
- 首次运行不得混入 CUDA kernel 初始化、模型加载或数据下载时间。

### 5.3 分节点性能

当前每封邮件调用 Qwen 四次：分类、摘要、待办、回复。分别记录：

- 输入/输出 token 数。
- 模型推理耗时。
- 节点 wall time。
- tokens/s。

还要单独报告非模型节点耗时，例如 IMAP、记忆检索和 MCP。端到端延迟不能用各节点平均值替代，应直接测量整封邮件从输入到最终输出的 wall time。

## 6. 公平比较控制项

比较 baseline、prompt、LoRA、量化或推测解码时固定：

- 同一 GPU 型号、数量、驱动、CUDA 和 vLLM 版本。
- 同一测试索引、预处理、prompt、chat template 和最大上下文。
- `temperature=0`，固定最大输出 token 数和停止条件。
- 同一 batch、并发和输入长度分布。
- 机器空闲，记录其他 GPU 进程。

每次实验至少输出：

```text
outputs/runs/<run_id>/
  config.json
  dataset_manifest.json
  predictions.jsonl
  traces.jsonl
  metrics.json
  report.md
```

## 7. 实验矩阵

首轮只跑以下配置，避免变量同时变化：

| 实验 | 模型 | 推理配置 | 目的 |
| --- | --- | --- | --- |
| B0 | Qwen3-4B | vLLM FP/BF16 | 基线 |
| Q1 | Qwen3-4B | 改进分类 prompt | 测质量变化 |
| Q2 | Qwen3-4B + LoRA | 与 B0 相同 | 测微调收益 |
| S1 | Qwen3-4B | 提高 batch | 测吞吐 |
| S2 | Qwen3-4B | 量化 | 测显存/速度/质量权衡 |
| S3 | Qwen3-4B | 推测解码 | 测延迟收益 |

一次实验只改变一个主要变量。所有加速方案必须重新跑完整质量集。

## 8. 首版验收门槛

建议用 B0 基线锁定后再调整数值。初始门槛：

- Spam/Ham Macro F1 >= 0.90。
- Spam recall >= 0.95。
- 七分类 Macro F1 >= 0.75。
- JSON parse success rate >= 99.5%。
- Workflow completion rate >= 99%。
- 加速方案相对 B0 的 Macro F1 下降不超过 0.5 个百分点。
- 加速方案 warm p95 延迟降低至少 15%，或离线吞吐提高至少 20%。

若速度提升达标但质量越界，该方案不进入默认配置，只保留为可选性能档。

## 9. 实施顺序

1. 构建 Enron/TREC 的规范化 JSONL 和固定 split manifest。
2. 增加独立 `spam_detection` 评测，避免与七分类混算。
3. 给 vLLM client 增加 batch API、精确 token 计数和直接端到端计时。
4. 增加服务模式 benchmark，测 TTFT、p95/p99 和并发吞吐。
5. 建立七分类与摘要人工 gold set。
6. 生成 baseline 报告，再开始 prompt、LoRA 和推理加速实验。
