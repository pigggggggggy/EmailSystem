# EmailSystem 架构设计

目标：基于 Qwen3-4B 构建一个全链路邮件处理智能体，覆盖邮件接入、清洗、分类、摘要、行动项抽取、回复建议、工具调用、评测、微调和部署。

## 总体分层

```text
EmailSystem/
  README.md
  docs/
    architecture.md
    evaluation.md
    data.md
    agent_workflow.md
    finetuning.md
  configs/
    model/
    agent/
    mcp/
    eval/
  data/
    raw/
    interim/
    processed/
    labels/
    eval_sets/
  src/
    email_system/
      ingestion/
      preprocessing/
      schemas/
      agent/
      skills/
      mcp/
      models/
      evaluation/
      storage/
      observability/
      utils/
  scripts/
    build_dataset.py
    run_agent.py
    run_eval.py
    finetune.py
    export_model.py
  tests/
    unit/
    integration/
    fixtures/
  notebooks/
  outputs/
    runs/
    reports/
    predictions/
  third_party/
```

## 核心模块

### 1. 邮件接入与预处理

路径：`src/email_system/ingestion/`、`src/email_system/preprocessing/`

职责：
- 接入本地 `.eml`、`.mbox`、CSV、JSONL 或 IMAP 导出的邮件。
- 解析主题、正文、发件人、收件人、时间、附件元数据、邮件线程。
- 清洗 HTML、签名、引用历史、免责声明和乱码。
- 输出统一结构，供智能体、评测和微调共用。

建议统一数据格式：

```json
{
  "email_id": "string",
  "thread_id": "string",
  "subject": "string",
  "from": "string",
  "to": ["string"],
  "cc": ["string"],
  "timestamp": "ISO-8601",
  "body_text": "string",
  "attachments": [],
  "labels": {
    "category": "invoice|support|meeting|sales|spam|personal|other",
    "priority": "low|normal|high|urgent"
  }
}
```

### 2. 智能体工作流

路径：`src/email_system/agent/`

建议先实现一个可观测、可评测的确定性工作流，再逐步加入更复杂的规划能力。

基础流程：

```text
EmailInput
  -> NormalizeEmail
  -> DetectLanguage
  -> ClassifyIntent
  -> ExtractEntities
  -> Summarize
  -> ExtractActionItems
  -> DecideToolUse
  -> DraftReply
  -> HumanReviewPolicy
  -> FinalOutput
```

输出结构：

```json
{
  "category": "support",
  "priority": "high",
  "summary": "客户反馈无法登录后台，需要尽快排查账号状态。",
  "action_items": [
    {
      "owner": "support_team",
      "task": "检查客户账号状态",
      "due": null
    }
  ],
  "entities": {
    "people": [],
    "companies": [],
    "dates": [],
    "order_ids": []
  },
  "reply_draft": "您好，我们已经收到您的问题...",
  "confidence": {
    "category": 0.91,
    "summary": 0.86
  },
  "requires_human_review": true
}
```

### 3. Skills 设计

路径：`src/email_system/skills/`

每个 skill 做成可单独测试、可被 agent 编排的纯能力模块。

建议首批 skills：
- `classify_email`: 邮件类别、优先级、是否垃圾邮件。
- `summarize_email`: 单封邮件摘要、线程摘要、长邮件压缩。
- `extract_action_items`: 提取待办、负责人、截止日期。
- `extract_entities`: 提取人名、公司、金额、订单号、日期。
- `draft_reply`: 根据语气、类别、历史上下文生成回复草稿。
- `route_email`: 给出分派团队、标签、SLA。
- `risk_check`: 识别敏感信息、钓鱼、合规风险。

建议接口：

```python
class Skill:
    name: str

    def run(self, input: dict, context: dict) -> dict:
        ...
```

### 4. MCP 设计

路径：`src/email_system/mcp/`、`configs/mcp/`

MCP 用来把智能体连接到外部系统，但建议一开始用 mock/local 实现，避免工程刚起步就被真实服务绑定。

建议首批 MCP server：
- `email_store`: 查询邮件、线程、历史相似邮件。
- `contact_book`: 查询联系人、部门、客户等级。
- `calendar`: 查询日程、创建会议草稿。
- `ticketing`: 创建工单、查询工单状态。
- `knowledge_base`: 检索内部 FAQ、政策、历史答复。

MCP 工具示例：

```text
email_store.search(query, sender, date_range)
email_store.get_thread(thread_id)
contact_book.lookup(email_address)
ticketing.create_ticket(title, body, priority)
knowledge_base.search(query, top_k)
```

### 5. Qwen3-4B 模型层

路径：`src/email_system/models/`、`configs/model/`

建议分三层：
- `BaseLLMClient`: 抽象生成、批量生成、流式生成。
- `QwenLocalClient`: 本地 vLLM 或 transformers 推理。
- `QwenFinetunedClient`: 指向 LoRA/QLoRA 微调后的模型。

优先支持的推理后端：
- vLLM：评测速度、吞吐、部署优先。
- transformers：调试和微调流程优先。

配置建议：

```yaml
model_name_or_path: Qwen/Qwen3-4B
backend: vllm
max_model_len: 8192
temperature: 0.2
top_p: 0.9
max_tokens: 1024
```

### 6. 微调任务

路径：`scripts/finetune.py`、`configs/model/`、`data/processed/`

建议先做 LoRA/QLoRA，不直接全量微调。

任务拆分：
- 分类 SFT：输入邮件，输出类别、优先级、理由。
- 摘要 SFT：输入邮件或线程，输出结构化摘要。
- 行动项抽取 SFT：输出 JSON 待办列表。
- 回复草稿 SFT：输出符合风格和政策的回复。
- 偏好优化：后续用 DPO/ORPO 优化回复质量。

训练样本统一为 JSONL：

```json
{
  "task": "summarize_email",
  "messages": [
    {"role": "system", "content": "你是企业邮件处理助手。"},
    {"role": "user", "content": "邮件内容..."},
    {"role": "assistant", "content": "{\"summary\":\"...\"}"}
  ],
  "metadata": {
    "email_id": "email-001",
    "source": "manual_label"
  }
}
```

### 7. 评测体系

路径：`src/email_system/evaluation/`、`configs/eval/`、`outputs/reports/`

评测分为离线质量评测、性能评测和端到端工作流评测。

分类指标：
- Accuracy
- Macro F1
- Per-class Precision/Recall/F1
- Confusion Matrix

摘要指标：
- ROUGE-L
- BERTScore 或 embedding similarity
- 长度控制误差
- JSON 格式有效率
- 人工评分字段：忠实性、覆盖度、简洁度、可执行性

速度指标：
- 单封邮件端到端延迟
- 摘要生成 tokens/s
- 首 token 延迟 TTFT
- 每秒处理邮件数
- 批处理吞吐
- GPU 显存峰值

智能体指标：
- 工具调用成功率
- 工具调用必要性准确率
- 工单/日程等动作字段完整率
- Human review 触发准确率
- 结构化输出解析成功率

建议评测输出：

```text
outputs/runs/<timestamp>/
  config.yaml
  predictions.jsonl
  metrics.json
  report.md
  traces.jsonl
```

### 8. 可观测性

路径：`src/email_system/observability/`

每次运行记录：
- 输入邮件 ID，不记录敏感原文到日志。
- prompt 版本。
- model 版本。
- skill 执行时间。
- MCP 工具调用参数摘要。
- token 用量。
- 输出解析错误。
- 评测指标。

### 9. 测试策略

路径：`tests/`

优先测试：
- 邮件解析边界：HTML、引用链、空主题、多收件人。
- JSON schema 输出校验。
- 每个 skill 的 deterministic mock 测试。
- agent workflow 集成测试。
- eval metrics 的小样本正确性。

## 推荐里程碑

### Milestone 1：最小可运行闭环

目标：本地输入邮件 JSONL，跑 Qwen3-4B，输出分类、摘要、行动项和回复草稿。

产物：
- 基础数据 schema。
- Qwen client。
- 4 个核心 skills。
- `scripts/run_agent.py`。
- 小型样例数据。

### Milestone 2：评测闭环

目标：可以稳定比较 prompt、模型和配置。

产物：
- 分类 Accuracy/Macro F1。
- 摘要速度 tokens/s、端到端延迟。
- `scripts/run_eval.py`。
- `outputs/reports/*.md`。

### Milestone 3：MCP 工具闭环

目标：agent 可以查询本地邮件库、联系人和知识库。

产物：
- mock MCP servers。
- agent tool decision。
- tool trace。

### Milestone 4：LoRA 微调

目标：用标注数据提升分类、摘要和结构化抽取稳定性。

产物：
- SFT 数据构建脚本。
- LoRA 训练配置。
- 微调模型导出。
- base vs finetuned 评测报告。

### Milestone 5：生产化雏形

目标：形成可服务化的邮件处理系统。

产物：
- API 服务。
- 批处理队列。
- 权限和敏感信息处理。
- dashboard/report。

## 近期建议

下一步建议先落地 Milestone 1 和 Milestone 2 的骨架：
- 建 `src/email_system/` Python 包。
- 定义邮件和 agent 输出 schema。
- 实现 mock LLM client，保证无 GPU 时也能跑单元测试。
- 再接入 Qwen3-4B vLLM。
- 先做 20-100 封邮件的小评测集，避免一开始陷入数据工程泥潭。
