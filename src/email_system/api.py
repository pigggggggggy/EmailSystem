from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from email_system.agent import EmailAgentWorkflow
from email_system.env import load_local_env
from email_system.imap_mail import IMAPConfig, IMAPEmailClient
from email_system.models import build_llm_client
from email_system.schemas import Email

load_local_env(Path(__file__).resolve().parents[2])


class EmailProcessRequest(BaseModel):
    subject: str = Field(default="", max_length=500)
    sender: str = Field(default="", max_length=320)
    to: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None
    body_text: str = Field(default="", max_length=80000)
    email_id: Optional[str] = None
    thread_id: Optional[str] = None


class EmailProcessResponse(BaseModel):
    output: Dict[str, Any]
    elapsed_ms: float


class GmailRecentRequest(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)
    mailbox: str = Field(default="INBOX", max_length=120)
    search: str = Field(default="ALL", max_length=500)
    host: str = Field(default="imap.gmail.com", max_length=255)
    port: int = Field(default=993, ge=1, le=65535)
    timeout: float = Field(default=20.0, ge=1.0, le=120.0)


class GmailRecentResponse(BaseModel):
    emails: List[Dict[str, Any]]
    elapsed_ms: float


app = FastAPI(
    title="EmailSystem Agent API",
    description="LangGraph email-processing agent wrapped with FastAPI.",
    version="0.1.0",
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


@app.get("/api/health")
def health() -> Dict[str, Any]:
    workflow = _workflow_or_503()
    return {
        "status": "ok",
        "backend": _setting("BACKEND", "mock"),
        "model_path": _setting("MODEL_PATH", "models/Qwen3-4B"),
        "graph_backend": workflow.graph_backend,
    }


@app.post("/api/process", response_model=EmailProcessResponse)
async def process_email(request: EmailProcessRequest) -> EmailProcessResponse:
    if not request.subject.strip() and not request.body_text.strip():
        raise HTTPException(status_code=400, detail="subject or body_text is required")
    email = Email(
        email_id=request.email_id or f"api:{uuid.uuid4().hex}",
        thread_id=request.thread_id,
        subject=request.subject.strip(),
        sender=request.sender.strip(),
        to=[item.strip() for item in request.to if item.strip()],
        cc=[item.strip() for item in request.cc if item.strip()],
        timestamp=request.timestamp,
        body_text=request.body_text.strip(),
    )
    start = time.perf_counter()
    workflow = _workflow_or_503()
    output = await run_in_threadpool(workflow.run, email)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return EmailProcessResponse(output=output.to_dict(), elapsed_ms=elapsed_ms)


@app.post("/api/gmail/recent", response_model=GmailRecentResponse)
async def process_recent_gmail(request: GmailRecentRequest = GmailRecentRequest()) -> GmailRecentResponse:
    user = _secret("IMAP_USER", "EMAILSYSTEM_IMAP_USER")
    password = _secret("IMAP_PASSWORD", "EMAILSYSTEM_IMAP_PASSWORD")
    if not user or not password:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing Gmail IMAP credentials. Set EMAILSYSTEM_IMAP_USER and "
                "EMAILSYSTEM_IMAP_PASSWORD, or EMAILSYSTEM_API_IMAP_USER and "
                "EMAILSYSTEM_API_IMAP_PASSWORD before starting the API."
            ),
        )

    def run_batch() -> List[Dict[str, Any]]:
        client = IMAPEmailClient(
            user=user,
            password=password,
            config=IMAPConfig(
                host=request.host,
                port=request.port,
                mailbox=request.mailbox,
                timeout=request.timeout,
            ),
        )
        emails = client.fetch_recent(limit=request.limit, search=request.search)
        workflow = _workflow_or_503()
        rows = []
        for email in emails:
            output = workflow.run(email).to_dict()
            rows.append(
                {
                    "email": {
                        "email_id": email.email_id,
                        "thread_id": email.thread_id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "to": email.to,
                        "cc": email.cc,
                        "timestamp": email.timestamp,
                        "body_preview": email.body_text[:500],
                    },
                    "output": output,
                }
            )
        return rows

    start = time.perf_counter()
    rows = await run_in_threadpool(run_batch)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return GmailRecentResponse(emails=rows, elapsed_ms=elapsed_ms)


def _workflow_or_503() -> EmailAgentWorkflow:
    try:
        return _workflow()
    except HTTPException:
        raise
    except Exception as exc:
        _workflow.cache_clear()
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model backend failed to start: {type(exc).__name__}: {exc}. "
                "Check GPU memory, stop other GPU processes, choose another CUDA_VISIBLE_DEVICES, "
                "or lower --gpu-memory-utilization."
            ),
        ) from exc


@lru_cache(maxsize=1)
def _workflow() -> EmailAgentWorkflow:
    backend = _setting("BACKEND", "mock")
    llm = build_llm_client(
        backend,
        model_path=_setting("MODEL_PATH", "models/Qwen3-4B"),
        device_map=_setting("DEVICE_MAP", "auto"),
        torch_dtype=_setting("TORCH_DTYPE", "auto"),
        max_model_len=int(_setting("MAX_MODEL_LEN", "8192")),
        tensor_parallel_size=int(_setting("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(_setting("GPU_MEMORY_UTILIZATION", "0.75")),
        enforce_eager=_setting("ENFORCE_EAGER", "true").lower() in {"1", "true", "yes"},
        quantization=_setting("QUANTIZATION", "") or None,
        speculative_model_path=_setting("EAGLE3_MODEL_PATH", "") or None,
        speculative_tokens=int(_setting("SPECULATIVE_TOKENS", "3")),
    )
    return EmailAgentWorkflow(llm)


def _setting(name: str, default: str) -> str:
    return os.environ.get(f"EMAILSYSTEM_API_{name}", default)


def _secret(api_name: str, fallback_env: str) -> str:
    return os.environ.get(f"EMAILSYSTEM_API_{api_name}") or os.environ.get(fallback_env, "")


_INDEX_HTML = r'''
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EmailSystem Agent</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-soft: #f0f3f7;
      --text: #1d2733;
      --muted: #667385;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --warn: #b45309;
      --danger: #b42318;
      --ok: #177245;
      --shadow: 0 18px 48px rgba(22, 34, 51, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app-shell { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(14px);
    }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
    .mark {
      width: 36px; height: 36px; border-radius: 8px;
      background: var(--accent); color: white; display: grid; place-items: center;
      font-weight: 800;
    }
    h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    .subtle { color: var(--muted); font-size: 13px; }
    .status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); white-space: nowrap; }
    .dot { width: 9px; height: 9px; border-radius: 999px; background: var(--warn); }
    .dot.ready { background: var(--ok); }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 0.9fr) minmax(420px, 1.1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1440px;
      width: 100%;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel-head {
      display: flex; justify-content: space-between; align-items: center; gap: 12px;
      padding: 16px 18px; border-bottom: 1px solid var(--line);
    }
    .panel-title { font-weight: 700; font-size: 15px; }
    .form { padding: 16px 18px 18px; display: grid; gap: 12px; }
    label { display: grid; gap: 6px; font-size: 12px; color: var(--muted); font-weight: 650; }
    input, textarea {
      width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 10px 11px;
      font: inherit; color: var(--text); background: #fff; outline: none;
    }
    textarea { min-height: 360px; resize: vertical; line-height: 1.55; }
    input:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.13); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 2px; }
    button {
      border: 0; border-radius: 6px; background: var(--accent); color: white; padding: 10px 14px;
      font: inherit; font-weight: 750; cursor: pointer; min-width: 128px;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { cursor: wait; opacity: 0.72; }
    .results { padding: 16px 18px 18px; display: grid; gap: 14px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .metric { background: var(--panel-soft); border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 70px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .metric strong { display: block; font-size: 17px; overflow-wrap: anywhere; }
    .block { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .block h2 { margin: 0; padding: 10px 12px; font-size: 13px; background: var(--panel-soft); border-bottom: 1px solid var(--line); }
    .block .body { padding: 12px; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; min-height: 48px; }
    .actions-list { margin: 0; padding: 0; list-style: none; display: grid; gap: 8px; }
    .actions-list li { padding: 9px 10px; background: #fff; border: 1px solid var(--line); border-radius: 6px; }
    .secondary { background: #334155; }
    .secondary:hover { background: #1f2937; }
    .batch-list { display: grid; gap: 10px; }
    .email-card { text-align: left; width: 100%; min-width: 0; background: #fff; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 10px; cursor: pointer; }
    .email-card:hover { border-color: var(--accent); background: #f8fbfb; }
    .email-card strong { display: block; overflow-wrap: anywhere; margin-bottom: 4px; }
    .email-card span { display: block; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .trace { display: grid; gap: 8px; }
    .trace-row { display: grid; grid-template-columns: 160px 1fr 90px; gap: 10px; align-items: center; font-size: 13px; }
    .bar { height: 8px; background: var(--panel-soft); border-radius: 999px; overflow: hidden; }
    .bar i { display: block; height: 100%; width: 0; background: var(--accent); border-radius: inherit; }
    .empty { color: var(--muted); }
    .error { color: var(--danger); font-weight: 650; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      textarea { min-height: 260px; }
    }
    @media (max-width: 560px) {
      header { align-items: flex-start; flex-direction: column; padding: 14px 16px; }
      main { padding: 12px; }
      .row, .grid { grid-template-columns: 1fr; }
      .trace-row { grid-template-columns: 1fr; }
      button { width: 100%; }
      .actions { align-items: stretch; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header>
      <div class="brand">
        <div class="mark">@</div>
        <div>
          <h1>EmailSystem Agent</h1>
          <div class="subtle">邮件分类、总结、待办和回复建议</div>
        </div>
      </div>
      <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">连接中</span></div>
    </header>
    <main>
      <section>
        <div class="panel-head">
          <div class="panel-title">输入邮件</div>
          <div class="subtle" id="backendText">backend: -</div>
        </div>
        <form class="form" id="emailForm">
          <label>主题<input id="subject" placeholder="例如：Need help with invoice" value="Juan added you to favorites" /></label>
          <div class="row">
            <label>发件人<input id="sender" placeholder="sender@example.com" value="notify@flirtwish.com" /></label>
            <label>收件人<input id="to" placeholder="me@example.com" value="me@example.com" /></label>
          </div>
          <label>正文<textarea id="body" placeholder="粘贴邮件正文">Juan has added you to favorites on Hotti. Take the lead and chat first!!</textarea></label>
          <div class="actions">
            <div class="subtle">不会发送邮件，只生成建议。</div>
            <div class="row">
              <button id="submitBtn" type="submit">处理邮件</button>
              <button id="gmailBtn" class="secondary" type="button">读取 Gmail 前 10 条</button>
            </div>
          </div>
        </form>
      </section>
      <section>
        <div class="panel-head">
          <div class="panel-title">Agent 输出</div>
          <div class="subtle" id="elapsedText">等待处理</div>
        </div>
        <div class="results">
          <div class="grid">
            <div class="metric"><span>分类</span><strong id="category">-</strong></div>
            <div class="metric"><span>优先级</span><strong id="priority">-</strong></div>
            <div class="metric"><span>置信度</span><strong id="confidence">-</strong></div>
            <div class="metric"><span>审核</span><strong id="review">-</strong></div>
          </div>
          <div class="block"><h2>总结</h2><div class="body empty" id="summary">等待邮件处理结果</div></div>
          <div class="block"><h2>回复建议</h2><div class="body empty" id="reply">等待邮件处理结果</div></div>
          <div class="block"><h2>待办事项</h2><div class="body"><ul class="actions-list" id="actionsList"><li class="empty">暂无</li></ul></div></div>
          <div class="block"><h2>Gmail 批量结果</h2><div class="body batch-list" id="batchList"><div class="empty">暂无</div></div></div>
          <div class="block"><h2>工作流轨迹</h2><div class="body trace" id="trace"><div class="empty">暂无</div></div></div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    const form = $('emailForm');
    const button = $('submitBtn');
    const gmailButton = $('gmailBtn');

    async function loadHealth() {
      try {
        const res = await fetch('/api/health');
        const data = await res.json();
        $('statusDot').classList.add('ready');
        $('statusText').textContent = `${data.status} · ${data.graph_backend}`;
        $('backendText').textContent = `backend: ${data.backend}`;
      } catch (err) {
        $('statusText').textContent = '服务不可用';
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = '处理中';
      $('elapsedText').textContent = 'Agent 正在运行';
      clearError();
      try {
        const payload = {
          subject: $('subject').value,
          sender: $('sender').value,
          to: splitList($('to').value),
          body_text: $('body').value
        };
        const res = await fetch('/api/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '处理失败');
        renderResult(data);
      } catch (err) {
        $('summary').className = 'body error';
        $('summary').textContent = err.message;
        $('elapsedText').textContent = '处理失败';
      } finally {
        button.disabled = false;
        button.textContent = '处理邮件';
      }
    });


    gmailButton.addEventListener('click', async () => {
      gmailButton.disabled = true;
      button.disabled = true;
      gmailButton.textContent = '读取中';
      $('elapsedText').textContent = '正在读取 Gmail 并运行 Agent';
      $('batchList').innerHTML = '<div class="empty">正在处理前 10 条邮件，Qwen3 后端可能需要一些时间</div>';
      try {
        const res = await fetch('/api/gmail/recent', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ limit: 10 })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '读取 Gmail 失败');
        renderBatch(data);
        if (data.emails && data.emails.length) renderResult({ output: data.emails[0].output, elapsed_ms: data.elapsed_ms });
      } catch (err) {
        $('batchList').innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`;
        $('elapsedText').textContent = 'Gmail 处理失败';
      } finally {
        gmailButton.disabled = false;
        button.disabled = false;
        gmailButton.textContent = '读取 Gmail 前 10 条';
      }
    });

    function renderResult(data) {
      const out = data.output;
      $('category').textContent = out.category || '-';
      $('priority').textContent = out.priority || '-';
      $('confidence').textContent = out.confidence ? Number(out.confidence.category || 0).toFixed(2) : '-';
      $('review').textContent = out.requires_human_review ? '需要' : '不需要';
      $('summary').className = 'body';
      $('summary').textContent = out.summary || '无总结';
      $('reply').className = 'body';
      $('reply').textContent = out.reply_draft || '无回复建议';
      $('elapsedText').textContent = `${Math.round(data.elapsed_ms)} ms · ${out.delivery_status || 'no-op'}`;
      renderActions(out.action_items || []);
      renderTrace(out.workflow_trace || []);
    }


    function renderBatch(data) {
      const box = $('batchList');
      const rows = data.emails || [];
      if (!rows.length) {
        box.innerHTML = '<div class="empty">没有读取到邮件</div>';
        return;
      }
      box.innerHTML = '';
      rows.forEach((row, index) => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'email-card';
        const category = row.output.category || '-';
        const review = row.output.requires_human_review ? '需审核' : '可处理';
        card.innerHTML = `<strong>${index + 1}. ${escapeHtml(row.email.subject || '(无主题)')}</strong><span>${escapeHtml(row.email.sender || '')}</span><span>${escapeHtml(category)} · ${escapeHtml(review)} · ${escapeHtml(row.output.summary || '')}</span>`;
        card.addEventListener('click', () => renderResult({ output: row.output, elapsed_ms: data.elapsed_ms }));
        box.appendChild(card);
      });
      $('elapsedText').textContent = `Gmail ${rows.length} 封 · ${Math.round(data.elapsed_ms)} ms`;
    }

    function renderActions(items) {
      const list = $('actionsList');
      list.innerHTML = '';
      if (!items.length) {
        list.innerHTML = '<li class="empty">暂无</li>';
        return;
      }
      for (const item of items) {
        const li = document.createElement('li');
        li.textContent = `${item.task || '未命名事项'}${item.due ? ' · ' + item.due : ''}`;
        list.appendChild(li);
      }
    }

    function renderTrace(trace) {
      const box = $('trace');
      box.innerHTML = '';
      if (!trace.length) {
        box.innerHTML = '<div class="empty">暂无</div>';
        return;
      }
      const max = Math.max(...trace.map(row => row.latency_ms || 0), 1);
      for (const row of trace) {
        const line = document.createElement('div');
        line.className = 'trace-row';
        const width = Math.max(2, ((row.latency_ms || 0) / max) * 100);
        line.innerHTML = `<strong>${escapeHtml(row.node || '')}</strong><div class="bar"><i style="width:${width}%"></i></div><span>${Math.round(row.latency_ms || 0)} ms</span>`;
        box.appendChild(line);
      }
    }

    function splitList(value) { return value.split(',').map(x => x.trim()).filter(Boolean); }
    function clearError() {
      $('summary').className = 'body empty';
      $('summary').textContent = '等待邮件处理结果';
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
    }
    loadHealth();
  </script>
</body>
</html>
'''
