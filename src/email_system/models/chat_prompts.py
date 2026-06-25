from __future__ import annotations


def messages_for_task(prompt: str, task: str) -> list[dict[str, str]]:
    system = "你是企业邮件处理助手。输出必须严格遵守用户要求，不要添加无关解释。"
    if task == "classify_email":
        user = (
            "请分析下面邮件，只输出 JSON："
            "{\"category\": \"invoice|support|meeting|sales|spam|personal|other\", "
            "\"priority\": \"low|normal|high|urgent\", \"confidence\": 0.0}\n\n"
            f"{prompt}"
        )
    elif task == "summarize_email":
        user = f"请用一句话总结下面邮件，只输出 JSON：{{\"summary\": \"...\", \"confidence\": 0.0}}\n\n{prompt}"
    elif task == "extract_action_items":
        user = (
            "请提取下面邮件里的待办事项，只输出 JSON："
            "{\"action_items\": [{\"owner\": null, \"task\": \"...\", \"due\": null}]}\n\n"
            f"{prompt}"
        )
    elif task == "draft_reply":
        user = f"请根据下面邮件生成一段简洁、专业的中文回复草稿。只输出回复正文。\n\n{prompt}"
    else:
        user = prompt
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
