from __future__ import annotations


PROMPT_VERSION = "email-tasks-20260629-v2"


def messages_for_task(prompt: str, task: str) -> list[dict[str, str]]:
    system = "你是企业邮件处理助手。输出必须严格遵守用户要求，不要添加无关解释。"
    if task == "classify_email":
        user = (
            "请分析下面邮件并只输出一个 JSON 对象。\n"
            "category 只能是 invoice、support、meeting、sales、spam、personal、other 之一。\n"
            "priority 只能是 low、normal、high、urgent 之一。\n"
            "confidence 必须是 0 到 1 之间的数字，表示你对 category 判断的真实把握；"
            "证据不足时降低该值，不要固定照抄示例。\n"
            "输出示例：{\"category\": \"other\", \"priority\": \"normal\", \"confidence\": 0.85}\n\n"
            f"{prompt}"
        )
    elif task == "summarize_email":
        user = (
            "请用一句话总结下面邮件，只输出 JSON。confidence 必须是 0 到 1 之间的数字。\n"
            f"输出示例：{{\"summary\": \"...\", \"confidence\": 0.85}}\n\n{prompt}"
        )
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
