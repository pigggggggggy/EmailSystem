from __future__ import annotations


PROMPT_VERSION = "email-tasks-20260630-v3"


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
        user = (
            "请根据下面邮件写一封可直接发送给发件人的中文回复草稿。\n"
            "要求：\n"
            "1. 以收件人的身份回复发件人，使用自然礼貌的称呼和结尾。\n"
            "2. 不要复述或总结原邮件；要针对对方诉求给出回应、下一步或婉拒。\n"
            "3. 不要编造事实、承诺付款、点击链接、提供隐私信息或代表用户做未确认决定。\n"
            "4. 如果邮件像广告、陌生交友、钓鱼或垃圾邮件，写成谨慎的拒绝/不回应建议，不要表现出兴趣。\n"
            "5. 正文建议 80 到 180 个中文字符，必要时可分 2 到 3 个短段落。\n"
            "只输出回复正文，不要输出标题、分析、JSON 或说明。\n\n"
            f"{prompt}"
        )
    else:
        user = prompt
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
