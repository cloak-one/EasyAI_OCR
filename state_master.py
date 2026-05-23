# state_master.py
# -*- coding: utf-8 -*-
IDLE = "IDLE"
CHAT = "CHAT"
FIND_PAGE = "FIND_PAGE"


class StateMaster:
    """
    轻量状态机：
    - 仅保留当前业务使用的意图状态切换能力；
    - 按意图切换到 CHAT / FIND_PAGE。
    """

    def __init__(self):
        self.state = IDLE
        self.last_intent: str = "idle"

    def set_intent_state(self, intent: str):
        intent = (intent or "chat").strip().lower()
        mapping = {
            "find_page": FIND_PAGE,
            "chat": CHAT,
        }
        next_state = mapping.get(intent, CHAT)

        self.state = next_state
        self.last_intent = intent
