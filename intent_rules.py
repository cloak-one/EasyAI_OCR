# -*- coding: utf-8 -*-

import re
from typing import Iterable


NON_VISUAL_EXACT = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "继续",
    "停止",
    "停下",
    "停一下",
    "暂停",
    "算了",
    "别说了",
    "打住",
    "听",
    "嗯",
    "啊",
    "哦",
    "好的",
    "谢谢",
    "收到",
}

NON_VISUAL_SUBSTRINGS = (
    "听得到吗",
    "能听见吗",
    "继续说",
    "继续讲",
    "停止播放",
    "先停一下",
)

VISUAL_STRONG_SUBSTRINGS = (
    "看图",
    "图片",
    "照片",
    "画面",
    "场景",
    "描述",
    "介绍",
    "说明",
    "讲讲",
    "说说",
    "识别",
    "总结",
    "镜头",
    "摄像头",
    "图里",
    "这张图",
    "这幅图",
    "二维码",
    "条形码",
    "识别一下",
    "识别这个",
    "读一下这个",
    "读出这个",
    "帮我找",
    "帮我看看",
    "帮我看下",
    "看看周围",
    "周围环境",
    "当前环境",
    "哪一页",
    "第几页",
    "看看这个",
    "看看这里",
    "看看现在",
)

VISUAL_DEICTIC_WORDS = (
    "这个",
    "这里",
    "这边",
    "当前",
    "现在",
    "上面",
    "下面",
    "左边",
    "右边",
    "前面",
    "后面",
)

VISUAL_ACTION_WORDS = (
    "是什么",
    "写了什么",
    "有没有",
    "看到什么",
    "能看见",
    "能看清",
    "内容",
    "文字",
    "页码",
    "哪一页",
    "翻到哪",
)


def _compact_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.strip().lower()
    return "".join(ch for ch in lowered if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"))


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def is_visual_intent(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return False

    if compact in NON_VISUAL_EXACT:
        return False
    if _contains_any(compact, NON_VISUAL_SUBSTRINGS):
        return False

    if re.search(r"第[0-9一二三四五六七八九十百千万两]+页", compact):
        return True

    if _contains_any(compact, VISUAL_STRONG_SUBSTRINGS):
        return True

    if _contains_any(compact, VISUAL_DEICTIC_WORDS) and _contains_any(compact, VISUAL_ACTION_WORDS):
        return True

    return False
