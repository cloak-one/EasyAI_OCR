# -*- coding: utf-8 -*-
"""
全局环境配置
-----------
从项目根目录的 .env 文件加载配置，所有敏感信息（如 API 密钥）统一在此管理。
"""

import os


def _load_dotenv():
    """加载 .env 文件到环境变量（兼容有无 python-dotenv 的情况）"""
    try:
        from dotenv import load_dotenv as _load
        _load()
        return
    except ImportError:
        pass

    # 未安装 python-dotenv 时，手动解析 .env
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(_env_path):
        return
    with open(_env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

# ============================================================================
# DashScope  API 密钥
# ============================================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError(
        "未设置 DASHSCOPE_API_KEY，请在项目根目录的 .env 文件中配置：\n"
        '    DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n'
        "（可参考 .env.example 文件）"
    )