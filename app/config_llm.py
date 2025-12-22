import os
from typing import Optional, List
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools


_DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "README.md").exists() and (p / "app").is_dir():
            return p
    return start.parent


def _load_dotenv_if_present() -> None:
    """Load environment variables from a local .env file (if present).

    Security note:
    - .env must NOT be committed to the repository.
    - We only set keys that are not already present in os.environ.

    This avoids having to type DEEPSEEK_API_KEY every time.
    """

    try:
        repo_root = _find_repo_root(Path(__file__))
        env_path = repo_root / ".env"
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ.setdefault(key, value)
    except Exception:
        # Never fail app startup because of dotenv parsing.
        return


def _configure_openai_debug_logging() -> None:
    """Optionally enable OpenAI SDK debug logging.

    Controlled by env var OPENAI_LOG (default: empty/off).
    Accepted values: debug, info.
    Compatible with openai==0.28.1.
    """

    level = (os.getenv("OPENAI_LOG") or "").strip().lower()
    if not level:
        return

    try:
        import openai  # openai==0.28.1
    except Exception:
        return

    if level not in {"debug", "info"}:
        return

    try:
        openai.log = level
    except Exception:
        pass


def get_llm(temperature: float = 0.0, model: Optional[str] = None) -> ChatOpenAI:
    """Return a ChatOpenAI instance configured to use DeepSeek's OpenAI-compatible API.

    Environment variables:
    - DEEPSEEK_API_KEY (required; raises RuntimeError if missing/empty)
    - DEEPSEEK_BASE_URL (optional; default https://api.deepseek.com/v1)
    - DEEPSEEK_MODEL (optional; default deepseek-chat)

    For compatibility with older LangChain versions, this function configures the
    OpenAI client primarily through environment variables OPENAI_API_KEY and
    OPENAI_API_BASE.
    """

    _load_dotenv_if_present()

    api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "缺少 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY=你的DeepSeek密钥 后再运行。"
        )

    base_url = (os.getenv("DEEPSEEK_BASE_URL") or _DEFAULT_DEEPSEEK_BASE_URL).strip() or _DEFAULT_DEEPSEEK_BASE_URL
    env_model = (os.getenv("DEEPSEEK_MODEL") or "").strip()
    final_model = (model or env_model or _DEFAULT_DEEPSEEK_MODEL).strip() or _DEFAULT_DEEPSEEK_MODEL

    # LangChain 0.0.352 + openai==0.28.x typically reads these env vars.
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url

    _configure_openai_debug_logging()

    timeout_s = int((os.getenv("OPENAI_REQUEST_TIMEOUT") or "60").strip() or "60")

    return ChatOpenAI(model_name=final_model, temperature=temperature, request_timeout=timeout_s)


def get_tools(llm: ChatOpenAI):
    """Return tools with graceful degradation.

    - Always includes wikipedia.
    - Only includes google-serper when SERPER_API_KEY exists and is non-empty.
    - If SERPER_API_KEY is missing/empty, do NOT error; just skip google-serper.
    """

    tool_names: List[str] = ["wikipedia"]

    serper_key = (os.getenv("SERPER_API_KEY") or "").strip()
    if serper_key:
        tool_names = ["google-serper"] + tool_names

    return load_tools(tool_names, llm=llm)
