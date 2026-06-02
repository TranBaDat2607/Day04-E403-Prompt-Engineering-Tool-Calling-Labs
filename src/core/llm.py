from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# Gemini's free tier caps generate_content at a few requests per minute (5 RPM for
# gemini-2.5-flash). The grader fires many calls in a row (agent loop + LLM judge across
# every case), so without throttling it trips a 429 RESOURCE_EXHAUSTED almost immediately.
# A single shared rate limiter keeps the COMBINED request rate (every model instance the
# process builds) safely under that ceiling. Tune via LLM_REQUESTS_PER_MINUTE.
def _build_google_rate_limiter():
    try:
        from langchain_core.rate_limiters import InMemoryRateLimiter
    except ImportError:
        return None
    requests_per_minute = float(os.getenv("LLM_REQUESTS_PER_MINUTE", "4"))
    if requests_per_minute <= 0:
        return None
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60.0,
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )


_GOOGLE_RATE_LIMITER = _build_google_rate_limiter()


def normalize_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        text = raw.get("text")
        return str(text).strip() if text is not None else str(raw).strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            text = normalize_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(raw).strip()


def build_chat_model(
    *,
    provider: str = "google",
    model_name: str | None = None,
    temperature: float = 0.0,
):
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            rate_limiter=_GOOGLE_RATE_LIMITER,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        # OpenAI's rate limits are far higher than Gemini's free tier, so no throttling
        # is needed here — this is the fast path for grading.
        return ChatOpenAI(
            model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
    if provider == "custom":
        from langchain_openai import ChatOpenAI

        # OpenAI-compatible endpoint configured entirely via .env
        # (CUSTOM_LLM_MODEL / CUSTOM_LLM_KEY / CUSTOM_LLM_URL).
        return ChatOpenAI(
            model=model_name or os.getenv("CUSTOM_LLM_MODEL"),
            openai_api_key=os.getenv("CUSTOM_LLM_KEY"),
            openai_api_base=os.getenv("CUSTOM_LLM_URL"),
            temperature=temperature,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    raise ValueError("This lab supports only the `google`, `openai`, `custom`, and `ollama` providers.")


def extract_json_object(raw: Any) -> dict[str, Any]:
    text = normalize_content(raw)
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def judge_answer_with_llm(
    *,
    query: str,
    answer: str,
    rubric: str,
    provider: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    prompt = f"""
You are grading a student order-agent answer.
Return JSON only with:
- score: integer from 0 to 10
- verdict: short string
- feedback: short list of strings

Rubric:
{rubric}

User query:
{query}

Student answer:
{answer}
""".strip()
    payload = extract_json_object(model.invoke(prompt).content)
    score = max(0, min(10, int(payload.get("score", 0))))
    return {
        "score": score,
        "verdict": str(payload.get("verdict", "")).strip(),
        "feedback": [str(item).strip() for item in payload.get("feedback", []) if str(item).strip()],
    }
