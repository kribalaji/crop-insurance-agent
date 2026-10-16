"""Pluggable chat-model factory.

Select the provider with LLM_PROVIDER (gemini | anthropic | ollama) and optionally
override the model with LLM_MODEL. Provider SDKs are imported lazily so only the
one in use has to be importable and configured.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from crop_insurance_agent.config import LLMProvider, Settings, get_settings

DEFAULT_MODELS: dict[LLMProvider, str] = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "llama3.1",
}


class LLMConfigError(RuntimeError):
    """Raised when the selected provider is unknown or missing credentials."""


def get_llm(
    provider: LLMProvider | None = None,
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Return a LangChain chat model for the configured (or given) provider."""
    settings = settings or get_settings()
    provider = provider or settings.llm_provider
    if provider not in DEFAULT_MODELS:
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER {provider!r}; expected one of {sorted(DEFAULT_MODELS)}"
        )
    model = model or settings.llm_model or DEFAULT_MODELS[provider]
    temperature = settings.llm_temperature if temperature is None else temperature

    if provider == "gemini":
        if settings.google_api_key is None:
            raise LLMConfigError("LLM_PROVIDER=gemini requires GOOGLE_API_KEY")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=settings.google_api_key,
        )

    if provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise LLMConfigError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature, base_url=settings.ollama_base_url)
