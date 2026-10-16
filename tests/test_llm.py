import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from crop_insurance_agent.config import Settings
from crop_insurance_agent.llm import DEFAULT_MODELS, LLMConfigError, get_llm


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_default_provider_is_gemini():
    llm = get_llm(settings=_settings(google_api_key="test"))
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model.removeprefix("models/") == DEFAULT_MODELS["gemini"]


def test_anthropic_with_model_override():
    s = _settings(llm_provider="anthropic", anthropic_api_key="test", llm_model="claude-x")
    llm = get_llm(settings=s)
    assert isinstance(llm, ChatAnthropic)
    assert llm.model == "claude-x"


def test_ollama_needs_no_key():
    llm = get_llm(provider="ollama", settings=_settings())
    assert isinstance(llm, ChatOllama)
    assert llm.base_url == "http://localhost:11434"


@pytest.mark.parametrize("provider", ["gemini", "anthropic"])
def test_missing_key_raises(provider):
    with pytest.raises(LLMConfigError, match="API_KEY"):
        get_llm(provider=provider, settings=_settings())


def test_unknown_provider_raises():
    with pytest.raises(LLMConfigError, match="Unknown"):
        get_llm(provider="openai", settings=_settings())  # type: ignore[arg-type]
