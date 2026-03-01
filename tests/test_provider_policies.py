from republic.core import provider_policies


def test_should_use_responses_respects_global_flag() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="openai",
            model_id="gpt-4o-mini",
            has_tools=False,
            use_responses=False,
            supports_responses=True,
        )
        is False
    )


def test_should_use_responses_accepts_provider_capability() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="anthropic",
            model_id="claude-3-5-haiku-latest",
            has_tools=False,
            use_responses=True,
            supports_responses=True,
        )
        is True
    )


def test_should_use_responses_openrouter_policy_fallback() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="openrouter",
            model_id="openai/gpt-4o-mini",
            has_tools=False,
            use_responses=True,
            supports_responses=False,
        )
        is True
    )


def test_should_use_responses_requires_explicit_policy_or_capability() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="anthropic",
            model_id="claude-3-5-haiku-latest",
            has_tools=False,
            use_responses=True,
            supports_responses=False,
        )
        is False
    )


def test_should_use_responses_openrouter_anthropic_tools_fallbacks_to_completion() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="openrouter",
            model_id="anthropic/claude-3.5-haiku",
            has_tools=True,
            use_responses=True,
            supports_responses=False,
        )
        is False
    )


def test_should_use_responses_openrouter_anthropic_without_tools_still_uses_responses() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="openrouter",
            model_id="anthropic/claude-3.5-haiku",
            has_tools=False,
            use_responses=True,
            supports_responses=False,
        )
        is True
    )


def test_completion_stream_usage_policy() -> None:
    assert provider_policies.should_include_completion_stream_usage("openai")
    assert provider_policies.should_include_completion_stream_usage("openrouter")
    assert not provider_policies.should_include_completion_stream_usage("anthropic")


def test_completion_max_tokens_arg_policy() -> None:
    assert provider_policies.completion_max_tokens_arg("openai") == "max_completion_tokens"
    assert provider_policies.completion_max_tokens_arg("openrouter") == "max_tokens"
    assert provider_policies.completion_max_tokens_arg("anthropic") == "max_tokens"


def test_provider_policy_uses_exact_match_not_substring() -> None:
    assert not provider_policies.should_include_completion_stream_usage("my-openrouter-proxy")
    assert provider_policies.completion_max_tokens_arg("my-openrouter-proxy") == "max_tokens"
