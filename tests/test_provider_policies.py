from republic.core import provider_policies


def test_should_use_responses_respects_global_flag() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="openai",
            use_responses=False,
            supports_responses=True,
        )
        is False
    )


def test_should_use_responses_accepts_provider_capability() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="anthropic",
            use_responses=True,
            supports_responses=True,
        )
        is True
    )


def test_should_use_responses_openrouter_policy_fallback() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="openrouter",
            use_responses=True,
            supports_responses=False,
        )
        is True
    )


def test_should_use_responses_requires_explicit_policy_or_capability() -> None:
    assert (
        provider_policies.should_use_responses(
            provider_name="anthropic",
            use_responses=True,
            supports_responses=False,
        )
        is False
    )


def test_completion_stream_usage_policy() -> None:
    assert provider_policies.should_include_completion_stream_usage("openai")
    assert provider_policies.should_include_completion_stream_usage("openrouter")
    assert not provider_policies.should_include_completion_stream_usage("anthropic")


def test_completion_max_tokens_arg_policy() -> None:
    assert provider_policies.completion_max_tokens_arg("openai") == "max_completion_tokens"
    assert provider_policies.completion_max_tokens_arg("openrouter") == "max_tokens"
    assert provider_policies.completion_max_tokens_arg("anthropic") == "max_tokens"
