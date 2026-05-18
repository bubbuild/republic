from republic.core import provider_policies


def test_responses_rejection_reason_follows_sdk_metadata() -> None:
    reason = provider_policies.responses_rejection_reason(
        provider_name="openrouter",
        model_id="openai/gpt-4o-mini",
        supports_responses=False,
    )
    assert reason == "responses format is not supported by this provider"


def test_responses_rejection_reason_for_provider_without_responses() -> None:
    reason = provider_policies.responses_rejection_reason(
        provider_name="anthropic",
        model_id="claude-3-5-haiku-latest",
        supports_responses=False,
    )
    assert reason is not None
    assert "not supported" in reason


def test_supports_messages_format() -> None:
    assert provider_policies.supports_messages_format(
        provider_name="anthropic",
        model_id="claude-3-5-haiku-latest",
    )
    assert provider_policies.supports_messages_format(
        provider_name="openrouter",
        model_id="anthropic/claude-3.5-haiku",
    )
    assert provider_policies.supports_messages_format(
        provider_name="openai",
        model_id="gpt-4o-mini",
    )


def test_completion_stream_usage_policy() -> None:
    assert provider_policies.should_include_completion_stream_usage("openai")
    assert provider_policies.should_include_completion_stream_usage("openrouter")
    assert provider_policies.should_include_completion_stream_usage("github-copilot")
    assert not provider_policies.should_include_completion_stream_usage("anthropic")


def test_provider_policy_uses_exact_match_not_substring() -> None:
    assert not provider_policies.should_include_completion_stream_usage("my-openrouter-proxy")
