# Provider samples: provenance

Inputs for `tests/test_provider_golden.py`. Every case under
`cases/provider_sample_*` here is a response body recorded by a real client against the real
service: six cases holding ten byte-verified bodies. Nothing here is a live recording made by us:
no credentials were used and no model was called, and re-recording is out of scope for a test-run
environment.

Schema-constructed samples are not recordings and are deliberately absent from this file.

Each entry records the local SHA-256, the immutable upstream location, the license, the exact
transformations applied, and how the sample is classified. Expected values in the tests come from
the recorded payload's fields and the contract evidence identified below, never observed Republic
output. Republic source identifies routing and local integration conventions;
source inspection alone does not establish a provider-wide wire requirement.

## Case index

| Case | Provider family | Kind | Bodies |
| --- | --- | --- | --- |
| `provider_sample_zai_glm_reasoning` | Z.ai (ZAI) | recorded | `0.json` |
| `provider_sample_zai_glm_tool` | Z.ai (ZAI) | recorded | `0.json`, `1.json` |
| `provider_sample_grok_tool` | Grok (xAI direct) | recorded | `0.json`, `1.json` |
| `provider_sample_openrouter_kimi_tool` | OpenRouter (Kimi upstream) | recorded | `0.json`, `1.json` |
| `provider_sample_openrouter_grok_streaming` | OpenRouter (xAI upstream) | recorded | `0.sse` |
| `provider_sample_azure_tool` | Azure OpenAI | recorded | `0.json`, `1.json` |

Bodies are numbered in the order Republic sends them. A `tool*` case holds the first response (the
model's tool call) and the second (the answer after the caller's tool output), so the recorded
request bodies also show what a real client sent back.

## Recorded samples: Fantasy VCR cassettes

Source repository: [`charmbracelet/fantasy`](https://github.com/charmbracelet/fantasy) at commit
`b0fadc9ae2e7da81a5d0262563d3ced9b594bd75`, license Apache-2.0
([`LICENSE`](https://github.com/charmbracelet/fantasy/blob/b0fadc9ae2e7da81a5d0262563d3ced9b594bd75/LICENSE),
SHA-256 `0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594`). A copy is kept beside
this file as `LICENSE.fantasy`.

These cassettes are `go-vcr` recordings of the provider's own test suite: each interaction stores
a real outgoing request and the real response body, and the suite replays them through an HTTP
transport. The bodies below are the `response.body` field verbatim, so they are genuine
provider-authored API responses, not hand-written fixtures.

We fetch from `raw.githubusercontent.com` at the pinned commit and compare SHA-256 against the same
path in the commit tarball, so the bytes are verified against two independent transports.

### `provider_sample_zai_glm_reasoning/0.json`

- Case: `TestOpenAICompatibleCommon/zai-glm-4-7-flash/simple.yaml`
- Path: `providertests/testdata/TestOpenAICompatibleCommon/zai-glm-4-7-flash/simple.yaml`
- Upstream file SHA-256: `f435f5086e66e034a1629a0d028026fbc1aad6ea17a8b77b0777fd937b86315d`
- Local SHA-256: `ef3ef5b8a4d45d15c6f213693de9a31daa0d368f2b83297490605d5fdc73bbb6`
- Classification: recorded, upstream real response
- Transformation: the YAML `response.body` string is written out as bytes. No reformatting, no field
  edits, no re-encoding.
- Why it is necessary: GLM chat answers with `reasoning_content` beside `content`, reports
  `completion_tokens_details.reasoning_tokens` and `prompt_tokens_details.cached_tokens`.
  This is the only family here whose recorded
  non-streaming answer keeps thinking and answer text apart in one message.
- Limitations: the recording is one short greeting; it exercises usage and reasoning decoding, not
  a long answer. Its request uses `/api/coding/paas/v4/chat/completions`; it does not establish
  Republic's default `/api/paas/v4` base. The URL assertion checks Republic's configured routing,
  not a route verified by this cassette. The cassette uses `max_tokens`, so it does not show what
  Z.ai does when
  `max_completion_tokens` is sent — the test asserts Republic's own mapping instead.

### `provider_sample_zai_glm_tool/0.json`, `1.json`

- Case: `TestOpenAICompatibleCommon/zai-glm-4-7-flash/tool.yaml`
- Upstream file SHA-256: `dc98aee75f2f046941fb6c271d7e4a5ee3900a8cd2363d0b589375482867e5f0`
- Local SHA-256: `8d417387f2e02926bee4edddd39b9edae3c60471e6a62f4a8531386d0b235fa0` (`0.json`),
  `087bcb0050ff0c982fc6eb97ecff5e23febd56354c1dabe9f5a910a47ed2149e` (`1.json`)
- Classification: recorded, upstream real responses
- Transformation: byte-identical extraction of both `response.body` values.
- Why it is necessary: the round trip is the point. The first response carries a Z.ai tool call
  (`call_-7807249954186984837`) and thinking; the recorded second request shows the real client
  resending that thinking as `reasoning_content`. Republic's `MaxTokensReasoningContentChat` exists
  for exactly this, and only a recorded pair can show it.
- Limitations: one tool, one turn. The recorded arguments `{"location":"Florence,Italy"}` are
  treated as an opaque string by the test, which compares them to the decoded call rather than
  re-encoding them.

### `provider_sample_grok_tool/0.json`, `1.json`

- Case: `TestOpenAICompatibleCommon/xai-grok-4-fast/tool.yaml`
- Upstream file SHA-256: `2ace0fca86c730b56e7efca378bf919d418fbf1407bf242bdea4c243997b5f28`
- Local SHA-256: `6c674d6b5a3f86e88e18338e67b279c7cf75439fe66751c7002d972746d349cc` (`0.json`),
  `ad6987149618b8f1ec90b2cd953670234e98d47453ca77884a0b6d370eeac718` (`1.json`)
- Classification: recorded, upstream real responses
- Transformation: byte-identical extraction of both `response.body` values.
- Why it is necessary: this is xAI's own `api.x.ai` endpoint, not a gateway. It shows Grok's
  non-OpenAI usage envelope (`reasoning_tokens`, `cached_tokens`, `cost_in_usd_ticks`) and that no
  reasoning field is replayed on the next turn — the counterpart to the Z.ai case.
- Limitations: `grok-4-fast-reasoning` reports reasoning tokens but no reasoning text in this
  recording, so the test asserts token accounting and replay shape, not reasoning decoding.
  Republic's `Grok` provider uses the xAI OAuth login by default; the test passes
  `api_key="test-key"` with `api_format="chat"`, because a bearer header is what the recorded
  request used. The OAuth flow itself is covered by `tests/test_grok.py`.

### `provider_sample_openrouter_kimi_tool/0.json`, `1.json`

- Case: `TestOpenRouterCommon/kimi/tool.yaml`
- Upstream file SHA-256: `bfc1bca915d600116d6aabf58f121dd30e7e11b713f8cd088a9743444d0293d4`
- Local SHA-256: `4fcf68ece5e374717a4c89c2c8f43142c31ebaca91c5a1792c53b74fb5cb215f` (`0.json`),
  `f6e81e87249dab8a9c0da78196de3d195781c7bbab0c4843ceb644d08bd33b5c` (`1.json`)
- Classification: recorded, upstream real responses
- Transformation: byte-identical extraction of both `response.body` values. The recorded bodies
  begin with several blank/whitespace lines from the gateway; these are kept as recorded.
- Why it is necessary: the tool-call id `functions.weather:0` is a gateway-shaped id, unlike
  OpenAI's `call_…`. It proves Republic neither rewrites nor rejects foreign ids, and that the id
  survives into the outgoing `tool_call_id`. Kimi's thinking arrives in `reasoning`, the field
  Republic reads for gateways.
- Limitations: Kimi is reached through OpenRouter here; the direct Moonshot provider differs only in
  base URL and dialect hooks, which the shared corpus already covers. The `provider` field of the
  recording is whichever upstream OpenRouter routed to, so it is not a stable Moonshot fact.

### `provider_sample_openrouter_grok_streaming/0.sse`

- Case: `TestOpenRouterCommon/grok/simple_streaming.yaml`
- Upstream file SHA-256: `4f22a8f7c52af1223f70b2f99ef7ee090576c731a35025ceef9d669d59c0d047`
- Local SHA-256: `3f9e271722ffa27d7a65d58f4e402680b0dfb15449a702defd037567f743343b`
- Classification: recorded, upstream real SSE stream
- Transformation: the YAML block-scalar `response.body` is written out as bytes unchanged,
  including the three `: OPENROUTER PROCESSING` comment lines and the trailing `data: [DONE]`.
- Why it is necessary: OpenRouter streams `reasoning` incrementally with no `reasoning_content`,
  sends usage in a final chunk after the finish reason, and emits SSE comments as keepalives. It is
  the streaming counterpart to the Kimi case and the only sample here with gateway keepalives.
- Limitations: xAI is the upstream, so it does not distinguish OpenRouter from xAI beyond the
  gateway envelope. The recorded request used `"usage":{"include":true}` (an OpenRouter body
  extension); Republic sends `stream_options.include_usage`. The response recording alone does
  not validate that request option.

### `provider_sample_azure_tool/0.json`, `1.json`

- Case: `TestAzureCommon/azure-gpt-5-mini/tool.yaml`
- Upstream file SHA-256: `4848812cfd263a7ea3a32d0cd1ab3c8dda0f8137566a85ca2a4ff5a9b29789de`
- Local SHA-256: `793f9d33718111aecacaf0ac23441a448c9b52cd89fbd899b51b518c2d2522ae` (`0.json`),
  `7c6d0e0708e0e4d8885fea90d3041e224d2553d3ab0786ed1e594c5373a71251` (`1.json`)
- Classification: recorded, upstream real responses
- Transformation: byte-identical extraction of both `response.body` values. The recorded host
  `fantasy-playground-resource.openai.azure.com` is kept inside the body only insofar as it appears
  there; the test's `api_base` uses the same resource name so the URL is not invented.
- Why it is necessary: Azure's v1 API answers with extra `content_filter_results` and
  `prompt_filter_results` blocks, names the model `gpt-5-mini-2025-08-07`, and authenticates with
  `api-key`. It is the only sample that exercises the Azure auth header and deployment naming.
- Limitations: the recording is chat, not the Responses API. It contains no `api-key` header of its
  own — the VCR matches only method/URL/body — so the header assertion comes from Republic's
  `AzureOpenAI._api_key_auth`, not from the recording. Independent evidence is the pinned OpenAI
  SDK `src/openai/lib/azure.py` at `59738a289ee82c7b86bf1e10230cceb6daa9716d`, SHA-256
  `002f5e8b264e0b28534855f1e0245e5d2b8591a1e696bc872de4e9a98508322a`. It supports
  both API-key and Azure AD bearer modes; only API-key mode is sampled here.

## Families not sampled here

The following supported families have no case in the recorded corpus. In short: `deepseek`,
`moonshot` (direct), `ollama`
and `together` reuse the shared chat mechanics already covered by `test_compatible_providers.py`
and `test_provider_golden.py`, and adding a fifth near-identical body would not test a new
distinction. `minimax`'s `reasoning_split` request field and default Messages routing are covered by
existing tests; no pinned recording of its `reasoning_details` reply was found in the sources above.

## Reproducing

```console
uv run pytest tests/test_provider_golden.py -q
```

No network access, credentials, or model calls are needed; `FakeService` answers from these files.
