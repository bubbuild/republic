# Message media inputs

Use `FilePart` in the original message order. URL, base64 and explicit provider
file references are JSON data; Republic never opens local paths, fetches URLs or
uploads files on the caller's behalf. `FilePart.from_bytes` encodes supplied bytes
for persistence. Service/model media availability still requires live validation.

```python
from republic import FilePart, Message, TextPart

message = Message(role="user", parts=[
    TextPart(text="Describe these inputs"),
    FilePart(data="https://example.test/image.png", media_type="image/png"),
    FilePart.from_bytes(b"caller-supplied-audio", media_type="audio/wav"),
])
restored = Message.model_validate_json(message.model_dump_json())
assert restored == message
```

The example illustrates data construction, not a valid audio recording. Supply
actual encoded media when making a real request.

| Adapter | Supported user input wire | Limits |
| --- | --- | --- |
| OpenAI Chat | Images via image_url; audio via input_audio; PDF via file; text files decoded as text | Audio/PDF need inline base64 (including matching data URLs); no automatic URL fetch. PDF accepts file_id. Native OpenAI audio is mp3/wav. |
| OpenAI-compatible Chat / OpenRouter | The same parts; video_url with HTTP(S)/data URL; provider-supported audio formats | Video and extra audio formats are compatible-service extensions, not native OpenAI guarantees. Choose an explicit compatible endpoint/model. |
| OpenAI Responses | input_image URL/base64/file_id; input_file PDF URL/base64/file_id; inline UTF-8 text files | No audio/video wire in this standard adapter. Other independent media operations are separate increments. |
| Anthropic Messages | image URL/base64/file_id; document PDF URL/base64/file_id; text/plain document | JPEG/PNG/GIF/WebP; no audio/video. File IDs must already belong to the service; Files API beta/version headers remain caller configuration where required. |
| ChatGPT Codex | Native Responses input_image URL/base64/file_id | The sourced Codex subset establishes images, not PDF documents. Audio/video and PDF remain unsupported here. |
| Copilot / Grok OAuth | No restored media claim | Their specific media wire/account acceptance remains unverified; these adapters still reject media. |

For a reference use `FilePart(data="file-id", encoding="file_id", media_type=...)`.
IDs and native metadata are protocol-specific; changing the endpoint/provider does
not migrate the asset or guarantee access to it.

Chat images and Responses images/PDF accept native `openai.detail` metadata.
Chat audio format defaults from the MIME subtype (`audio/mpeg` maps to `mp3`);
`provider_metadata={"openai": {"format": "ogg"}}` explicitly overrides it for
compatible services. Video uses `video_url`; optional `openai.processing` selects
its documented `agentic` or `static` mode. No synthetic content-type flag is needed.

Anthropic media accepts `anthropic.cache_control`; documents also accept native
`title`, `context` and `citations` configuration. Generated citation blocks/deltas
remain outside the output parser, so requesting citations may produce an explicit
unsupported-output error. Filename is a Chat/Responses inline PDF field; it is
not silently treated as an Anthropic title. Unrepresentable filenames/metadata
are rejected. Plain text files follow the upstream UTF-8 conversion; supplied
text-file URLs in Chat/Responses are text, not downloaded file contents.

OpenRouter `reasoning_details` is retained in assistant message metadata,
including encrypted video context. Stream fragments with an index or ID aggregate
text/summary/data/signature in arrival order, retaining constant identity fields.
Conflicting identity fails. Unidentified complete records remain separate;
fragment identity is not invented. JSON round-trip and subsequent requests retain
these values. Existing Responses encrypted items and Anthropic thinking signatures
remain unchanged. Generated image/audio output is not covered by this input API.

## Sources and evidence

Conversion is trimmed from ai-python
`c788059dd1db2d93ae1c3da6daffb660eca07dbb` OpenAI/Anthropic protocol functions,
without their downloads or history repair. File references and compatible video
are checked against official sources (2026-09-28):

- [OpenAI vision](https://developers.openai.com/api/docs/guides/images-vision) and [file inputs](https://developers.openai.com/api/docs/guides/file-inputs).
- [Anthropic vision](https://platform.claude.com/docs/en/build-with-claude/vision) and [PDFs](https://platform.claude.com/docs/en/build-with-claude/pdf-support).
- [OpenRouter audio](https://openrouter.ai/docs/guides/overview/multimodal/audio), [video](https://openrouter.ai/docs/guides/overview/multimodal/videos) and [reasoning details](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
- [Codex ContentItem/ImageReference](https://github.com/openai/codex/blob/21eb35513df478a2a090bfc2c0293caaf435b36d/codex-rs/protocol/src/models.rs).

Real official SDK HTTP/SSE fixtures verify ordered payloads, serialization, native
history replay and rejected inputs. They do not establish live model/account
support or decode the supplied synthetic media bytes. See the separate
[Bub integration evidence](bub-integration.md) for consumer acceptance.
