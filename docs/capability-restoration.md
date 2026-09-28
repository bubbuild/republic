# Capability inventory and restoration

The chat-only rebuild omitted independent model capabilities and broke Bub media
input. This inventory corrects that scope; existing auth/caller-policy and agent
boundaries remain unchanged. It does not claim live service acceptance.

| Capability | Source / prior behavior | Delivery |
| --- | --- | --- |
| Text embeddings | Republic `216098ef` clients/embedding.py, LLM.embed/embed_async; ai-python ops/embeddings.py | Independent request/result/protocol and real OpenAI-compatible `/embeddings`; no LLM facade or hidden batching |
| Document reranking | ai-python ops/reranking.py and AI Gateway protocol/v4.py `/reranking-model`; **not a dedicated old Republic API** | Independent real adapter; provider choice awaiting the caller's Cohere/Gateway decision |
| Message images/audio/video/files | ai-python OpenAI/Anthropic converters; Bub build_prompt emits image_url/input_audio/video_url | Restore FilePart wire conversion and Bub hook → runner → durable tape path, with protocol-specific media support |
| Image generation/editing | ai-python ops/images.py and provider implementations | Next independent increment: explicit image request/results and actual image endpoint, bytes/URL output, no automatic fetch |
| Speech generation | ai-python ops/audio.py | Next independent increment: speech request/result and real audio endpoint, caller-selected encoding/voice |
| Transcription | ai-python ops/transcriptions.py | Next independent increment: supplied bytes/file data → actual transcription endpoint, metadata/usage, no automatic download |
| Video generation | ai-python ops/videos.py | Separate asynchronous job/result capability; explicit polling/ownership policy, distinct from video messages |
| Evaluation | ai-python ops/evaluation.py | Separate scope review; no evaluation workflow or agent runtime implicitly added |

Reference snapshot: Vercel `ai-python` commit
`c788059dd1db2d93ae1c3da6daffb660eca07dbb` (Apache-2.0). The old Republic embedding
helper delegated to any-llm and had sync/async methods. The new API is async and
independent of conversation providers. Reranking scores are never synthesized
with a chat prompt. Vector stores, RAG orchestration and tool execution stay with
consumers. Caller-supplied URLs/bytes/base64 are data; adapters neither download
URLs nor read arbitrary local files.

The increments are embedding, rerank, SDK media inputs, then actual Bub media
consumption. Each receives its own semantic local commit and HTTP fixtures;
clean-wheel consumer validation follows the committed SDK changes.
