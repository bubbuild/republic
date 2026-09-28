# Client configuration and caller policy

Providers accept `client`, `base_url`, `headers`, `timeout` and `max_retries`.
Owned clients default to zero SDK retries. Borrowed clients retain retries,
redirects, headers, query, organization/project and transport configuration on a
private SDK copy; Republic neither mutates nor closes the caller's client.
Explicit constructor values override borrowed settings, which override service
defaults. Request `provider_options["extra_headers"]` overrides constructor
headers. An OAuth access token supplies the bearer credential; an explicit
Authorization header can override it. Choose endpoints and redirect policy
appropriate for your credentials. With a borrowed client, its base URL is used
unless `base_url` is passed explicitly.

One generate/stream is one logical model operation, without login, refresh,
agent/tool execution or follow-up inference. Caller-selected SDK/transport retries
may make multiple HTTP attempts. Set `max_retries=0` and configure the HTTP
transport accordingly when a single HTTP attempt is required. Individual streams
release their response without closing a reusable client. After refresh, build a
new provider with the returned access token; lifecycle policy belongs to the caller.

OAuth providers accept either their simple token data or an existing access-token
string. Copilot expects an inference token, not an ordinary GitHub login token.
Refresh tokens and known expiry are not inference prerequisites. `is_expired()` is
an optional local hint, not an authorization decision; providers do not gate
requests on it. Refresh functions require a refresh token only when called.
Authentication helpers and optional explicit-path JSON helpers can be composed
independently. Republic does not require a particular credential file, login UX,
refresh schedule, fallback policy or storage format from a consumer.

Responses defaults request self-contained history: `store=False`, encrypted
reasoning included and (outside OAuth defaults) truncation disabled. Native
`store`, `include`, `truncation`, `previous_response_id` and `instructions` can be
selected by the caller. This forwards protocol configuration, not a guarantee of
server-side history availability. No conversation recovery or background polling
is implemented. `extra_body` extends native JSON fields and `extra_headers`
configures the request. Managed model/messages/input/tools/stream/common options
and duplicate native keys are rejected rather than silently overridden. Native
extensions are endpoint-dependent; unsupported output items/media still fail
explicitly. Choosing to omit encrypted reasoning may prevent history replay.
