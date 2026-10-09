# Why Republic stops at providers

An application can need a model call without needing an agent runtime. It may already have functions for its business rules, a database for its state, and a web framework for its requests. Connecting it to an AI service should not require replacing those choices.

Republic builds provider requests, authenticates them, and returns messages, events, and usage. Your application decides when to call a model, what data it can use, and what to do with the result. Gateways, agent loops, and application tool execution stay outside the package.

## A model call is useful on its own

A function can await `model.chat()` and use `response.text`. A streaming endpoint can iterate over text events. Neither needs an agent class, a workflow definition, or a new state store. The model object provides access to the service; the existing function still determines execution.

This is enough for many tasks. When an application needs more, it can compose those calls with code it already owns.

## An agent needs a feedback loop

Suppose a model must answer questions about a project. Give it a tool that reads project files. The first response can request a file, and a Python function can read it.

Executing the function is only half of the round trip. If the harness discards its output, the model never learns what the file contains. Append the assistant's message and the tool result, then make another request. Repeat until the model replies without calling a tool.

The [minimal agent](guides/minimal-agent.md) is this loop. The application chooses the available files, executes the function, and decides when to stop. Republic supplies the provider calls and message objects. A different harness could expose an editor operation or a database query without changing that division of responsibility.

## A message carries more than display text

A service can require an original call ID, a signed thinking block, or opaque reasoning state on the next request. Rebuilding a conversation from visible text loses that information.

`tool_result()` keeps the original call with its output. `response.message` preserves the assistant turn, including the provider data needed to continue it. The application can send these objects back without reconstructing wire fields.

Opaque `ProviderData` belongs to the format that produced it. Other formats skip it, so switching formats mid-conversation may lose service-specific state. A shared response shape does not make every piece of a conversation portable.

## Service differences belong at the boundary

Changing the model service introduces different URLs, credentials, request fields, and stream events. Republic keeps those concerns in a few interfaces:

| Interface | Responsibility |
| --- | --- |
| Provider | Service endpoint, credentials, HTTP configuration, and format selection |
| API format | Request encoding and response parsing for a wire protocol |
| Auth | Authenticate a request, including exchange or refresh when required |
| Model | Public methods appropriate to its kind: `chat()`, `stream()`, `embed()`, or `decide()` |

Several services can share a format. One service can expose several formats. Changing a provider can leave the agent loop in place; selecting `api_format=` changes the protocol used for the request. Each model still has its own capabilities and restrictions.

Authentication follows the same boundary. Republic uses the standard `httpx2.Auth` interface and Authlib's OAuth support. Provider-specific implementations can reuse a CLI login or refresh a credential while the application's model calls stay the same.

## The application keeps ownership

The application closes a supplied HTTP client, chooses a credential store, and decides how to persist conversation history. Explicit login helpers return auth objects; normal model requests do not start an interactive authorization flow. Optional history support records messages but does not schedule another call.

Provider-run tools, such as a service's web search, execute at the service and appear in the response. Application tools execute in the caller's code. Supporting both does not require Republic to become a local tool executor.

The small harness is the example of this design: model access from Republic, application behavior from ordinary Python.
