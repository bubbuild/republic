# Azure OpenAI

Use the `azure-openai` provider to call Azure OpenAI deployments through the v1 API. It supports chat and embeddings. Chat requests default to Responses; Chat Completions is also available.

## Get started

Install Republic using the [quickstart](../quickstart.md#install-republic), set `REPUBLIC_AZURE_OPENAI_API_KEY`, set `REPUBLIC_AZURE_OPENAI_RESOURCE` to your resource name, and replace `DEPLOYMENT` with a deployment name:

```python
import asyncio

import republic


async def main():
    model = republic.get_model("azure-openai:DEPLOYMENT")
    response = await model.chat("Say hello in one sentence.")
    print(response.text)


asyncio.run(main())
```

## Authentication

Republic reads `REPUBLIC_AZURE_OPENAI_API_KEY` when the provider is created and sends the key in the `api-key` header. You can pass a key loaded by your application as `api_key=`; see [API-key authentication](../guides/authentication.md#use-an-api-key). For Microsoft Entra ID, pass an `auth=` object that sends `Authorization: Bearer` with a token for the `https://ai.azure.com/.default` scope.

## Model support

The model name is the deployment name. Republic builds the endpoint `https://RESOURCE.openai.azure.com/openai/v1` from the resource name, read from `resource=` on `AzureOpenAI` or `REPUBLIC_AZURE_OPENAI_RESOURCE`. A full `api_base=` or `REPUBLIC_AZURE_OPENAI_API_BASE`, such as a `services.ai.azure.com` endpoint, takes precedence. Creating the provider without a resource or base URL raises `ValueError`.

```python
from republic.providers import AzureOpenAI

model = AzureOpenAI(resource="my-resource").get_model("DEPLOYMENT")
```

Select `api_format="chat"` for deployments without Responses support. Use `get_embedding_model()` for embedding deployments.

See [configuration](../reference/configuration.md) for endpoint overrides and request options.
