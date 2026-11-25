# `bf16ai` — Official Python SDK for [bf16.ai](https://bf16.ai)

Zero-dependency Python SDK (`>=3.9`) for **[bf16.ai](https://bf16.ai)** — uncensored 1M-context open-weight LLM inference API (DeepSeek V4.1 Flash, GLM-5.3, GLM-5.3 Flash, Kimi K3.1 Pro) with automatic prompt caching and strict Zero Data Retention (ZDR).

## Install

```bash
pip install bf16ai
```

## Quickstart

```python
import bf16ai

client = bf16ai.Client()  # Reads BF16_API_KEY from env

# Check live balance, tier & concurrency via GET /v1/me
me = client.me()
print(f"Balance: ${me['balance']['available_usd']:.4f} ({me['tier']['name']})")

# Run a 1M-context chat completion
res = client.chat(
    [{"role": "user", "content": "Give 3 practical tips for prompt-cache hit rates in coding agents."}],
    model=bf16ai.MODEL_DEEPSEEK_V41_FLASH_FAST,
)
print(res["choices"][0]["message"]["content"])
```

## Streaming

```python
for chunk in client.chat_stream(
    [{"role": "user", "content": "Audit this FastAPI dependency for auth bypass risks."}],
    model=bf16ai.MODEL_GLM_53,
):
    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
    print(delta, end="", flush=True)
```

## Official Links

- Website & Console (`$0.20` starter credit): [https://bf16.ai](https://bf16.ai)
- API Documentation: [https://bf16.ai/docs](https://bf16.ai/docs)
- Live Model Catalog & Pricing: [https://bf16.ai/models](https://bf16.ai/models)
- Coding Agent Cost Calculator: [https://bf16.ai/tools/cost-calculator](https://bf16.ai/tools/cost-calculator)
- Open-Source Inference Grants (`$50` API credits): [https://bf16.ai/open-source](https://bf16.ai/open-source)
- Official CLI (`npx bf16ai claude`): [https://github.com/BF16-AI/bf16](https://github.com/BF16-AI/bf16)

## License

MIT © BF16 Team
