import time
from typing import List
import concurrent.futures
from tqdm import tqdm  # noqa: F401 (kept for parity)

import openai
from openai import OpenAI
import anthropic

def get_llm_output(
    prompt: str | List[str], model: str, cache: bool = True, system_prompt=None, history: list = [], max_tokens: int = 256
) -> str | List[str]:
    if isinstance(prompt, list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(
                    get_llm_output, p, model, cache, system_prompt, history, max_tokens
                )
                for p in prompt
            ]
            concurrent.futures.wait(futures)
            return [future.result() for future in futures]

    openai.api_base = (
        "https://api.openai.com/v1" if model != "llama-3-8b" else "http://localhost:8000/v1"
    )
    provider = "openai"
    if "gpt" in model:
        client = OpenAI()
    elif model == "llama-3-8b":
        client = OpenAI(base_url="http://localhost:8000/v1")
    else:
        provider = "anthropic"
        client = anthropic.Anthropic()

    try:
        api_key = getattr(client, "api_key", None)
    except Exception:
        api_key = None

    systems_prompt = "You are a helpful assistant." if not system_prompt else system_prompt

    if "gpt" in model:
        messages = (
            [{"role": "system", "content": systems_prompt}] + history + [{"role": "user", "content": prompt}]
        )
    elif "claude" in model:
        messages = history + [{"role": "user", "content": prompt}]
    else:
        messages = (
            [{"role": "system", "content": systems_prompt}] + history + [{"role": "user", "content": prompt}]
            )

    for _ in range(3):
        try:
            if "gpt-3.5" in model:
                client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                response = completion.choices[0].message.content.strip()
            elif "gpt-4" in model:
                completion = client.chat.completions.create(model=model, messages=messages)
                response = completion.choices[0].message.content.strip()
            elif "claude-opus" in model:
                completion = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    system=systems_prompt,
                )
                response = completion.content[0].text
            elif "claude" in model:
                completion = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                response = completion.content[0].text
            elif model == "llama-3-8b":
                completion = client.chat.completions.create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_body={"stop_token_ids": [128009]},
                )
                response = completion.choices[0].message.content.strip().replace("<|eot_id|>", "")
            else:
                completion = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
                response = completion.choices[0].message.content.strip()

            return response
        except Exception as e:
            print(f"LLM Error: {e}")
            time.sleep(5)
            if "Error code: 400" in str(e):
                messages = (
                    [{"role": "system", "content": systems_prompt}] + history + [{"role": "user", "content": str(prompt)[: len(str(prompt)) // 2]}]
                )
            else:
                raise
    return "LLM Error: Cannot get response."