import requests

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

api_keys = [
  "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",
  "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",
  "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",
  "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",
  "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",
  "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",
  "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",
  "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",
  "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",
  "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"
]

MODEL = "google/gemma-4-31b-it"

def ask(question: str) -> str:
    """轮询所有 API key，返回第一个成功的响应内容。"""
    for key in api_keys:
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json"
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "stream": False,
        }
        try:
            resp = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                short_key = key[:20] + "..."
                print(f"[跳过] key={short_key} HTTP {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            short_key = key[:20] + "..."
            print(f"[跳过] key={short_key} 异常: {e}")

    raise RuntimeError("所有 API key 均不可用")


if __name__ == "__main__":
    question = input("请输入问题: ").strip()
    if not question:
        print("问题不能为空")
    else:
        print(f"\n正在调用 {MODEL} ...\n")
        answer = ask(question)
        print("回答:")
        print(answer)
