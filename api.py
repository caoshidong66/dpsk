import os
from volcenginesdkarkruntime import Ark

client = Ark(
    api_key=os.environ.get("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


def ark_chat_with_stop(
    client,
    model,
    messages,
    stop_token,
    thinking: str = "disabled",
    temperature: float = 0.0,
):
    """
    使用 Ark 原生 chat.completions 流式接口。

    - messages: 标准的 chat 消息列表
    - stop_token: 一旦在增量内容中出现该子串就截断
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        thinking={"type": thinking},
        max_tokens=4096,
        temperature=temperature,
        stream=True,
    )

    final = ""
    for chunk in stream:
        delta = chunk.choices[0].delta

        # delta.content 可能为 None，跳过
        if delta is None or delta.content is None:
            continue

        text = delta.content

        # Stop token detected
        if stop_token in text:
            final += text.split(stop_token)[0]
            break

        final += text

    return final


if __name__ == "__main__":
    # ---- 示例调用 ----
    output = ark_chat_with_stop(
        client=client,
        model="deepseek-v3-2-251201",   # Ark 最新 DeepSeek V3 模型
        messages=[{"role": "user", "content": "Hello"}],
        stop_token="can",
    )
    print(output)
