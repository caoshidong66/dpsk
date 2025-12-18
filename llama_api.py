import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


def _default_model_dir() -> Path:
    """
    默认的本地 LLaMA3-8B 模型路径。

    在你的服务器上，cwd 是 /data/jsg_data/dpsk 时，
    /data/jsg_data/model/meta-llama/llama3-8b 就是你的 LLaMA3-8B 模型目录。
    """
    return Path("/data/jsg_data/model/meta-llama/llama3-8b")


@lru_cache(maxsize=1)
def _load_tokenizer_and_model(model_dir: Optional[str | Path] = None):
    """
    懒加载本地 LLaMA 模型，只在第一次调用时从磁盘加载。
    """
    model_path = Path(
        model_dir
        or os.environ.get("LLAMA_MODEL_DIR", "")
        or _default_model_dir()
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Local LLaMA model not found at: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def llama_completion(
    prompt: str,
    model_dir: Optional[str | Path] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    top_p: float = 0.95,
    generation_prefix: Optional[str] = "Step 1:",
    debug_print_full_prompt: bool = False,
) -> str:
    """
    使用本地 LLaMA3-8B 做一次 completion。

    输入:
      - prompt: 前缀字符串
      - model_dir: 模型目录，默认为 /data/jsg_data/model/meta-llama/llama3-8b 或环境变量 LLAMA_MODEL_DIR

    返回:
      - completion: 只包含新生成的文本（不含 prompt 部分）
    """
    tokenizer, model = _load_tokenizer_and_model(model_dir)

    system_prompt = (
        "You are an expert math problem solver. Follow the user's instructions carefully, "
        "show step-by-step reasoning,"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if generation_prefix:
        if not chat_prompt.rstrip().endswith(generation_prefix):
            chat_prompt += generation_prefix

    if debug_print_full_prompt or os.environ.get("LLAMA_PRINT_FULL_PROMPT") == "1":
        print("\n" + "=" * 40)
        print("[llama_api] Full prompt sent to model:")
        print(chat_prompt)
        print("=" * 40 + "\n")

    inputs = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的部分
    generated_ids = output_ids[0, input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text


if __name__ == "__main__":
    demo_prompt = "You are a helpful math assistant.\n\nQuestion: What is 1+1?\n\nAnswer:"
    print(llama_completion(demo_prompt))


"""
12.13目前在做Ablation on LLAMA的实验1 时序部分 主实验Qwen4B
12.14同步完成Ablation on LLAMA的实验3
12.16完成Qwen4B最后的MATH4/5
———————————上面就完成了LLAMA+Qwen4B两个一直在做的主实验——————————————————
12.17开始做Ablation on Qwen最后的prompt实验1
12.15-12.20完成Qwen7B(优先级较低，但是这个之前跑的少,得多跑)
12.20-12.22 Ablation on Qwen的实验1/2/3
———————————上面就完成所有的实验—————————————————
12.22-1.01
可以选择补充的其他实验
1.加大数据量
2.混乱推理
"""