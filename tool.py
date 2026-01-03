import re
from typing import Optional


def extract_model_answer(text: str) -> Optional[str]:
    """
    从模型输出中提取最终答案。

    约定模型在最后一行使用格式：
        Answer: <final_answer>
    """
    if not text:
        return None

    def _second_occurrence(haystack: str, needle: str) -> Optional[int]:
        first = haystack.find(needle)
        if first == -1:
            return None
        second = haystack.find(needle, first + len(needle))
        return None if second == -1 else second

    cut_positions = []
    for marker in (
        "\nProblem:",
        "\nYou are an expert math problem solver",
        "\nSolve the following math problem",
        "\nReasoning step by step:",
    ):
        pos = _second_occurrence(text, marker)
        if pos is not None:
            cut_positions.append(pos)
    if cut_positions:
        text = text[: min(cut_positions)]

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    # 从末尾往前找第一行以 "Answer:" 开头的
    for line in reversed(lines):
        if line.lower().startswith("answer:"):
            answer = line[len("answer:") :].strip()
            # Handle degenerate cases like "Answer: Answer: -6"
            while answer.lower().startswith("answer:"):
                answer = answer[len("answer:") :].strip()
            return answer or None
    return None


def extract_gt_answer(solution: str) -> Optional[str]:
    """
    从常见数学数据集的 solution/answer 字段中提取标准答案。

    支持的格式（按优先级）：
      - MATH / hendrycks_math: LaTeX \\boxed{...}
      - GSM8K: '#### <final_answer>'
      - 其他：回退为最后一行中可疑的答案片段（非常轻量）
    """
    if not solution:
        return None

    # 找最后一个 \boxed{...}
    matches = list(re.finditer(r"\\boxed\{([^}]*)\}", solution))
    if not matches:
        # GSM8K: "... #### 42"
        if "####" in solution:
            tail = solution.split("####")[-1].strip()
            return tail or None

        # Fallback: take last non-empty line and strip common wrappers
        lines = [ln.strip() for ln in solution.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        last = lines[-1]
        last = re.sub(r"^(answer\s*:)\s*", "", last, flags=re.IGNORECASE).strip()
        last = last.strip("$").rstrip(".")
        return last or None

    last = matches[-1]
    answer = last.group(1).strip()
    return answer or None


def _normalize(ans: str) -> str:
    """
    对答案做一个非常轻量的归一化，降低字符串比较的敏感度。

    不做复杂数学等价判断，只做：
      - 去掉首尾空格
      - 去掉首尾的美元符号和句号
      - 去掉内部空格
      - 转为小写
    """
    s = ans.strip()
    # 去掉外围的数学环境符号
    s = s.strip("$")
    # 去掉结尾的句号
    s = s.rstrip(".")
    # 去掉常见千分位逗号
    s = s.replace(",", "")
    # 去掉所有空格
    s = s.replace(" ", "")
    # 统一小写
    s = s.lower()
    try:
        if s:
            num = float(s)
            return format(num, ".15g")
    except ValueError:
        pass
    return s


def is_model_correct(model_output: str, solution: str) -> bool:
    """
    比较模型输出与数据集标准解答是否一致。

    返回:
      - True: 提取出的最终答案在轻量归一化后相同
      - False: 任一侧无法提取答案，或归一化后不相同
    """
    model_ans = extract_model_answer(model_output)
    gt_ans = extract_gt_answer(solution)

    if model_ans is None or gt_ans is None:
        return False

    return _normalize(model_ans) == _normalize(gt_ans)


def steps_for_level(level: Optional[object]) -> int:
    """
    根据题目的难度 level 决定需要多少个推理步骤。

    约定：
      - level 1 或 2 -> 4 步
      - level 3      -> 5 步
      - level 4      -> 6 步
      - level 5      -> 7 步

    输入 level 可以是：
      - int，例如 3
      - 字符串，例如 "3" 或 "Level 3"

    若无法解析，默认使用 5 步。
    """
    if level is None:
        return 5

    # 先尝试直接作为 int
    lvl_int: Optional[int] = None
    if isinstance(level, int):
        lvl_int = level
    elif isinstance(level, str):
        # 提取字符串中的第一个数字，如 "Level 3" -> 3
        m = re.search(r"(\d+)", level)
        if m:
            try:
                lvl_int = int(m.group(1))
            except ValueError:
                lvl_int = None

    if lvl_int is None:
        return 5

    if lvl_int in (1, 2):
        return 4
    if lvl_int == 3:
        return 5
    if lvl_int == 4:
        return 6
    if lvl_int == 5:
        return 7

    # 对于超出 1-5 的情况，退化为 5 步
    return 5


def steps_for_dataset(dataset_name: Optional[str]) -> int:
    """
    给没有 level 概念的数据集提供默认推理步数。
      - svamp: 3 steps
      - gsm8k: 5 steps
    其他/未知：5 steps
    """
    if not dataset_name:
        return 5
    name = dataset_name.strip().lower()
    if name == "svamp":
        return 3
    if name == "gsm8k":
        return 5
    return 5


if __name__ == "__main__":
    # 简单自测示例
    sample_output = "Step 1: ...\nAnswer: 2"
    sample_solution = (
        "The denominator factors as ... Therefore, the answer is \\boxed{2}."
    )
    print("Model answer:", extract_model_answer(sample_output))
    print("GT answer:", extract_gt_answer(sample_solution))
    print("Correct:", is_model_correct(sample_output, sample_solution))
