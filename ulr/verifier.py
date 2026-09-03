import logging
import os
import re
import time
from typing import Optional

from openai import OpenAI


DEFAULT_MAX_RETRIES = 3
DEFAULT_SLEEP_TIME = 0.2

_client: Optional[OpenAI] = None
_model_name: Optional[str] = None


def _load_dotenv_into_environ() -> None:
    cwd = os.path.dirname(os.path.abspath(__file__))
    current = cwd
    visited = set()
    while True:
        if current in visited:
            break
        visited.add(current)
        env_path = os.path.join(current, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            except Exception as exc:
                logging.warning("Failed to load .env from %s: %s", env_path, exc)
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def _get_model_name() -> str:
    global _model_name
    if _model_name is not None:
        return _model_name

    _load_dotenv_into_environ()
    _model_name = (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL_TYPE")
        or "gpt-4o-2024-08-06"
    )
    return _model_name


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    _load_dotenv_into_environ()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
    )

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    _client = OpenAI(**kwargs)
    return _client


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = str(answer).strip()

    boxed_match = re.search(r"\\boxed\{([^}]*)\}", answer)
    if boxed_match:
        answer = boxed_match.group(1).strip()

    answer_tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", answer, re.IGNORECASE | re.DOTALL)
    if answer_tag_match:
        answer = answer_tag_match.group(1).strip()
    else:
        open_tag_match = re.search(r"<answer>\s*(.*)", answer, re.IGNORECASE | re.DOTALL)
        if open_tag_match:
            answer = open_tag_match.group(1).strip()

    think_match = re.search(r"<think>.*?</think>\s*(.*)", answer, re.IGNORECASE | re.DOTALL)
    if think_match:
        answer = think_match.group(1).strip()
    elif re.match(r"<think>", answer, re.IGNORECASE):
        last_line = answer.rstrip().rsplit("\n", 1)[-1].strip()
        if last_line and not last_line.lower().startswith("<think"):
            answer = last_line

    return answer.strip().lower()


def simple_match(answer: str, ground_truth: str) -> bool:
    norm_answer = normalize_answer(answer)
    norm_gt = normalize_answer(ground_truth)

    if not norm_answer or not norm_gt:
        return False

    if norm_answer == norm_gt:
        return True

    yes_variants = {"yes", "y", "true", "correct", "1"}
    no_variants = {"no", "n", "false", "incorrect", "0"}

    if norm_answer in yes_variants and norm_gt in yes_variants:
        return True
    if norm_answer in no_variants and norm_gt in no_variants:
        return True

    return False


def _normalize_content(s: str) -> str:
    """Strip LaTeX wrappers / punctuation noise for content comparison."""
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def simple_option_match(answer: str, ground_truth: str) -> bool:
    if not answer or not ground_truth:
        return False

    ground_truth = str(ground_truth).strip()
    # MMMU-Pro options go up to J/K, not just A-D.
    gt_match = re.match(r"^[\(]?([A-Za-z])[\)]?[\s.:：]+(.+)$", ground_truth, re.DOTALL)
    if not gt_match:
        return False

    gt_option_letter = gt_match.group(1).lower()
    gt_option_content = _normalize_content(gt_match.group(2))

    candidates = [str(answer).strip(), normalize_answer(answer)]
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        cand_lower = cand.lower()
        cand_norm = _normalize_content(cand)

        option_only = re.match(r"^\(?([A-Za-z])\)?$", cand_lower)
        if option_only:
            return option_only.group(1) == gt_option_letter

        option_with_content = re.match(r"^\(?([A-Za-z])\)?[\s.:：]+", cand, re.IGNORECASE)
        if option_with_content:
            return option_with_content.group(1).lower() == gt_option_letter

        # Content match after normalization (LaTeX + whitespace tolerant).
        if cand_norm and cand_norm == gt_option_content:
            return True

    return False

#
def create_judge_prompt(answer: str, ground_truth: str, question: str = None) -> str:
    if question:
        return f"""You are an answer equivalence judge for multiple choice questions. Determine if the candidate answer matches the ground truth answer.

Question (with options):
{question}

Ground Truth: {ground_truth}

Candidate Answer: {answer}

Rules (in priority order):
1. FIRST check option letters: If candidate answer contains an option letter (A/B/C/D or a/b/c/d), it MUST match the ground truth's option letter.
   - Example: Ground truth "(a) A historical site", answer "(c)" -> FALSE (c != a)
   - Example: Ground truth "(a) A historical site", answer "(a)" -> TRUE (a = a)

2. If candidate answer has NO option letter, compare the content:
   - Example: Ground truth "(d) Dawn", answer "Dawn" -> TRUE (content matches)
   - Example: Ground truth "(b) Grenada", answer "Grenada" -> TRUE (content matches)

3. Be STRICT about option letters: "(c)" answering "(a): content" is WRONG, even if content seems related.

Is the candidate answer correct? Respond with ONLY "True" or "False"."""

    return f"""You are a strict answer equivalence judge. Compare the candidate answer with the ground truth answer and determine if they express the same final result.

Rules:
1. Focus ONLY on the final answer/conclusion, not the reasoning.
2. Ignore formatting differences (e.g., "No" vs "\\boxed{{No}}" are equivalent).
3. For numeric answers, consider equivalent representations (e.g., "0.5" and "1/2" are equivalent).
4. For Yes/No questions, only check if both indicate the same response.
5. Be strict: if the answers are different, return False.

Candidate Answer: {answer}

Ground Truth: {ground_truth}

Are these two answers equivalent? Respond with ONLY "True" or "False"."""


def _call_with_retries(prompt: str, max_retries: int = DEFAULT_MAX_RETRIES, sleep_time: float = DEFAULT_SLEEP_TIME) -> str:
    client = _get_client()
    model = _get_model_name()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            time.sleep(sleep_time)
            return content.strip()
        except Exception as exc:
            logging.warning(
                "Verifier API error (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                logging.error("Verifier retries exhausted; defaulting to False.")
                return ""


def verify_solution_equivalence(solution: str, ground_truth: str, question: str = None) -> bool:
    """
    Return True if the provided solution is equivalent to ground truth, judged by an LLM verifier.
    The API credentials and model are read from local .env in OpenAI format.
    Supported keys: OPENAI_API_KEY, OPENAI_BASE_URL / OPENAI_API_BASE_URL, OPENAI_MODEL / MODEL_TYPE.
    """
    logging.info("\x1b[32mSolution: %s\x1b[0m", str(solution)[:10])
    logging.info("\x1b[32mGround truth: %s\x1b[0m", ground_truth)

    try:
        if solution in ("", None) or ground_truth in ("", None):
            return False

        if simple_match(solution, ground_truth):
            return True

        if question and simple_option_match(solution, ground_truth):
            return True

        prompt = create_judge_prompt(str(solution), str(ground_truth), question)
        content = _call_with_retries(prompt)
        text = content.strip().lower()

        if "true" in text and "false" not in text:
            result = True
        elif "false" in text and "true" not in text:
            result = False
        else:
            logging.warning("Unexpected verifier response content: %r", content)
            result = False

        logging.info("\x1b[32mParsed response: %s\x1b[0m", result)
        return result
    except Exception as exc:
        logging.error("Error verifying solution equivalence: %s", exc)
        return False