import torch
from typing import Optional, Tuple, List, Dict, Any
import re
import argparse

try:
    from qwen_vl_utils import process_vision_info as _hf_process_vision_info
except ImportError:  # pragma: no cover - optional dependency
    _hf_process_vision_info = None


def get_model_device(model) -> torch.device:
    """Return the device of the first parameter, defaulting to CPU."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_assistant_response(text: str, eos_token: Optional[str] = None) -> str:
    """Trim a decoded chat transcript down to the final assistant turn."""
    if not text:
        return ""

    if eos_token:
        text = text.replace(eos_token, " ")
    text = text.replace("</s>", " ")

    markers = [
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|assistant|>",
        "\nassistant\n",
        "assistant\n",
        "Assistant:\n",
    ]

    for marker in markers:
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()

    return text.strip()


def process_vision_payload(messages: List[Dict[str, Any]]) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """Use the official Qwen helper when available, otherwise fall back to bare data."""
    if _hf_process_vision_info is not None:
        return _hf_process_vision_info(messages)

    image_inputs: List[Any] = []
    video_inputs: List[Any] = []
    for message in messages:
        for item in message.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                image_inputs.append(item.get("image"))
            elif item.get("type") == "video":
                video_inputs.append(item.get("video"))

    return (image_inputs or None), (video_inputs or None)


def print_generation_inputs(image, inputs, saved_input_ids, processor):
    """
    Print image info and input tokens before generation (in red color).
    
    Args:
        image: Image object or path string
        inputs: Model inputs dictionary
        saved_input_ids: Saved input_ids tensor (can be None)
        processor: Processor object with tokenizer
    """
    print(f"\033[91m{'='*80}\033[0m")
    print(f"\033[91m[Before Generation] Image Info:\033[0m")
    if image is not None:
        if isinstance(image, str):
            print(f"\033[91m  Image type: str (file path)\033[0m")
            print(f"\033[91m  Image path: {image}\033[0m")

    # Print pixel_values and other image-related inputs
    if 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values']
        print(f"\033[91m  pixel_values shape: {pixel_values.shape}\033[0m")
        print(f"\033[91m  pixel_values dtype: {pixel_values.dtype}\033[0m")
        print(f"\033[91m  pixel_values device: {pixel_values.device}\033[0m")
        print(f"\033[91m  pixel_values min: {pixel_values.min().item():.4f}, max: {pixel_values.max().item():.4f}, mean: {pixel_values.mean().item():.4f}\033[0m")
    else:
        print(f"\033[91m  pixel_values: Not in inputs\033[0m")

    if 'image_grid_thw' in inputs:
        image_grid_thw = inputs['image_grid_thw']
        print(f"\033[91m  image_grid_thw: {image_grid_thw}\033[0m")
    else:
        print(f"\033[91m  image_grid_thw: Not in inputs\033[0m")

    print(f"\033[91m[Before Generation] Input Tokens Info:\033[0m")
    if saved_input_ids is not None:
        input_tokens = saved_input_ids[0].cpu().tolist()
        print(f"\033[91m  Input token count: {len(input_tokens)}\033[0m")
        print(f"\033[91m  First 10 tokens: {input_tokens[:10]}\033[0m")
        print(f"\033[91m  Last 10 tokens: {input_tokens[-10:]}\033[0m")
        # Decode tokens to text for readability
        try:
            decoded_text = processor.tokenizer.decode(input_tokens, skip_special_tokens=False)
            decoded_text = decoded_text.replace('<|image_pad|>', '')
            print(f"\033[91m  Decoded text:\n{decoded_text}\033[0m")
        except Exception as e:
            print(f"\033[91m  Failed to decode tokens: {e}\033[0m")
    else:
        print(f"\033[91m  Input tokens: Not available (using inputs_embeds)\033[0m")
        if 'inputs_embeds' in inputs:
            embeds_shape = inputs['inputs_embeds'].shape
            print(f"\033[91m  Input embeddings shape: {embeds_shape}\033[0m")
    print(f"\033[91m{'='*80}\033[0m")


def extract_answer(text: str) -> str:
    """
    Extract the FINAL CHOICE from a solution.
    Priority:
      1) <answer>...</answer> tag
      2) Last occurrence inside \boxed{...} (supports nested braces, only extracts if braces are balanced)
      3) Return empty string if neither found
    Cleans common LaTeX wrappers like \text{...}, \displaystyle, and surrounding $ ... $.
    """
    try:
        s = text

        # -------- 1) First try: <answer>...</answer> --------
        low = s.lower()
        start = low.find("<answer>")
        end = low.find("</answer>")
        if start != -1 and end != -1 and end > start:
            ans = s[start + len("<answer>"):end].strip()
            ans = ans.strip('$')
            ans = re.sub(r'\\displaystyle\s*', '', ans)
            ans = re.sub(r'\s+', ' ', ans).strip()
            if ans:
                return ans

        # -------- 2) Second try: Gather all contents inside \boxed{...} with balanced brace checking --------
        boxed_contents = []

        # find all occurrences of "\boxed" followed by optional spaces then "{"
        for m in re.finditer(r'\\boxed\s*\{', s):
            # position of the opening brace "{"
            open_brace_pos = s.find('{', m.end() - 1)
            if open_brace_pos == -1:
                continue

            # Brace matching to find the corresponding closing brace
            # Only extracts if braces are properly balanced (not incomplete like \boxed{\frac{16})
            depth = 0
            i = open_brace_pos

            while i < len(s):
                ch = s[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        # Found matching closing brace - extract inside { ... }
                        boxed = s[open_brace_pos + 1:i]

                        # light cleaning: strip spaces and surrounding $ ... $
                        boxed = boxed.strip()
                        boxed = boxed.strip('$')

                        # remove \displaystyle and similar display-style macros at the front
                        boxed = re.sub(r'\\displaystyle\s*', '', boxed)

                        # unwrap \text{...} if the whole thing is a single \text{...}
                        def unwrap_text_env(x: str) -> str:
                            x_strip = x.strip()
                            if x_strip.startswith(r'\text{') and x_strip.endswith('}'):
                                # naive unwrap of a single-level \text{...}
                                inner = x_strip[len(r'\text{'):-1].strip()
                                return inner
                            return x
                        boxed = unwrap_text_env(boxed)

                        # collapse whitespace
                        boxed = re.sub(r'\s+', ' ', boxed).strip()

                        if boxed:
                            boxed_contents.append(boxed)
                        break
                i += 1

        # If we found any \boxed content with balanced braces, return the last one (FINAL CHOICE)
        if boxed_contents:
            return boxed_contents[-1]

    except Exception:
        # fall through to empty string fallback
        pass

    # -------- 3) Fallback: return the original text ----------
    return text.strip()  # fix


def args_to_dict(args):
    """
    Convert argparse.Namespace to a JSON-serializable dictionary.
    
    Args:
        args: argparse.Namespace object
        
    Returns:
        Dictionary with all args, with non-serializable values converted
    """
    result = {}
    for key, value in vars(args).items():
        # Handle different types
        if value is None:
            result[key] = None
        elif isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, torch.Tensor):
            # Convert tensor to list or scalar
            if value.numel() == 1:
                result[key] = value.item()
            else:
                result[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Recursively convert list/tuple items
            result[key] = [
                item.item() if isinstance(item, torch.Tensor) and item.numel() == 1
                else item.tolist() if isinstance(item, torch.Tensor)
                else item
                for item in value
            ]
        elif isinstance(value, dict):
            # Recursively convert dict values
            result[key] = {
                k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1
                else v.tolist() if isinstance(v, torch.Tensor)
                else v
                for k, v in value.items()
            }
        else:
            # For other types, try to convert to string
            try:
                result[key] = str(value)
            except Exception:
                result[key] = None
    return result


def extract_true_answer(text, name="gsm8k"):
    '''
    Extract answer from text.
    Since we only load from JSON files now, the answer is already in the correct format.

    Args:
        text: input text (answer from JSON file)
        name: name of the dataset or file path (kept for compatibility, not used)

    Returns:
        answer: extracted answer (returns text as-is for JSON files)
    '''
    # For JSON files, the answer is already in the correct format
    return text


def judge_answer(input, label, data_name="gsm8k", extract=True, prompt_idx=0):
    """Score.

    Judge whether the answer is correct or not.
    Since we only load from JSON files now, we use a unified judgment logic.

    Args:
        input (str): model response
        label (str): ground truth
        data_name (str): name of the dataset or file path (kept for compatibility, not used)
        extract (bool): whether to extract answer from model response
        prompt_idx (int): index of the solver prompt (kept for compatibility, not used)

    Returns:
        bool: True if the answer is correct, False otherwise
    """
    # Extract answer from model response if needed
    if extract:
        input = extract_answer(input)

    # Convert to strings for comparison
    input_str = str(input).strip()
    label_str = str(label).strip()

    # Try multiple verification methods
    # 1. Exact match (case-sensitive)
    if input_str == label_str:
        return True

    # 2. Exact match (case-insensitive for multi-choice questions)
    if input_str.upper() == label_str.upper():
        return True

    # 3. Check if answer appears in input (for cases where model includes extra text)
    if label_str and label_str in input_str:
        return True

    return False
