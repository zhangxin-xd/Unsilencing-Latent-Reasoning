import torch
import torch.nn as nn
import torch.nn.functional as F
from .reward import RewardModel
from .logger import log
from typing import Dict, Union, Optional, Tuple, List
from pathlib import Path
import numpy as np
import os
import csv
from PIL import Image
from .prompts import SYSTEM_PROMPT
from transformers.cache_utils import DynamicCache

try:
    from colorama import Fore, Style
    init_colorama = True
except ImportError:
    class Fore:
        BLUE = '\033[94m'
        GREEN = '\033[92m'

    class Style:
        RESET_ALL = '\033[0m'
    init_colorama = False


def get_stop_reason_vl(outputs, input_length, max_new_tokens, tokenizer):
    '''
    Determine the stop reason for generation
    
    Args:
        outputs: generated token ids (tensor or list)
        input_length: length of input tokens
        max_new_tokens: maximum number of new tokens
        tokenizer: tokenizer object
    
    Returns:
        stop_reason: string indicating why generation stopped
    '''
    if isinstance(outputs, torch.Tensor):
        generated_tokens = outputs[0] if outputs.dim() > 1 else outputs
    else:
        generated_tokens = outputs[0] if isinstance(outputs, list) and len(outputs) > 0 else outputs

    if isinstance(generated_tokens, torch.Tensor):
        generated_tokens = generated_tokens.tolist()

    generated_length = len(generated_tokens)
    new_tokens_length = generated_length - input_length

    # Check if reached max_new_tokens
    if new_tokens_length >= max_new_tokens:
        return "length"

    # Check if ended with eos_token
    if tokenizer.eos_token_id is not None:
        if generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id:
            return "eos_token"

    # Check if ended with pad_token
    if tokenizer.pad_token_id is not None:
        if generated_tokens and generated_tokens[-1] == tokenizer.pad_token_id:
            return "pad_token"

    # Default: other reason
    return "other"


def build_vl_inputs(
    processor,
    num_thought_tokens: int,
    question: str,
    image=None,
    messages=None,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    """
    Build inputs for VL model with thought tokens
    
    Args:
        processor: Qwen2VL processor
        num_thought_tokens: number of thought tokens
        question: question text
        image: PIL image or None
        messages: chat messages
        device: device to use
        data_name: dataset name
        model_name: model name
    
    Returns:
        inputs: processed inputs dict
        thought_idx: [start_idx, end_idx] of thought tokens
    """
    if num_thought_tokens <= 0:
        raise ValueError('You must specify a positive int for num_thought_tokens')

    # Use endoftext tokens for Qwen VL models
    latent_thought_tokens = '<|endoftext|>' * num_thought_tokens
    problem_type = 'code reasoning' if 'cruxeval' in data_name else 'math'

    # Detect if it's a multiple choice question
    is_multiple_choice = 'Choice' in question or 'choice' in question or any(x in question for x in ['\nA:', '\nB:', '\nC:', '\nD:'])

    # Adjust answer format instruction based on question type
    if is_multiple_choice:
        answer_instruction = (
            'IMPORTANT: This is a multiple choice question.\n'
            '- First, solve the problem step by step.\n'
            '- Then, provide your final answer as the option letter (A, B, C, or D) within \\boxed{{}}.\n'
            '- Example: \\boxed{{A}} or \\boxed{{B}}\n'
        )
    else:
        answer_instruction = (
            'IMPORTANT: You MUST always put your final numerical answer within \\boxed{{}}.\n'
        )

    # Build the input content with thought tokens
    input_content = (
        f'PROBLEM: {question}\n\n'
        f'The following special tokens represent YOUR INTERNAL THINKING SPACE where your reasoning happens implicitly.\n'
        f'You do NOT need to output explicit reasoning steps'
        f'After these tokens, directly provide your final answer.'
        f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
    )

    if image is not None and isinstance(image, str):
        lower = image.lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            if os.path.exists(image):
                try:
                    image = Image.open(image).convert("RGB")
                except Exception as exc:
                    raise ValueError(f"Failed to open image file: {image}: {exc}")
            else:
                raise ValueError(f"Image path does not exist: {image}")

    # Build messages
    if messages is None:
        # Check if SYSTEM_PROMPT exists and is not empty
        if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
            input_messages = [{'role': 'system', 'content': SYSTEM_PROMPT.strip()}]
        else:
            input_messages = []

        if image is not None:
            input_messages.append({
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': input_content}
                ]
            })
        else:
            input_messages.append({'role': 'user', 'content': input_content})
    else:
        # Use provided messages and append thought tokens to the last user message
        # Ensure messages is a list
        if isinstance(messages, dict):
            input_messages = [messages]
        else:
            input_messages = messages.copy() if isinstance(messages, list) else [messages]

        # Check if system prompt already exists
        has_system_prompt = any(
            isinstance(msg, dict) and msg.get('role') == 'system' for msg in input_messages
            if isinstance(msg, dict)
        )

        # Add SYSTEM_PROMPT if not present
        if not has_system_prompt and SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
            input_messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT.strip()})

        # IMPORTANT: Add actual image to messages if provided
        if image is not None and isinstance(input_messages[-1]['content'], list):
            # Replace placeholder {'type': 'image'} with actual image
            for i, content_item in enumerate(input_messages[-1]['content']):
                if isinstance(content_item, dict) and content_item.get('type') == 'image':
                    input_messages[-1]['content'][i] = {'type': 'image', 'image': image}
                    break

        if isinstance(input_messages[-1]['content'], list):
            input_messages[-1]['content'].append({'type': 'text', 'text': f"\n\n{input_content}"})
        else:
            input_messages[-1]['content'] = f"{input_messages[-1]['content']}\n\n{input_content}"

    # Apply chat template
    text = processor.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Process inputs with or without image
    if image is not None:
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(device)
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(device)

    # Calculate the start and end index of the latent thought tokens
    input_ids = inputs['input_ids'][0]
    question_positions = _find_question_positions(processor.tokenizer, input_ids, question)
    pure_input_length = len(input_ids)
    # For Qwen models, adjust the index
    input_thought_start_idx = pure_input_length - 1 - num_thought_tokens - 1
    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    thought_idx = [input_thought_start_idx, input_thought_end_idx]

    # Add generation prompt tokens
    gen_prompt = '<|im_start|>assistant\n'
    gen_prompt_ids = torch.tensor(
        processor.tokenizer.encode(gen_prompt, add_special_tokens=False),
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    # Append the generation_prompt tokens to inputs['input_ids']
    inputs['input_ids'] = torch.cat([inputs['input_ids'], gen_prompt_ids], dim=1)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'], device=device)

    # Handle image-related inputs
    if 'pixel_values' in inputs:
        # Keep pixel_values as is
        pass
    if 'image_grid_thw' in inputs:
        # Keep image_grid_thw as is
        pass

    return inputs, thought_idx, question_positions


def get_confidence(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    k=10,
    thought_positions=None,
    return_logits=False,
):
    """
    Confidence = mean over thought positions of (-mean(log(top-k probs))).
    Higher = sharper top-K distribution at thought positions.

    Args:
        model: VL model
        inputs: dict with inputs_embeds (thought positions overwritten in place)
        thought_idx: [start, end) of contiguous thought positions
        thought_hidden_states: latent tensor (num_thought, hidden_dim) to insert
        k: top-k for confidence
        thought_positions: optional explicit positions (overrides thought_idx)
        return_logits: if True, also return logits for downstream reuse

    Returns:
        confidence: scalar tensor
        (logits): only if return_logits=True
    """
    if thought_positions is None:
        inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
        thought_positions = list(range(thought_idx[0], thought_idx[1]))
    else:
        for i, pos in enumerate(thought_positions):
            inputs['inputs_embeds'][0, pos] = thought_hidden_states[i]

    outputs = model(**inputs, return_dict=True)
    logits = outputs['logits'][0]
    probs = torch.softmax(logits, dim=-1)
    confidence = 0.0
    for idx in thought_positions:
        topk = torch.topk(probs[idx], k=k, largest=True)[0]
        confidence -= torch.sum(torch.log(topk + 1e-10)) / k
    confidence = confidence / max(1, len(thought_positions))

    if return_logits:
        return confidence, logits
    return confidence


def _confidence_from_logits_positions(
    logits: torch.Tensor,
    positions: List[int],
    k: int,
) -> torch.Tensor:
    """
    Compute the same confidence surrogate used in get_confidence(), but directly
    from precomputed logits and explicit token positions.
    """
    if logits is None or not positions:
        return torch.tensor(0.0, device=logits.device if logits is not None else "cpu")

    probs = torch.softmax(logits, dim=-1)
    vocab_k = min(int(k), probs.size(-1))
    if vocab_k <= 0:
        return torch.tensor(0.0, device=logits.device)

    confidence = torch.tensor(0.0, device=logits.device)
    valid = 0
    seq_len = probs.size(0)
    for idx in positions:
        pos = int(idx)
        if pos < 0 or pos >= seq_len:
            continue
        topk = torch.topk(probs[pos], k=vocab_k, largest=True)[0]
        confidence = confidence - torch.sum(torch.log(topk + 1e-10)) / vocab_k
        valid += 1

    if valid == 0:
        return torch.tensor(0.0, device=logits.device)
    return confidence / valid


def _compute_thought_progress_reward(
    logits: torch.Tensor,
    thought_positions: List[int],
    k: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Intra-thought trend reward: encourages later thought tokens to have
    higher confidence than earlier ones in the same forward pass.

    Returns:
        trend_reward: mean log-ratio of adjacent thought confidences.
        first_conf: confidence at the first thought position.
        last_conf: confidence at the last thought position.
    """
    if logits is None or not thought_positions:
        zero = torch.tensor(0.0, device=logits.device if logits is not None else "cpu")
        return zero, zero, zero

    # Softmax ONCE (not per-position); confidence = -mean(log(top-k probs)) at each pos.
    probs = torch.softmax(logits, dim=-1)
    vocab_k = min(int(k), probs.size(-1))
    seq_len = probs.size(0)

    confs: List[torch.Tensor] = []
    for pos in thought_positions:
        p = int(pos)
        if 0 <= p < seq_len:
            topk = torch.topk(probs[p], k=vocab_k, largest=True)[0]
            confs.append(-torch.sum(torch.log(topk + 1e-10)) / vocab_k)

    if not confs:
        zero = torch.tensor(0.0, device=logits.device)
        return zero, zero, zero

    first_conf, last_conf = confs[0], confs[-1]
    if len(confs) < 2:
        return torch.tensor(0.0, device=logits.device), first_conf, last_conf

    # Vectorized adjacent log-ratio.
    confs_t = torch.stack(confs)
    log_ratios = torch.log((confs_t[1:] + eps) / (confs_t[:-1] + eps))
    return log_ratios.mean(), first_conf, last_conf


def _compute_contrastive_align_loss(
    hidden_states: torch.Tensor,
    thought_positions: List[int],
    align_pos_pairs: Dict[int, List[int]],
    align_neg_pairs: Optional[Dict[int, List[int]]],
    tau: float,
) -> torch.Tensor:
    losses = []
    tau = max(1e-6, float(tau))
    for i, pos in enumerate(thought_positions):
        pos_positions = align_pos_pairs.get(i)
        if not pos_positions:
            continue
        neg_positions = align_neg_pairs.get(i) if align_neg_pairs else None
        if not neg_positions:
            continue
        thought_vec = hidden_states[0, pos]
        pos_vecs = hidden_states[0, pos_positions]
        neg_vecs = hidden_states[0, neg_positions]
        pos_sims = F.cosine_similarity(thought_vec.unsqueeze(0), pos_vecs, dim=-1) / tau
        neg_sims = F.cosine_similarity(thought_vec.unsqueeze(0), neg_vecs, dim=-1) / tau
        logits = torch.cat([pos_sims, neg_sims], dim=0)
        loss = -(torch.logsumexp(pos_sims, dim=0) - torch.logsumexp(logits, dim=0))
        losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=hidden_states.device)
    return torch.stack(losses).mean()


def _resolve_image_grid_tuple(image_grid_thw) -> Optional[Tuple[int, int, int]]:
    """
    Convert image grid metadata into a (T, H, W) tuple when available.
    """
    if image_grid_thw is None:
        return None

    values = None
    if isinstance(image_grid_thw, torch.Tensor):
        grid = image_grid_thw.detach().cpu()
        if grid.ndim >= 2:
            grid = grid[0]
        values = grid.tolist()
    elif isinstance(image_grid_thw, (list, tuple)):
        values = list(image_grid_thw)
    else:
        try:
            values = list(image_grid_thw)
        except TypeError:
            values = None

    if not values or len(values) < 3:
        return None

    return int(values[0]), int(values[1]), int(values[2])


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    max_start = len(haystack) - len(needle)
    for i in range(max_start + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def _find_question_positions(tokenizer, input_ids: torch.Tensor, question: str) -> List[int]:
    ids_list = input_ids.tolist()
    candidates = [
        f"PROBLEM: {question}\n\n",
        f"PROBLEM: {question}",
        question,
    ]
    for text in candidates:
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        start = _find_subsequence(ids_list, token_ids)
        if start is not None:
            return list(range(start, start + len(token_ids)))
    return []


def compute_image_token_meta(
    input_ids: torch.Tensor,
    processor,
    model=None,
) -> Dict[str, Union[int, torch.Tensor]]:
    """
    Pre-compute image token indices for faster lookup during optimization.
    """
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    vs_mask = input_ids == vision_start_id
    ve_mask = input_ids == vision_end_id

    if vs_mask.any() and ve_mask.any():
        vs_idx = torch.where(vs_mask)[0][0].item()
        ve_idx = torch.where(ve_mask)[0][0].item()
        image_positions = torch.arange(vs_idx + 1, ve_idx, device=input_ids.device)
    else:
        image_token_id = None
        if hasattr(processor, "image_token"):
            image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        elif hasattr(processor.tokenizer, "image_token_id"):
            image_token_id = processor.tokenizer.image_token_id
        elif model is not None and hasattr(model.config, "image_token_id"):
            image_token_id = model.config.image_token_id
        # else:
            # <--- FIX 4: REMOVED bad fallback logic that used "<|image_pad|>"
            # image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        if image_token_id is None:
            raise ValueError(
                "Could not determine a valid image_token_id and <|vision_start|>/<|vision_end|> boundaries not found."
            )

        image_positions = torch.where(input_ids == image_token_id)[0]
        if image_positions.numel() == 0:
            raise ValueError(f"No image tokens (id={image_token_id}) or vision boundaries found in input_ids!")

    return {
        "positions": image_positions,
        "start": image_positions[0].item(),
        "end": image_positions[-1].item() + 1,
        "count": image_positions.numel(),
    }


def generate_vl(
    processor,
    model,
    reward_model: RewardModel,
    question: str,
    image=None,
    messages=None,
    num_thought_tokens: int = 8,
    lr: float = 0.005,
    sigma: float = 20.0,
    sigma_decay: float = 0.95,
    max_num_steps: int = 20,
    reward_threshold: float = -1,
    max_new_tokens: int = 2048,
    disable_best_reward: bool = False,
    kv_reuse: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    device=None,
    align_pos_k: Optional[int] = 32,
    data_idx: Optional[int] = None,
    align_weight: float = 0.1,
    align_contrastive: bool = False,
    align_contrastive_tau: float = 0.07,
    align_neg_k: int = 32,
    patch_select_from_question: bool = False,
    question_patch_assign_mode: str = "rank_slice",
    reward_csv_path: Optional[str] = None,
    **kwargs,
):
    """
    Optimize a small set of learnable latent thought tokens inserted into a
    Vision-Language prompt, then generate the final answer.

    Method (two sequential stages over `max_num_steps` total steps):

      1. Insert `num_thought_tokens` learnable latent embeddings into the prompt
         (initialized from <|endoftext|>, a random Gaussian, or a chosen token).

      2. Stage 1 — Alignment SGD (first `align_sgd_pre_steps` steps, SFT-like):
           a. Forward pass to get logits + attentions + hidden states.
           b. Select top-K image patches (positives) and bottom-K (negatives)
              from question-to-image attention as alignment targets.
           c. Compute contrastive_align_loss between thought latents and
              positive/negative patch hidden states.
           d. Backward through the loss and take an auto-grad SGD step on the
              latents (model weights stay frozen).

      3. Stage 2 — NES reward optimization (remaining steps):
           a. Perturb the latents with Gaussian noise (std = sigma, decayed).
           b. A single image-aware forward pass with the perturbed latents
              (pixel_values are passed so the reward actually sees the image);
              the Stage-1 shared forward is skipped for every Stage-2 step.
           c. Compute reward = confidence + attn_weight * attention_progress_aux,
              both terms derived from that same forward's logits.
              (Alignment is not part of the NES reward; it only drives Stage 1.)
           d. Estimate the gradient with NES (reward-weighted noise) and take
              a manual gradient-ascent step on the latents.

      4. Restore the best-reward latent (unless --disable_best_reward), then
         run standard `model.generate()` to produce the final answer.

    Returns:
        (response_text, best_reward, best_reward_step, stop_reason)
    """

    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 1) build input with latent tokens
    inputs, thought_idx, question_positions = build_vl_inputs(
        processor=processor,
        num_thought_tokens=num_thought_tokens,
        question=question,
        image=image,
        messages=messages,
        data_name=data_name,
        model_name=model_name,
        device=resolved_device,
    )

    # 2) embed text (and image placeholders)
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    input_ids_saved = inputs['input_ids'].clone()
    inputs.pop('input_ids')

    # 3) initialize latent token embeddings (no visual residual now)
    # Rebuttal: opt-in random init (via kwarg latent_init_mode="random").
    # Default behavior — copy the <|endoftext|> embedding — is unchanged.
    _latent_init_mode = str(kwargs.get("latent_init_mode", "endoftext")).lower()
    if _latent_init_mode == "random":
        # Default std=0.02 matches Qwen word embedding element scale; override via
        # kwarg for scale-sweep ablations.
        _latent_init_scale = float(kwargs.get("latent_init_scale", 0.02))
        base_init = torch.randn_like(
            inputs_embeds[0, thought_idx[0]:thought_idx[1]]
        ) * _latent_init_scale
    elif _latent_init_mode == "token":
        # Rebuttal ablation: copy a specific vocab token's embedding into the latent slots.
        # Tests whether <|endoftext|>'s semantic prior is special vs. any other single-token
        # anchor at the same scale.
        _tok_str = str(kwargs.get("latent_init_token", "<|endoftext|>"))
        tokenizer = getattr(processor, "tokenizer", processor)
        _tok_id = tokenizer.convert_tokens_to_ids(_tok_str)
        if _tok_id is None or _tok_id < 0:
            raise ValueError(f"latent_init_token={_tok_str!r} not in vocabulary")
        _emb_layer = model.get_input_embeddings()
        _tok_emb = _emb_layer(
            torch.tensor([_tok_id], device=inputs_embeds.device)
        ).squeeze(0).to(inputs_embeds.dtype)
        _latent_length = thought_idx[1] - thought_idx[0]
        base_init = _tok_emb.unsqueeze(0).expand(_latent_length, -1).contiguous().clone()
    else:
        base_init = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()
    thought_hidden_states = torch.nn.Parameter(
        base_init.detach().requires_grad_(True)
    )
    optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()

    embed_device = inputs_embeds.device
    reward_log_entries = []
    # Positive-patch budget per thought token (None = no cap, use all image tokens).
    pos_k = align_pos_k if (align_pos_k is not None and align_pos_k > 0) else None

    thought_vision_attn_reward_weight = float(kwargs.get("thought_vision_attn_reward_weight", 0.0))
    use_thought_vision_attn_reward = thought_vision_attn_reward_weight != 0.0

    # Two-stage optimization in auto-grad mode:
    # Stage 1: visual alignment SGD ("SFT-like") for first N steps
    # Stage 2: NES reward optimization (confidence + aux) for remaining steps
    align_sgd_pre_steps = max(0, int(kwargs.get("align_sgd_pre_steps", 3)))
    if verbose >= 1:
        log.info(
            f"Optimization: {align_sgd_pre_steps} Stage-1 align-SGD step(s) then "
            f"Stage-2 NES; max_num_steps={max_num_steps}"
        )
    if patch_select_from_question and not question_positions and verbose >= 1:
        log.warning("patch_select_from_question enabled but question span not found; falling back to thought-based selection.")
        patch_select_from_question = False
    thought_positions_abs = list(range(thought_idx[0], thought_idx[1]))

    # Patch selection is sample-invariant when it depends only on the (frozen) prefix:
    # question -> image attention doesn't change across RL steps, so we can compute
    # align_pairs / align_neg_pairs once and reuse them across every step that needs
    # them. The thought-based fallback path (patch_select_from_question=False) uses
    # per-step thought attention, so we cannot cache in that case.
    patch_selection_is_invariant = (
        align_contrastive and patch_select_from_question and bool(question_positions)
    )
    align_pairs_cache = None
    align_neg_pairs_cache = None
    thought_positions_cache = None

    # --- kv_reuse: prefill the frozen prefix ONCE and reuse its KV every Stage-2 step ---
    # Only latent thought tokens change across steps; under causal masking the prefix
    # (system + image + question) K/V are identical every step. So prefill them once and
    # let each Stage-2 step forward only the [thought + suffix] tail on top of the cache.
    kv_ts, kv_te = thought_idx[0], thought_idx[1]
    kv_seq_len = input_ids_saved.shape[1]
    # Correct M-RoPE position ids for the whole prompt (image 2D positions). The model can
    # only derive these from input_ids, but every optimization forward here uses
    # inputs_embeds (to carry the latent), so we compute them ONCE and pass them explicitly
    # to every forward -- otherwise HF falls back to plain sequential positions.
    kv_attn_full = torch.ones_like(input_ids_saved)
    with torch.no_grad():
        mrope_position_ids, _ = model.model.get_rope_index(
            input_ids_saved,
            image_grid_thw=inputs.get("image_grid_thw"),
            attention_mask=kv_attn_full,
        )
    prefix_cache = None
    if kv_reuse:
        with torch.no_grad():
            prefix_cache = DynamicCache()
            model(
                input_ids=input_ids_saved[:, :kv_ts],
                attention_mask=kv_attn_full[:, :kv_ts],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                position_ids=mrope_position_ids[..., :kv_ts],
                cache_position=torch.arange(0, kv_ts, device=embed_device),
                past_key_values=prefix_cache,
                use_cache=True,
            )
        if verbose >= 1:
            log.info(
                f"kv_reuse: prefilled prefix={kv_ts} tokens; "
                f"Stage-2 tail={kv_seq_len - kv_ts} tokens/step"
            )

    # =========================
    # OPTIMIZATION LOOP (Stage 1 SGD -> Stage 2 NES)
    # =========================
    for step in range(max_num_steps):
        # ── per-step reset ────────────────────────────────────────
        thought_positions = None
        align_pairs = None
        align_neg_pairs = None
        first_thought_conf = torch.tensor(0.0, device=embed_device)
        last_thought_conf = torch.tensor(0.0, device=embed_device)
        thought_vision_attn_reward = torch.tensor(0.0, device=embed_device)
        optimizer.zero_grad()

        # Stage 1 (SGD alignment) for the first `align_sgd_pre_steps` steps;
        # every remaining step runs Stage 2 (NES).
        is_sgd = step < align_sgd_pre_steps

        # ── (A) build candidate latent ────────────────────────────
        if is_sgd:
            epsilon = torch.zeros_like(thought_hidden_states, device=embed_device)
            candidate_latent = thought_hidden_states + epsilon              # keep autograd graph
        else:
            epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(embed_device)
            candidate_latent = thought_hidden_states.detach() + epsilon
        inputs_embeds_step = inputs_embeds.clone()
        inputs_embeds_step[0, thought_idx[0]:thought_idx[1]] = candidate_latent

        # ── (B) alignment forward — run only in Stage 1, to get the attentions +
        # hidden_states used to build alignment pairs. Stage 2 skips it (its reward
        # comes from get_confidence()'s own forward).
        need_align_pairs = is_sgd
        attentions = None
        hidden_states = None
        if need_align_pairs:
            outputs = model(
                inputs_embeds=inputs_embeds_step,
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                position_ids=mrope_position_ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states[-1]
            del outputs

        # ── (C) select positive/negative alignment pairs (attention-based) ──
        # Only Stage 1 (SGD) consumes align_pairs / align_neg_pairs; skip in NES.
        new_inputs_embeds = inputs_embeds_step
        new_attn_mask = inputs['attention_mask']
        if align_contrastive and need_align_pairs:
            # Fast path: patch selection depends only on prefix, so reuse the
            # first step's result on every subsequent step.
            if patch_selection_is_invariant and align_pairs_cache is not None:
                align_pairs = align_pairs_cache
                align_neg_pairs = align_neg_pairs_cache
                thought_positions = thought_positions_cache
            else:
                image_token_meta = compute_image_token_meta(input_ids_saved[0], processor, model)
                image_start, image_end = image_token_meta["start"], image_token_meta["end"]

                valid_attn = [attn for attn in attentions if attn is not None]
                avg_attention = torch.cat(valid_attn, dim=1).mean(dim=1)  # (1, seq_len, seq_len)

                num_thought = thought_idx[1] - thought_idx[0]
                thought_positions = list(range(thought_idx[0], thought_idx[1]))
                align_pairs = {}
                if align_contrastive:
                    align_neg_pairs = {}
                question_att_to_images = None
                question_sorted_rel_indices = None
                question_index_sorted_rel_indices = None
                question_cursor = 0
                for think_offset in range(num_thought):
                    if patch_select_from_question and question_positions:
                        if question_att_to_images is None:
                            question_att_to_images = avg_attention[0, question_positions, image_start:image_end].mean(dim=0)
                            question_sorted_rel_indices = torch.argsort(question_att_to_images, descending=True).tolist()
                            if question_patch_assign_mode == "index_slice":
                                total_image_tokens = question_att_to_images.size(0)
                                total_k = 0
                                for _ in range(num_thought):
                                    k_limit = min(pos_k, total_image_tokens) if pos_k is not None else total_image_tokens
                                    total_k += max(0, k_limit)
                                top_rel = question_sorted_rel_indices[:max(0, total_k)]
                                question_index_sorted_rel_indices = sorted(top_rel)
                        att_to_images = question_att_to_images
                    else:
                        current_thought_idx = thought_idx[0] + think_offset
                        att_to_images = avg_attention[0, current_thought_idx, image_start:image_end]
                    total_image_tokens = att_to_images.size(0)

                    k_limit = min(pos_k, total_image_tokens) if pos_k is not None else total_image_tokens

                    if k_limit <= 0:
                        continue

                    if patch_select_from_question and question_positions:
                        full_sorted_rel_indices = question_sorted_rel_indices
                        if question_patch_assign_mode == "index_slice":
                            base = question_index_sorted_rel_indices or []
                            sorted_rel_indices = base[question_cursor:question_cursor + k_limit]
                            question_cursor += k_limit
                        else:
                            sorted_rel_indices = question_sorted_rel_indices[question_cursor:question_cursor + k_limit]
                            question_cursor += k_limit
                    else:
                        full_sorted_rel_indices = torch.argsort(att_to_images, descending=True).tolist()
                        sorted_rel_indices = full_sorted_rel_indices
                    chosen_abs_ids: List[int] = []
                    for rel_idx in sorted_rel_indices:
                        abs_idx = image_start + rel_idx
                        chosen_abs_ids.append(abs_idx)
                        if len(chosen_abs_ids) >= k_limit:
                            break

                    if chosen_abs_ids:
                        align_pairs[think_offset] = chosen_abs_ids
                        if align_contrastive and align_neg_pairs is not None and align_neg_k > 0:
                            neg_abs_ids: List[int] = []
                            for rel_idx in reversed(full_sorted_rel_indices):
                                abs_idx = image_start + rel_idx
                                if abs_idx in chosen_abs_ids:
                                    continue
                                neg_abs_ids.append(abs_idx)
                                if len(neg_abs_ids) >= align_neg_k:
                                    break
                            if neg_abs_ids:
                                align_neg_pairs[think_offset] = neg_abs_ids

                # Persist for reuse across subsequent RL steps (default config only).
                if patch_selection_is_invariant:
                    align_pairs_cache = align_pairs
                    align_neg_pairs_cache = align_neg_pairs
                    thought_positions_cache = thought_positions

        # Pass pixel_values / image_grid_thw so the confidence forward actually SEES the
        # image: HF Qwen-VL injects visual features into inputs_embeds only when
        # pixel_values is provided; without it the reward is computed image-blind.
        inputs_step = dict(
            inputs_embeds=new_inputs_embeds,
            attention_mask=new_attn_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            position_ids=mrope_position_ids,
        )

        # ── (D) stage-specific reward computation + latent update ─
        if is_sgd:
            # ─── Stage 1: alignment SGD ───────────────────────────
            # Compute contrastive alignment loss with autograd, backprop directly.
            total_align_loss = torch.tensor(0.0, device=embed_device)
            if align_pairs and align_neg_pairs:
                align_loss = _compute_contrastive_align_loss(
                    hidden_states,
                    thought_positions_abs,
                    align_pairs,
                    align_neg_pairs,
                    align_contrastive_tau,
                )
                total_align_loss = total_align_loss + align_loss
                if verbose >= 1:
                    log.info(f"Step {step}: align_loss = {align_loss.item():.6f}")
            elif verbose >= 1:
                log.warning(f"Step {step} (Stage 1): no align_pairs/neg_pairs, SGD step skipped.")
            align_objective = -align_weight * total_align_loss
            if align_objective.requires_grad:
                align_objective.backward(retain_graph=False)
                optimizer.step()
            # Reward tensor is used only for best-step tracking / logging in this stage.
            reward = align_objective.detach()
        else:
            # ─── Stage 2: NES reward optimization ─────────────────
            # reward = confidence + attn_weight * attention_progress_aux
            # (alignment is not part of the NES reward; it drives Stage 1 only.)
            with torch.no_grad():
                if kv_reuse:
                    # forward only the [thought + suffix] tail on the cached prefix KV;
                    # thought positions are the first (kv_te - kv_ts) tokens of the tail.
                    tail_embeds = inputs_embeds_step[:, kv_ts:, :]
                    out_tail = model(
                        inputs_embeds=tail_embeds,
                        attention_mask=torch.ones(1, kv_seq_len, device=embed_device, dtype=torch.long),
                        position_ids=mrope_position_ids[..., kv_ts:],
                        cache_position=torch.arange(kv_ts, kv_seq_len, device=embed_device),
                        past_key_values=prefix_cache,
                        use_cache=True,
                    )
                    prefix_cache.crop(kv_ts)  # drop appended tail; keep prefix for next step
                    logits_tail = out_tail.logits[0]
                    tail_thought_positions = list(range(kv_te - kv_ts))
                    conf_reward = _confidence_from_logits_positions(
                        logits_tail, tail_thought_positions, top_k)
                    if use_thought_vision_attn_reward:
                        thought_vision_attn_reward, cf, cl = _compute_thought_progress_reward(
                            logits=logits_tail, thought_positions=tail_thought_positions, k=top_k)
                        first_thought_conf = cf.detach()
                        last_thought_conf = cl.detach()
                else:
                    conf_reward, logits_step = get_confidence(
                        model=model,
                        inputs=inputs_step,
                        thought_idx=thought_idx,
                        thought_hidden_states=candidate_latent,
                        k=top_k,
                        return_logits=True,
                    )
                    if use_thought_vision_attn_reward:
                        thought_vision_attn_reward, cf, cl = _compute_thought_progress_reward(
                            logits=logits_step,
                            thought_positions=thought_positions_abs,
                            k=top_k,
                        )
                        first_thought_conf = cf.detach()
                        last_thought_conf = cl.detach()
            reward = conf_reward + thought_vision_attn_reward_weight * thought_vision_attn_reward
            # NES manual gradient-ascent update.
            denom = max(float(sigma) ** 2, 1e-8)
            grad_ascent = lr * reward * epsilon / denom
            with torch.no_grad():
                thought_hidden_states += grad_ascent

        sigma *= sigma_decay

        reward_value = float(reward.detach().cpu().item())

        # check and update best reward
        is_new_best = reward_value > best_reward
        if is_new_best:
            best_reward, best_reward_step = reward_value, step
            best_thought_hidden_states = thought_hidden_states.clone()

        reward_log_entries.append({
            "step": int(step),
            "reward": reward_value,
            "sigma": float(sigma),
            "is_new_best": int(is_new_best),
            "patch_budget": int(pos_k) if pos_k is not None else -1,
            "first_thought_conf": float(first_thought_conf.detach().cpu().item()),
            "last_thought_conf": float(last_thought_conf.detach().cpu().item()),
            "tv_attn_reward": float(thought_vision_attn_reward.detach().cpu().item()),
        })

        # log reward for each step (blue if no update, green if updated)
        if verbose >= 1:
            if is_new_best:
                color = Fore.GREEN
                update_msg = " [NEW BEST!]"
            else:
                color = Fore.BLUE
                update_msg = ""

            log.debug(f"{color}Step {step}: Reward = {reward_value:.6f}, sigma = {sigma:.6f}, best_reward = {best_reward:.6f}{update_msg}{Style.RESET_ALL}")

        if reward_threshold > 0 and reward_value >= reward_threshold:
            break

        # Clean up memory at the end of each step
        del attentions, hidden_states, new_inputs_embeds, new_attn_mask, inputs_step
        torch.cuda.empty_cache()

    if reward_csv_path is not None:
        try:
            reward_dir = os.path.dirname(reward_csv_path)
            if reward_dir:
                os.makedirs(reward_dir, exist_ok=True)
            with open(reward_csv_path, "w", newline="") as csv_file:
                fieldnames = [
                    "step",
                    "reward",
                    "sigma",
                    "is_new_best",
                    "patch_budget",
                    "first_thought_conf",
                    "last_thought_conf",
                    "tv_attn_reward",
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(reward_log_entries)
        except Exception as exc:
            log.error(f"Failed to write reward log to {reward_csv_path}: {exc}")

    # =========================
    # Apply final latent
    # =========================
    # disable_best_reward=True means use the latest optimized latent directly,
    # instead of rolling back to the best-reward step latent.
    final_latent = thought_hidden_states if disable_best_reward else best_thought_hidden_states

    # =========================
    # Final generation
    # =========================
    if "qwen3" in model.config.model_type.lower():
        bad_words = []
        bad_words_ids = None
    else:
        bad_words = ["addCriterion"]
        bad_words_ids = processor.tokenizer(bad_words, add_special_tokens=False).input_ids

    _gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        bad_words_ids=bad_words_ids,
        do_sample=False,
        num_beams=1,
    )
    # Rebuttal bench: opt-in streamer for TTFT measurement (does not affect other runs).
    _bench_streamer = kwargs.get("_bench_streamer")
    if _bench_streamer is not None:
        _gen_kwargs["streamer"] = _bench_streamer

    # Generate from input_ids so Qwen-VL computes correct M-RoPE (image 2D positions) and
    # continues rope correctly; splice the optimized latent into the thought slots via an
    # embedding hook that fires only on the prefill pass.
    _emb_layer = model.get_input_embeddings()

    def _latent_inject_hook(_mod, _inp, _out):
        if _out.shape[1] > 1:  # prefill only; skip single-token decode steps
            _out = _out.clone()
            _out[0, thought_idx[0]:thought_idx[1]] = final_latent.to(_out.dtype)
        return _out

    _hook = _emb_layer.register_forward_hook(_latent_inject_hook)
    try:
        outputs = model.generate(
            input_ids=input_ids_saved,
            attention_mask=torch.ones_like(input_ids_saved),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            **_gen_kwargs,
        )
    finally:
        _hook.remove()

    input_length = input_ids_saved.shape[1]
    response = processor.decode(outputs[0][input_length:], skip_special_tokens=True)
    stop_reason = get_stop_reason_vl(outputs, input_length, max_new_tokens, processor.tokenizer)

    return response, best_reward, best_reward_step, stop_reason
