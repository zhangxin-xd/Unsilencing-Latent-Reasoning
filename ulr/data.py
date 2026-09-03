"""
Data API for Vision-Language Models
Load datasets from JSON files
"""
from datasets import Dataset
import os
import json
import logging
from multiprocessing import Pool, cpu_count
from .prompts import vl_cot_prompt


def process_single_item(item, index):
    """
    Process a single item from JSON data.
    
    Args:
        item: Dictionary with prompt, solution, image_path, idx
        index: Index in the list (fallback for idx)
        
    Returns:
        Processed item dictionary
    """
    processed_item = {
        'question': item.get('prompt', ''),
        'answer': item.get('solution', ''),
        'idx': item.get('idx', index),
        'image_path': item.get('image_path', ''),
        # Store image_path directly instead of loading PIL Image
        'image': item.get('image_path', '')  # Use path directly, not PIL Image
    }
    return processed_item


def load_json_dataset(json_path: str, num_workers=None, start_idx=None, end_idx=None):
    """
    Load dataset from JSON file using multiprocessing.
    
    Expected JSON format:
    [
      {
        "prompt": "...",
        "solution": "...",
        "image_path": "...",
        "idx": 0
      }
    ]
    
    Args:
        json_path: Path to JSON file
        num_workers: Number of worker processes (default: cpu_count())
        start_idx: Optional start index for subset (inclusive)
        end_idx: Optional end index for subset (exclusive)
        
    Returns:
        Dataset object with question, answer, image_path columns
        Note: image field contains the path string, not PIL Image object
    """
    print(f"📄 Loading from JSON file: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of samples, got {type(data)}")

    # Apply index range if specified (before processing to save time)
    total_items = len(data)
    base_index = 0  # Base index for item indices
    if start_idx is not None and end_idx is not None:
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, total_items)
        base_index = start_idx  # Store the base index for correct item indexing
        data = data[start_idx:end_idx]
        print(f"📊 Selected subset: indices {start_idx} to {end_idx} ({len(data)} items from {total_items} total)")
    else:
        print(f"📊 Total items in file: {total_items}")

    # Use multiprocessing to process items in parallel
    if num_workers is None:
        num_workers = min(cpu_count(), 16)  # Limit to 16 workers max

    print(f"🔄 Processing {len(data)} items with {num_workers} workers...")

    if num_workers > 1 and len(data) > 100:  # Use multiprocessing for large datasets
        with Pool(processes=num_workers) as pool:
            # Use base_index + i to maintain correct global indices
            processed_data = pool.starmap(process_single_item, [(item, base_index + i) for i, item in enumerate(data)])
    else:
        # Sequential processing for small datasets
        processed_data = [process_single_item(item, base_index + i) for i, item in enumerate(data)]

    dataset = Dataset.from_list(processed_data)
    print(f"✓ Loaded {len(dataset)} samples from JSON file")
    return dataset


def get_vl_dataset(data_name_or_path, processor, prompt_idx, start_idx=None, end_idx=None):
    """
    Load vision-language dataset from JSON file
    
    Args:
        data_name_or_path: path to JSON file
        processor: VL processor
        prompt_idx: which query prompt to use
        start_idx: optional start index for subset
        end_idx: optional end index for subset
    
    Returns:
        dataset: formatted dataset
    
    Usage:
        From JSON file: get_vl_dataset("data/mathverse.json", ...)
    """

    # Check if it's a JSON file
    if not (data_name_or_path.endswith('.json') and os.path.exists(data_name_or_path)):
        raise ValueError(
            f"Only JSON files are supported. "
            f"Expected a JSON file path, got: {data_name_or_path}"
        )

    # Load from JSON file
    dataset = load_json_dataset(data_name_or_path, start_idx=start_idx, end_idx=end_idx)

    # Use standard column names for all datasets
    question_col = "question"
    answer_col = "answer"
    image_col = "image"

    # Add a simple messages formatter - NO map operation needed!
    # Map is slow due to PyArrow serialization, we'll build messages on-the-fly instead
    def add_messages_column(example):
        """Dynamically add messages field when accessed"""
        q = example[question_col]

        # Apply CoT prompt based on prompt_idx
        # prompt_idx=0: Baseline CoT (default) - "solve step by step"
        # prompt_idx=1: Detailed CoT - explicit step-by-step instructions
        # prompt_idx=2: No CoT - direct answer only
        q_with_prompt = vl_cot_prompt(q, prompt_idx=prompt_idx)

        # Get image path or object
        image_data = example.get(image_col, None)

        # Keep image as path string directly, don't load PIL Image
        # This allows both local paths and URLs to be passed through
        image_obj = image_data  # Use directly as path/URL string

        # If it's not a string (e.g., already a PIL Image from other datasets), keep as is
        if image_data is not None and not isinstance(image_data, str):
            image_obj = image_data

        # Don't include image object in messages - it's passed separately to processor
        example['messages'] = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},  # Placeholder for chat template
                    {'type': 'text', 'text': q_with_prompt}
                ]
            }
        ]
        # Ensure consistent column names
        if 'question' not in example:
            example['question'] = q
        if 'answer' not in example:
            example['answer'] = example[answer_col]
        # Set image to the path/URL string or existing object
        example['image'] = image_obj
        # Preserve image_path
        if 'image_path' not in example:
            # Extract image_path from image field if it's a string
            if isinstance(image_data, str):
                example['image_path'] = image_data
            else:
                example['image_path'] = example.get('image_path', '')
        return example

    # Use set_transform for lazy evaluation - NO preprocessing overhead!
    dataset.set_transform(add_messages_column)

    print(f"Dataset ready (lazy transform applied, no preprocessing needed)")
    return dataset
