def vl_cot_prompt(q, prompt_idx=0):
    """
    CoT prompts for Vision-Language models (MathVerse, MathVista, etc.)
    
    Args:
        q (str): The question to be solved (without image, image is passed separately)
        prompt_idx (int): The index of the prompt to be used
    
    Returns:
        str: The formatted prompt text (not messages, since VL uses different format)
    """
    prompts = [
        # idx 0: Baseline CoT with step-by-step reasoning (DEFAULT)
        (
            "Please analyze the image carefully and solve this problem step by step.\n"
            "Show your reasoning process clearly, then put your final answer within \\boxed{}.\n\n"
            f"Question: {q}"
        ),
        # idx 1: Detailed CoT with explicit instructions
        (
            "Let's solve this visual math problem step by step:\n"
            "1. First, carefully observe the image and identify all relevant information.\n"
            "2. Break down the problem into smaller steps.\n"
            "3. Show your calculations and reasoning for each step.\n"
            "4. Finally, provide your answer within \\boxed{}.\n\n"
            f"Question: {q}"
        ),
        # idx 2: No CoT (Direct Answer) - for comparison
        (
            "Please provide your final answer within \\boxed{}.\n\n"
            f"Question: {q}"
        ),
        # idx 3: Short instruction for choice questions
        (
            "Answer with the option's letter from the given choices directly.\n\n"
            f"Question: {q}"
        ),

        (
            "Put your final answer within \\boxed{}.\n\n"
            f"Question: {q}"
        ),
        (
            f"Question: {q}"
            "\nPut your final answer within \\boxed{}.\n\n"

        ),
        (
            f"Question: {q}"
            "\nOutput only the final answer within \\boxed{}.\n\n"

        ),
    ]
    return prompts[prompt_idx]


SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
"""
