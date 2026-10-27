import json
import ast
import pandas as pd

def safe_parse_json(value):
    """
    Attempts to parse a string as JSON or Python literal. 
    Returns the parsed object or a default empty dict/list if parsing fails.
    
    UPDATED: Now handles Lists gracefully to prevent 'The truth value of an array' errors.
    """
    # 1. If it's already a list or dict, return immediately (Pass-through)
    if isinstance(value, (dict, list)):
        return value
        
    # 2. Check for empty/NaN
    if pd.isna(value) or value == "":
        return {}

    # 3. Try parsing stringified data
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return {}

def inspect_column_types(dataset, logger):
    """
    Checks the format of each column to determine if it is a plain string 
    or a JSON object serialized as a string.
    """
    logger.info("--- Inspecting Dataset Column Types ---")
    # Take a sample row from the first available split
    split = list(dataset.keys())[0]
    try:
        sample_row = dataset[split][0]
    except (IndexError, KeyError):
        logger.info("Dataset empty or invalid structure.")
        return

    for col, value in sample_row.items():
        col_type = type(value).__name__
        parsed_preview = str(value)[:50] + "..."
        
        # Check if it looks like a JSON string
        if isinstance(value, str):
            if value.strip().startswith(("{", "[")):
                col_type = "Potential JSON-String"

        logger.info(f"Column: {col:<35} | Detected Type: {col_type:<15} | Content: {parsed_preview}")
    logger.info("---------------------------------------")

def format_deepseek_prompt(example, logger=None):
    """
    Maps raw dataset columns to the DeepSeek-R1 input/output format.
    UPDATED: Uses parallel lists (section_titles/texts) instead of JSON dicts.
    """
    
    # --- 1. Retrieve Data from New List Columns ---
    # We use .get() with a default empty list [] to handle missing data safely
    
    # Sections (Parallel Lists)
    sec_titles = example.get('section_titles', [])
    sec_texts = example.get('section_texts', [])
    
    # Citations (Parallel Lists)
    case_titles = example.get('cited_cases', [])
    case_judgments = example.get('cited_case_judgments', [])

    # --- 2. Format Sections ---
    # We zip the titles and texts together. 
    # If one list is shorter, zip stops at the shortest length (safe).
    fmt_sections = ""
    if isinstance(sec_titles, list) and len(sec_titles) > 0:
        # Use zip to pair title with its text
        for idx, (title, text) in enumerate(zip(sec_titles, sec_texts), 1):
            fmt_sections += f"Statute {idx}. {title}:\n   {text}\n\n"
    else:
        fmt_sections = "No specific statutes provided."

    # --- 3. Format Citations ---
    fmt_citations = ""
    if isinstance(case_titles, list) and len(case_titles) > 0:
        # Use zip to pair case name with its judgment text
        for idx, (case, info) in enumerate(zip(case_titles, case_judgments), 1):
            # Fallback if judgment text is missing or None
            if info is None: 
                info = "Details not available."
            fmt_citations += f"Precedent {idx}. {case}:\n   {info}\n\n"
    else:
        fmt_citations = "No citations provided."

    # --- 4. Construct System & User Prompt ---
    system_msg = "SYSTEM:\nYou are a smart and intelligent legal assistant for the Indian legal domain. Based on the user's instructions, you will have to perform or assist in some tasks related to the Indian legal system. Since these tasks have some legal application, only provide responses you are extremely certain about, and avoid being ambiguous or uncertain. Ensure that your outputs adhere to the user's instructions or requirements.\n"
    
    # Use 'full_text' if available, otherwise 'Text'
    case_proceedings = example.get('Simplified_Facts', '')
    
    user_content = f"""USER:\n
    You are a legal expert tasked with making a judgment about whether an appeal should be accepted or rejected based on the provided case proceeding, cited statutes and cited cases. Your task is to evaluate whether the appeal should be accepted (1) or rejected (0) based on the input.

    ### Now, evaluate the following case:
    Case Proceedings: {case_proceedings}
    
    Relevant Statutes:
    {fmt_sections.strip()}
    
    Cited Cases Reference:
    {fmt_citations.strip()}
    
    Provide your judgment by strictly following this format:
    ##PREDICTION: [Insert your prediction here]
    ##EXPLANATION: [Insert your reasoning here that led you to your prediction.]
    Strictly do not include anything outside this format. Strictly follow the provided format. Do not generate placeholders. Just provide the final judgment and explanation.\n"""

    # --- 5. Construct Assistant Response (Target) ---
    assistant_header = f"""ASSISTANT:\n### Response:\n"""
    
    s_issue = example.get('Simplified_Issue', "Issue analysis not provided.")
    s_pet_args = example.get('Simplified_Arguments_of_Petitioner', "Petitioner arguments not provided.")
    s_res_args = example.get('Simplified_Arguments_of_Respondent', "Respondent arguments not provided.")

    thought_content = f"""**Legal Issue Analysis:**
        {s_issue}

        **Arguments of Petitioner:**
        {s_pet_args}

        **Arguments of Respondent:**
        {s_res_args}

        **Deliberation:**
        Weighing the arguments against the relevant statutes and cited cases to form a decision."""

    s_decision = example.get('Simplified_Decision', "0")
    s_reasoning = example.get('Simplified_Reasoning', "Reasoning not provided.")

    response_content = f"##PREDICTION: {s_decision}\n##EXPLANATION: {s_reasoning}"
    
    full_response = f"<think>\n{thought_content}\n</think>\n{response_content}"
    
    output = f"{assistant_header}{full_response}"
    if logger and (len(case_titles) == 0 and len(sec_titles) == 0):
        # Logging only the first 500 chars to keep logs readable. 
        # Remove [:500] if you want the full dump.
        logger.info("="*40)
        logger.info(f"INPUT PREVIEW:\n{system_msg + user_content}\n") 
        logger.info(f"TARGET PREVIEW:\n{output}\n")
        logger.info("="*40)

    return {
        "input_text": system_msg + user_content,
        "target_text": output
    }



# =========================================
#  PHI-4 REASONING PROMPT FORMATTER
# =========================================

def _format_case_data(example):
    """
    Shared helper: Extracts and formats case data from example dict.
    Returns (fmt_sections, fmt_citations, case_proceedings) strings.
    """
    sec_titles = example.get('section_titles', [])
    sec_texts  = example.get('section_texts', [])
    case_titles = example.get('cited_cases', [])
    case_judgments = example.get('cited_case_judgments', [])

    fmt_sections = ""
    if isinstance(sec_titles, list) and len(sec_titles) > 0:
        for idx, (title, text) in enumerate(zip(sec_titles, sec_texts), 1):
            fmt_sections += f"Statute {idx}. {title}:\n   {text}\n\n"
    else:
        fmt_sections = "No specific statutes provided."

    fmt_citations = ""
    if isinstance(case_titles, list) and len(case_titles) > 0:
        for idx, (case, info) in enumerate(zip(case_titles, case_judgments), 1):
            if info is None:
                info = "Details not available."
            fmt_citations += f"Precedent {idx}. {case}:\n   {info}\n\n"
    else:
        fmt_citations = "No citations provided."

    case_proceedings = example.get('Simplified_Facts', '')

    return fmt_sections, fmt_citations, case_proceedings


def format_phi4_prompt(example, tokenizer=None, logger=None):
    """
    Maps raw dataset columns to the Phi-4-reasoning ChatML input/output format.
    
    Phi-4-reasoning / Phi-4-reasoning-plus use ChatML tokens:
        <|im_start|>system<|im_sep|> ... <|im_end|>
        <|im_start|>user<|im_sep|> ... <|im_end|>
        <|im_start|>assistant<|im_sep|> ...

    Phi-4-mini-reasoning uses Phi-native tokens:
        <|system|> ... <|end|>
        <|user|> ... <|end|>
        <|assistant|> ...

    We detect which template to use based on the tokenizer's vocab.
    If tokenizer is None or has im_start token, we default to ChatML (Phi-4-reasoning).
    """

    # --- 1. Extract shared case data ---
    fmt_sections, fmt_citations, case_proceedings = _format_case_data(example)

    # --- 2. Build user message content ---
    user_content = (
        "You are a legal expert tasked with making a judgment about whether an appeal should be accepted or rejected based on the provided case proceeding, cited statutes and cited cases. Your task is to evaluate whether the appeal should be accepted (1) or rejected (0) based on the input.\n\n"
        f"### Now, evaluate the following case:\n"
        f"Case Proceedings: {case_proceedings}\n\n"
        f"Relevant Statutes:\n{fmt_sections.strip()}\n\n"
        f"Cited Cases Reference:\n{fmt_citations.strip()}\n\n"
        "Provide your judgment by strictly following this format:\n"
        "##PREDICTION: [Insert your prediction here]\n"
        "##EXPLANATION: [Insert your reasoning here that led you to your prediction.]\n"
        "Strictly do not include anything outside this format. Strictly follow the provided format. Do not generate placeholders. Just provide the final judgment and explanation."
    )

    # --- 3. Detect template format ---
    use_mini_format = False
    if tokenizer is not None:
        # Phi-4-mini-reasoning has <|system|> token, Phi-4-reasoning has <|im_start|>
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}
        if "<|system|>" in vocab and "<|im_start|>" not in vocab:
            use_mini_format = True

    # --- 4. Build system prompt ---
    system_prompt = (
        "You are a smart and intelligent legal assistant for the Indian legal domain. Based on the user's instructions, you will have to perform or assist in some tasks related to the Indian legal system. Since these tasks have some legal application, only provide responses you are extremely certain about, and avoid being ambiguous or uncertain. Ensure that your outputs adhere to the user's instructions or requirements.\n"
    )

    # --- 5. Construct full input text ---
    if use_mini_format:
        # Phi-4-mini-reasoning format
        input_text = (
            f"<|system|>{system_prompt}<|end|>"
            f"<|user|>{user_content}<|end|>"
            f"<|assistant|>\n"
        )
    else:
        # Phi-4-reasoning / Phi-4-reasoning-plus ChatML format
        input_text = (
            f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user<|im_sep|>\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant<|im_sep|>\n"
        )

    # --- 6. Construct target text (assistant response) ---
    s_issue = example.get('Simplified_Issue', "Issue analysis not provided.")
    s_pet_args = example.get('Simplified_Arguments_of_Petitioner', "Petitioner arguments not provided.")
    s_res_args = example.get('Simplified_Arguments_of_Respondent', "Respondent arguments not provided.")

    thought_content = (
        f"**Legal Issue Analysis:**\n"
        f"{s_issue}\n\n"
        f"**Arguments of Petitioner:**\n"
        f"{s_pet_args}\n\n"
        f"**Arguments of Respondent:**\n"
        f"{s_res_args}\n\n"
        f"**Deliberation:**\n"
        f"Weighing the arguments against the relevant statutes and cited cases to form a decision."
    )

    s_decision = example.get('Simplified_Decision', "0")
    s_reasoning = example.get('Simplified_Reasoning', "Reasoning not provided.")

    response_content = f"##PREDICTION: {s_decision}\n##EXPLANATION: {s_reasoning}"
    
    target_text = f"<think>\n{thought_content}\n</think>\n{response_content}"

    if logger:
        logger.info("=" * 40)
        logger.info(f"[PHI-4] Format: {'mini' if use_mini_format else 'chatml'}")
        logger.info(f"[PHI-4] INPUT PREVIEW:\n{input_text}...")
        logger.info(f"[PHI-4] TARGET PREVIEW:\n{target_text}...")
        logger.info("=" * 40)

    return {
        "input_text": input_text,
        "target_text": target_text
    }


# =========================================
#  QWEN 3.5 REASONING PROMPT FORMATTER
# =========================================

def format_qwen3_prompt(example, logger=None):
    """
    Maps raw dataset columns to the Qwen 3.5 ChatML input/output format.

    Qwen 3.5 uses ChatML tokens (like Phi-4-reasoning) but without <|im_sep|>:
        <|im_start|>system\n ... <|im_end|>
        <|im_start|>user\n ... <|im_end|>
        <|im_start|>assistant\n ...

    The model uses <think>...</think> for reasoning, same as DeepSeek and Phi-4.
    """

    # --- 1. Extract shared case data ---
    fmt_sections, fmt_citations, case_proceedings = _format_case_data(example)

    # --- 2. Build user message content ---
    user_content = (
        "You are a legal expert tasked with making a judgment about whether an appeal should be accepted or rejected based on the provided case proceeding, cited statutes and cited cases. Your task is to evaluate whether the appeal should be accepted (1) or rejected (0) based on the input.\n\n"
        f"### Now, evaluate the following case:\n"
        f"Case Proceedings: {case_proceedings}\n\n"
        f"Relevant Statutes:\n{fmt_sections.strip()}\n\n"
        f"Cited Cases Reference:\n{fmt_citations.strip()}\n\n"
        "Provide your judgment by strictly following this format:\n"
        "##PREDICTION: [Insert your prediction here]\n"
        "##EXPLANATION: [Insert your reasoning here that led you to your prediction.]\n"
        "Strictly do not include anything outside this format. Strictly follow the provided format. Do not generate placeholders. Just provide the final judgment and explanation."
    )

    # --- 3. Build system prompt ---
    system_prompt = (
        "You are a smart and intelligent legal assistant for the Indian legal domain. Based on the user's instructions, you will have to perform or assist in some tasks related to the Indian legal system. Since these tasks have some legal application, only provide responses you are extremely certain about, and avoid being ambiguous or uncertain. Ensure that your outputs adhere to the user's instructions or requirements.\n"
    )

    # --- 4. Construct full input text (ChatML without <|im_sep|>) ---
    input_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # --- 5. Construct target text (assistant response) ---
    s_issue = example.get('Simplified_Issue', "Issue analysis not provided.")
    s_pet_args = example.get('Simplified_Arguments_of_Petitioner', "Petitioner arguments not provided.")
    s_res_args = example.get('Simplified_Arguments_of_Respondent', "Respondent arguments not provided.")

    thought_content = (
        f"**Legal Issue Analysis:**\n"
        f"{s_issue}\n\n"
        f"**Arguments of Petitioner:**\n"
        f"{s_pet_args}\n\n"
        f"**Arguments of Respondent:**\n"
        f"{s_res_args}\n\n"
        f"**Deliberation:**\n"
        f"Weighing the arguments against the relevant statutes and cited cases to form a decision."
    )

    s_decision = example.get('Simplified_Decision', "0")
    s_reasoning = example.get('Simplified_Reasoning', "Reasoning not provided.")

    response_content = f"##PREDICTION: {s_decision}\n##EXPLANATION: {s_reasoning}"

    target_text = f"<think>\n{thought_content}\n</think>\n{response_content}"

    if logger:
        logger.info("=" * 40)
        logger.info(f"[QWEN3] INPUT PREVIEW:\n{input_text}...")
        logger.info(f"[QWEN3] TARGET PREVIEW:\n{target_text}...")
        logger.info("=" * 40)

    return {
        "input_text": input_text,
        "target_text": target_text
    }
