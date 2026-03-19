from datasets import load_dataset

PORTAL_COLUMNS = [
    "question_id",
    "user_input",
    "prefix",
    "suffix",
    "code_to_edit",
    "answer_a",
    "answer_b",
    "acceptedIndex",
]


def load_lmarena(data_filter):
    data = load_dataset("lmarena-ai/PPE-Human-Preference-V1", split="test")
    data = data.shuffle(seed=13)
    data = data.filter(remove_ties_filter)

    if data_filter == "no_ticks":
        data = data.filter(no_ticks_filter)
    elif data_filter == "has_ticks":
        data = data.filter(soft_tick_filter)
    elif data_filter == "has_tick_is_code":
        data = data.filter(is_tick_code_filter)
    elif data_filter == "has_tick_not_code":
        data = data.filter(tick_not_code_filter)

    df = data.to_pandas()
    df = df.rename(columns={"id": "question_id", "prompt": "user_instruction", "response_1": "answer_a", "response_2": "answer_b", "winner": "winner"})
    df['winner'] = df['winner'].apply(map_winner)
    return df


def load_lmarena_portal(filter_name: str):
    df = load_lmarena(filter_name)
    portal = df.rename(columns={"user_instruction": "user_input"}).copy()
    portal["prefix"] = ""
    portal["suffix"] = ""
    portal["code_to_edit"] = ""
    portal["acceptedIndex"] = portal["winner"].map({"A": 0, "B": 1})
    return portal[PORTAL_COLUMNS]

def remove_ties_filter(row):
    return row["winner"] in ["model_a", "model_b"]

def check_multiple_lang_code(prompt):
    return any("```" + tag in prompt.lower() for tag in ["python", "javascript", "js", "java", "c", "c++", "cpp", 'cs', 'go', 'yaml', 'js', 'c', 'C', 'php', 'sh', 'http', 'py', 'python3', 'solidity', 'TypeScript', 'bash', 'sql', 'lua', 'import', 'R', 'rs'])

def is_tick_code_filter(row):
    # only 35 here
    return row["is_code"] == True and check_multiple_lang_code(row["prompt"])

def soft_tick_filter(row):
    return row["is_code"] == True and "```" in row["prompt"]

def tick_not_code_filter(row):
    return row["is_code"] == True and "```" in row["prompt"] and not check_multiple_lang_code(row["prompt"])

def no_ticks_filter(row):
    return row["is_code"] == True and "```" not in row["prompt"]

def map_winner(winner_str):
    if winner_str == "model_a":
        return "A"
    elif winner_str == "model_b":
        return "B"
    else:
        raise ValueError(f"Unknown winner string: {winner_str}")

def lmarena_prompt_keys(setting):
    """ 
    Returns a dictionary mapping prompt keys to dataset column names for the lmarena dataset.
    ENSURE all keys are present in the prompt template.
    answer_a and answer_b will be added later in main since the order must be swapped in the two calls
    """
    # key = prompt variable name, value = dataset column name
    return {
        'user_instruction': 'user_instruction',
    }
