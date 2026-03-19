import pandas as pd

from pathlib import Path
from ast import literal_eval

def safe_literal_eval(x):
    """Safely evaluate string to literal Python object."""
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating string: {e}")
        print(f"Problematic string: {x[:100]}...")  # Print first 100 chars
        return None

def load_editbench(path, apply_filters: bool = True):
    p = Path(path)
    df = pd.read_csv(p)
    df['responseItems'] = df['responseItems'].apply(safe_literal_eval)
    df = df.dropna(subset=['responseItems'])

    if apply_filters:
        df = df[df['privacy'] == 'Research'].copy()
        if 'acceptedIndex' in df.columns:
            df = df[df['acceptedIndex'].isin([0, 1])].copy()
        df = df[df.apply(filter_valid_rows, axis=1)]

    df['answer_a'] = df.apply(extract_answer_A, axis=1)
    df['answer_b'] = df.apply(extract_answer_B, axis=1)
    if apply_filters:
        answer_a = df['answer_a'].fillna("").astype(str).str.strip()
        answer_b = df['answer_b'].fillna("").astype(str).str.strip()
        df = df[answer_a != answer_b].copy()
    df["prefix"] = df.apply(extract_prefix, axis=1)
    df["suffix"] = df.apply(extract_suffix, axis=1)
    df['user_input'] = df.apply(extract_user_instruction, axis=1)
    df['code_to_edit'] = df.apply(extract_code_to_edit, axis=1)
    if 'acceptedIndex' in df.columns:
        df["winner"] = df['acceptedIndex'].apply(lambda x: 'A' if x == 0 else 'B')
    else:
        df["winner"] = "B"
    df['question'] = df.apply(extract_question, axis=1)
    df = df.rename(columns={
        'pairId': 'question_id',
    })
    df["human_preference"] = df['acceptedIndex'].apply(lambda x: 'answer_a' if x == 0 else 'answer_b')
    return df

def filter_valid_rows(row):
    response_items = row['responseItems']
    return "prefix" in response_items[0].keys() and "user_input" in response_items[0].keys() and "code_to_edit" in response_items[0].keys()

def extract_question(row):
    return f"""
    This is the prefix of the coding file:
    {row['prefix']}

    This is the suffix of the file:
    {row['suffix']}

    This is the code selected by the user to rewrite:
    {row['code_to_edit']}

    The user has given the instructions:
    {row['user_input']}
    """

def extract_answer_A(row):
    return row['responseItems'][0].get('response', '')

def extract_answer_B(row):
    return row['responseItems'][1].get('response', '')

def extract_prefix(row):
    return row['responseItems'][0].get('prefix', '')

def extract_suffix(row):
    return row['responseItems'][0].get('suffix', '')

def extract_user_instruction(row):
    return row['responseItems'][0].get('user_input', '')

def extract_code_to_edit(row):
    return row['responseItems'][0].get('code_to_edit', '')

def editbench_prompt_keys(setting):
    """ 
    Returns a dictionary mapping prompt keys to dataset column names for the editbench dataset.
    ENSURE all keys are present in the prompt template except answer_a and answer_b.
    answer_a and answer_b will be added later in main since the order must be swapped in the two calls
    """
    # key = prompt variable name, value = dataset column name
    if setting == 'pairwise':
        return {
            "user_input": "user_input",
            "prefix": "prefix",
            "suffix": "suffix",
            "code_to_edit": "code_to_edit",
        }
    else:
        raise NotImplementedError(f"Setting {setting} not implemented for editbench prompt keys.")
