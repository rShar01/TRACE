import pandas as pd
import numpy as np

from json import loads
from pathlib import Path

def load_copilot(path):
    p = Path(path)
    df = pd.read_csv(p)
    df['completionItems'] = df['completionItems'].apply(loads)
    selected_data = df[df['acceptedIndex'] == 1].sample(frac=1.0, random_state=42)
    selected_data = selected_data[selected_data.apply(filter_valid_rows, axis=1)]
    selected_data['answer_a'] = selected_data.apply(extract_answer_A, axis=1)
    selected_data['answer_b'] = selected_data.apply(extract_answer_B, axis=1)

    # Swap randomly so preferences aren't all the same
    swap_mask = np.random.RandomState(42).rand(len(selected_data)) < 0.5
    tmp = selected_data.loc[swap_mask, "answer_a"].copy()
    selected_data.loc[swap_mask, "answer_a"] = selected_data.loc[swap_mask, "answer_b"].values
    selected_data.loc[swap_mask, "answer_b"] = tmp
    selected_data["human_preference"] = np.where(swap_mask, "answer_a", "answer_b")
    selected_data["winner"] = np.where(swap_mask, "A", "B")


    selected_data['prefix'] = selected_data.apply(extract_prefix, axis=1)
    selected_data['suffix'] = selected_data.apply(extract_suffix, axis=1)
    selected_data['question'] = selected_data.apply(extract_question, axis=1)
    selected_data = selected_data.rename(columns={
        'pairId': 'question_id',
        'userPrompt': 'user_instruction',
    })
    return selected_data


def extract_question(row):
    if (prompt := row["completionItems"][0].get("prompt")) is not None and len(prompt) > 0:
        if isinstance(prompt, list):
            task = prompt[0]["content"]
        elif isinstance(prompt, str):
            task = prompt
        else:
            raise ValueError("Prompt is empty or not in expected format")
    else:
        print("No prompt found, using prefix and suffix")
        task = f""" 
        <prefix>
        {row['completionItems'][0]['prefix']}
        </prefix>
        <suffix>
        {row['completionItems'][0]['suffix']}
        </suffix>
         """
    
    return """
    Fill in code and output nothing else. Respect spacing, new lines, and indentation.
    Begin your snippet with the last line of code in the prefix and end with the first line of the suffix. Be VERY mindful of indentation. Make sure it is correct.

    Task:
    """ + task


def filter_valid_rows(row):
    completion_items = row['completionItems']
    return "prefix" in completion_items[0].keys()

def extract_answer_A(row):
    return row['completionItems'][0]['completion']

def extract_answer_B(row):
    return row['completionItems'][1]['completion']

def extract_prefix(row):
    return row['completionItems'][0]['prefix']

def extract_suffix(row):
    return row['completionItems'][0]['suffix']


def copilot_prompt_keys(setting):
    """ 
    Returns a dictionary mapping prompt keys to dataset column names for the copilot dataset.
    ENSURE all keys are present in the prompt template except answer_a and answer_b.
    answer_a and answer_b will be added later in main since the order must be swapped in the two calls
    """
    # key = prompt variable name, value = dataset column name
    if setting == 'pairwise':
        return {
            "prefix": "prefix",
            "suffix": "suffix",
        }
    else:
        raise NotImplementedError(f"Setting {setting} not implemented for copilot prompt keys.")
