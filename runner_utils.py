import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import yaml

from judge_preference.model import Prompt
from lmarena_utils import load_lmarena, lmarena_prompt_keys
from lmarena_utils_new import load_lmarena as load_lmarena_new
from copilot_utils import load_copilot, copilot_prompt_keys
from editbench_utils import load_editbench, editbench_prompt_keys


def extract_answer(input_string: str) -> Optional[str]:
    # Pattern 1: Look for content within double square brackets inside <answer> tags
    pattern1 = r"<answer>\s*\[\[(.*?)\]\]\s*</answer>"
    match1 = re.search(pattern1, input_string)

    if match1:
        return match1.group(1)

    # Pattern 2: Look for [RESULT] followed by content
    pattern2 = r"\[RESULT\]\s+(.+?)(?:\n|$)"
    match2 = re.search(pattern2, input_string)

    if match2:
        return match2.group(1).strip()

    return None


def build_prompt_fill(
    prompt_key_map: Dict[str, str],
    row,
    extra_fill: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    curr_fill: Dict[str, str] = {}
    for prompt_key, row_key in prompt_key_map.items():
        if hasattr(row, row_key):
            curr_fill[prompt_key] = getattr(row, row_key)
        else:
            raise KeyError(f"Key {row_key} not found")

    if extra_fill:
        curr_fill.update(extra_fill)

    return curr_fill


def generate_answer(
    prompt_template: Dict[str, str],
    prompt_key_map: Dict[str, str],
    row,
    model,
    extra_fill: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional[str]]:
    prompt = Prompt()
    curr_fill = build_prompt_fill(prompt_key_map, row, extra_fill)
    prompt.sys_prompt = prompt_template["system"]
    prompt.query_prompt = prompt_template["query"].format(**curr_fill)

    output = model.generate(prompt)
    extracted_answer = extract_answer(output)

    return output, extracted_answer


def get_data(exp_config) -> pd.DataFrame:
    # rows should have at least columns question_id, user_instruction, answer_a, answer_b, winner
    if exp_config["dataset"] == "PPE-Human-Preference-V1":
        data = load_lmarena(exp_config["data_filter"])
    elif exp_config["dataset"] == "arena-human-preference-140k":
        data = load_lmarena_new()
    elif exp_config["dataset"] == "arena-human-preference-140k-preprocessed":
        data = pd.read_csv(exp_config["data_path"])
    elif exp_config["dataset"] == "copilot":
        data = load_copilot(exp_config["data_path"])
    elif exp_config["dataset"] == "editbench":
        apply_filters = not exp_config.get("editbench_skip_filters", False)
        data = load_editbench(exp_config["data_path"], apply_filters=apply_filters)
    else:
        raise NotImplementedError(f"Dataset {exp_config['dataset']} not implemented.")

    return data


def get_prompt_keys(experiment_config):
    if experiment_config["dataset"] == "PPE-Human-Preference-V1":
        return lmarena_prompt_keys("pairwise")
    if experiment_config["dataset"] == "arena-human-preference-140k":
        return lmarena_prompt_keys("pairwise")
    if experiment_config["dataset"] == "arena-human-preference-140k-preprocessed":
        return lmarena_prompt_keys("pairwise")
    if experiment_config["dataset"] == "copilot":
        return copilot_prompt_keys("pairwise")
    if experiment_config["dataset"] == "editbench":
        return editbench_prompt_keys("pairwise")

    raise NotImplementedError(
        f"Dataset {experiment_config['dataset']} not implemented for prompt keys."
    )


def load_prompt_template(prompt_name: str) -> Dict[str, str]:
    prompt_path = Path("prompts") / f"{prompt_name}.yaml"
    with open(prompt_path, "r") as f:
        prompt_template = yaml.safe_load(f)
    return prompt_template