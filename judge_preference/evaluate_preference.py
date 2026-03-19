import argparse

import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm

from judge_preference.model import get_model, get_model_config
from runner_utils import generate_answer, get_data, get_prompt_keys, load_prompt_template

def main(args):
    with open(args.experiment_config, "r") as f:
        experiment_config = yaml.safe_load(f)

    # datasets should have col for winners with value A or B
    data: pd.DataFrame = get_data(experiment_config)
    n = len(data)
    print(f"Found {n} examples in dataset {experiment_config['dataset']}")

    config = get_model_config(f"model_configs/{experiment_config['model']}.yaml")
    model = get_model(config)
    print("Loaded model ", experiment_config['model'])

    # prompt template should have user_instruction, answer_a, and answer_b as keys for str format
    prompt_template = load_prompt_template(experiment_config["judge_preference"]["prompt_name"])
    
    save_path = Path(experiment_config["judge_preference"]["save_file"])

    results_list = []
    num_samples = experiment_config["judge_preference"]["num_samples"] if experiment_config["judge_preference"]["num_samples"] else n
    progress_bar = tqdm(total=num_samples)
    for row in data.itertuples():
        id = row.question_id
        
        prompt_key_map = get_prompt_keys(experiment_config)
        prompt_key_map["answer_a"] = "answer_a"
        prompt_key_map["answer_b"] = "answer_b"
        first_generation, first_answer = generate_answer(prompt_template, prompt_key_map, row , model)
        # first_generation, first_answer = "", "A"
        print(first_generation)
        print(first_answer)

        prompt_key_map = get_prompt_keys(experiment_config)
        prompt_key_map["answer_a"] = "answer_b"
        prompt_key_map["answer_b"] = "answer_a"
        second_generation, second_answer = generate_answer(prompt_template, prompt_key_map, row , model)
        # second_generation, second_answer = "", "B"

        result = {}
        result['question_id'] = id
        result["winner"] = row.winner
        result['original_order'] = first_answer
        result['swapped_order'] = second_answer
        result['original_cot'] = first_generation.replace('\n', '\\n')
        result['swapped_cot'] = second_generation.replace('\n', '\\n')
        results_list.append(result)

        progress_bar.update(1)
        if len(results_list) == num_samples:
            break

    progress_bar.close()
    if save_path.is_file():
        prev_df = pd.read_csv(save_path)
        new_rows = pd.DataFrame(results_list)
        df = pd.concat([prev_df, new_rows])
        df = df.reset_index(drop=True)
    else:
        df = pd.DataFrame(results_list)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, default="experiment_config/lmarena/gemini-2.5-pro.yaml")
    args = parser.parse_args()
    main(args)
