import argparse
import os
from typing import List

import pandas as pd
import yaml

from rubric_analysis.utils.proposer import RubricProposer
from rubric_analysis.utils.ranker import RubricRanker
from rubric_analysis.utils.prefs import (
    get_pref_score, map_judge_preference
)
from rubric_analysis.utils.prefs import fit_multi_rubric_logit
from rubric_analysis.utils.llm import get_llm_output
from runner_utils import get_data

def compute_coefficients(
    original_df: pd.DataFrame,
    rubric_df: pd.DataFrame,
    judge_save_file: str,
    models: List[str]
) -> pd.DataFrame:
    # Human coefficients from original data preferences.
    human_pref_df = original_df[["conversation_id", "human_preference"]].copy()
    human_pref_df["preference_feature"] = human_pref_df["human_preference"].apply(lambda x: get_pref_score(x, models))

    human_coef_df = fit_multi_rubric_logit(rubric_df, human_pref_df)
    human_coef_df = human_coef_df[["rubric", "coef_preference"]].rename(
        columns={"coef_preference": "human_coefficient"}
    )

    if not os.path.isfile(judge_save_file):
        raise FileNotFoundError(f"Judge preference results file not found: {judge_save_file}")

    def _filter_invalid_judge(row) -> bool:
        # Filter inconsistent/invalid judge outputs.
        if pd.isna(row["original_order"]):
            return False
        if pd.isna(row["swapped_order"]) or row["original_order"] == row["swapped_order"]:
            return False
        if row["original_order"] not in ["A", "B"]:
            return False
        if row["swapped_order"] not in ["A", "B"]:
            return False
        return True

    judge_df = pd.read_csv(judge_save_file)
    judge_df = judge_df[judge_df.apply(_filter_invalid_judge, axis=1)]
    if len(judge_df) == 0:
        raise ValueError("No valid judge rows after filtering invalid/inconsistent results")

    judge_df["question_id"] = judge_df["question_id"].astype(str)
    judge_df = judge_df.drop_duplicates(subset="question_id").reset_index(drop=True)

    judge_df["judge_preference"] = judge_df["original_order"].apply(lambda x: map_judge_preference(x, models))
    judge_df = judge_df[judge_df["judge_preference"].isin(models)]

    judge_df["judge_preference_feature"] = judge_df["judge_preference"].apply(
        lambda x: get_pref_score(x, models)
    )

    judge_pref_df = judge_df[["question_id", "judge_preference_feature"]].copy()
    judge_pref_df = judge_pref_df.rename(
        columns={"question_id": "conversation_id", "judge_preference_feature": "preference_feature"}
    )

    judge_coef_df = fit_multi_rubric_logit(rubric_df, judge_pref_df)
    judge_coef_df = judge_coef_df[
        ["rubric", "coef_preference", "coef_lower_ci_preference", "coef_upper_ci_preference"]
    ].rename(
        columns={
            "coef_preference": "judge_coefficient",
            "coef_lower_ci_preference": "judge_ci_lower",
            "coef_upper_ci_preference": "judge_ci_upper",
        }
    )

    coefficients_df = human_coef_df.merge(judge_coef_df, on="rubric", how="outer")
    coefficients_df = coefficients_df.sort_values("rubric").reset_index(drop=True)
    return coefficients_df

def propose_rubrics(df: pd.DataFrame, config, current_rubrics: List[str], models) -> List[str]:
    rubrics = RubricProposer(models, config).propose(df, current_rubrics, config["num_rubrics"])
    if len(current_rubrics) > 0:
        rubrics = deduplicate_rubrics(current_rubrics, rubrics)
    print("Proposed Rubrics:")
    print("* " + "\n* ".join(rubrics))
    print("--------------------------------")
    return rubrics

def deduplicate_rubrics(existing_rubrics: List[str], new_rubrics: List[str]) -> List[str]:
    """
    Deduplicate rubrics by removing duplicates and keeping the first occurrence.
    """
    prompt = """Here is a list of properties on which two strings may vary.
{existing_axes} 
{new_axes}

It is likely that several of these axes measure similar things. Your task is to remove any redundant properties. Think about if a user would gain any new information from seeing both properties. For example, "Emotional Tone: High: Contains emotionally charged language. Low: Maintains a neutral tone." and "Empathy: High: Shows empathy. Low: Only factual answers without empathy." are redundant because they both measure the emotional content of the text. If two similar properties are found, keep the one that is more informative.

Output the reduced list of properties, seperated by a newline. Do not include any other text in your response.

Your Response:"""
    deduped_rubrics = get_llm_output(prompt.format(existing_axes="\n".join(existing_rubrics), new_axes="\n".join(new_rubrics)), model="gpt-4o")
    return deduped_rubrics.split("\n")


def rank_rubrics(rubrics: List[str], df: pd.DataFrame, config, models) -> pd.DataFrame:
    ranker = RubricRanker(config, models)
    rubric_df = ranker.score(
        rubrics,
        df,
        single_position_rank=config["single_position_rank"]
    )
    return rubric_df

def main(config):
    models = ["answer_a", "answer_b"]
    for api_key, value in config["api_keys"].items():
        os.environ[api_key] = value

    # Output dir identical naming but we will only write one csv
    output_dir = f"{config['rubric_analysis']['output_dir']}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = get_data(config)
    judge_df = pd.read_csv(config["judge_preference"]["save_file"])
    df = df[df["question_id"].isin(judge_df["question_id"])] # filter to only samples from the judges

    if not all([c in df.columns for c in models + ["question"]]):
        raise ValueError(f"Models {models} or question column not found in dataframe.")
    if (num_samples := config["rubric_analysis"]["num_samples"]) and num_samples < len(df):
        df = df.sample(config["rubric_analysis"]["num_samples"], random_state=42)

    # Keep only rows where preference is one of the compared models
    df = df[df["human_preference"].isin(models)].reset_index(drop=True)
    df["conversation_id"] = df["question_id"]

    # Rubric discovery
    print("Discovering rubrics...")
    proposer_config = config["rubric_analysis"]["proposer"]
    running_rubrics: List[str] = list(proposer_config["initial_rubrics"]) if len(proposer_config["initial_rubrics"]) > 0 else []
    proposer_df = df.copy()
    if (num_samples := proposer_config["num_samples"]) and num_samples < len(df):
        proposer_df = df.sample(proposer_config["num_samples"], random_state=42).reset_index(drop=True)
    if proposer_config["iterations"]:
        for it in range(proposer_config["iterations"]):
            print(f"Iteration {it + 1} of {proposer_config['iterations']}")
            running_rubrics = propose_rubrics(proposer_df, proposer_config, running_rubrics, models)

    rubrics_path = os.path.join(output_dir, "rubrics.csv")
    with open(rubrics_path, "w", encoding="utf-8") as f:
        for rubric in running_rubrics:
            f.write(f"{rubric}\n")

    # Rank rubrics
    print("Ranking rubrics...")
    ranker_config = config["rubric_analysis"]["ranker"]
    rubric_df_path = os.path.join(output_dir, "rubric_df.csv")
    if ranker_config["skip_ranking"]:
        if os.path.exists(rubric_df_path):
            rubric_df = pd.read_csv(rubric_df_path)
        else:
            raise ValueError("Rubric df doesn't exist")
    else:
        rubric_df = rank_rubrics(running_rubrics, df, ranker_config, models)
        rubric_df.to_csv(rubric_df_path, index=False)

    # Compute coefficients
    coefficients_df = compute_coefficients(df, rubric_df, config["judge_preference"]["save_file"], models)
    coefficients_df.to_csv(os.path.join(output_dir, "coefficients.csv"), index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, default="experiment_config/gemini-2.5-pro.yaml")
    args = parser.parse_args()

    with open(args.experiment_config, "r") as f:
        experiment_config = yaml.safe_load(f)

    main(experiment_config)


