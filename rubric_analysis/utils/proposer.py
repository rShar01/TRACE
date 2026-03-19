from typing import List
import pandas as pd

from rubric_analysis.utils.llm import get_llm_output
from rubric_analysis.utils.prompts import * 


def parse_bullets(text: str) -> List[str]:
    if "no differences found" in text.lower():
        return []
    lines = text.split("\n")
    bullets: List[str] = []
    current_bullet = ""
    found_first_bullet = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") or stripped.startswith("*"):
            found_first_bullet = True
            if current_bullet:
                bullets.append(current_bullet.strip())
            current_bullet = stripped.replace("* ", "", 1).replace("- ", "", 1).replace("**", "")
        elif stripped and found_first_bullet:
            current_bullet += " " + stripped.replace("**", "")
    if current_bullet:
        bullets.append(current_bullet.strip())
    return [b.strip() for b in bullets if b.strip()]


class RubricProposer:
    def __init__(self, models: List[str], config):
        self.models = models
        self.global_config = config
        self.config = config

    def propose(self, proposer_df: pd.DataFrame, current_rubrics: List[str] = [], num_rubrics: int = 10, **kwargs) -> List[str]:
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        proposer_df = proposer_df.copy()
        proposer_df["batch_id"] = proposer_df.index // self.config["batch_size"]
        unique_batches = proposer_df["batch_id"].unique()

        differences: List[str] = []
        for batch_id in unique_batches:
            batch_df = self._prepare_batch(proposer_df[proposer_df["batch_id"] == batch_id].copy(), current_rubrics)
            prompt_tmpl = refined_proposer_axis if len(current_rubrics) == 0 else refined_proposer_axis_iteration
            diffs = get_llm_output(
                [prompt_tmpl.format(combined_responses=row["combined_responses"]) for _, row in batch_df.iterrows()],
                self.config["model"],
            )
            differences.extend([b.replace("**", "") for diff in diffs for b in parse_bullets(diff)])

        if len(differences) > num_rubrics:
            return self._reduce_rubrics(differences, num_rubrics=num_rubrics)
        return differences

    def _prepare_batch(self, batch_df: pd.DataFrame, current_rubrics: List[str] = []) -> pd.DataFrame:
        def create_combined_response(row, model_order):
            return (
                f"User prompt:\n{row['question']}\n\n"
                f"Model 1:\n{row[model_order[0]]}\n\n"
                f"Model 2:\n{row[model_order[1]]}"
            )

        model_order = [self.models[0], self.models[1]]

        batch_df["single_combined_response"] = batch_df.apply(lambda row: create_combined_response(row, model_order), axis=1)
        batch_df["combined_responses"] = "\n-------------\n".join(batch_df["single_combined_response"].tolist())

        if current_rubrics:
            current_rubrics_str = "Differences I have already found:\n" + "\n".join(current_rubrics)
            batch_df["combined_responses"] = batch_df["combined_responses"].apply(lambda x: x + "\n\n" + current_rubrics_str)
        return batch_df.drop_duplicates("batch_id")

    def _reduce_rubrics(self, differences: List[str], num_rubrics: int) -> List[str]:
        """
        Reduce the list of differences to a smaller list of rubrics.
        """
        print(f"Number of total differences before reduction: {len(differences)}")
        summaries = get_llm_output(
            reduce_freeform_axis.format(differences='\n'.join(differences)),
            self.config["model"]
        )
        rubrics = parse_bullets(summaries)
        rubrics = [rubric.replace("*", "") for rubric in rubrics if rubric != ""]
        print(f"Number of total differences after reduction: {len(rubrics)}")
        return rubrics[:num_rubrics]


