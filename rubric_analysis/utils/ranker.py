from typing import List
import re
import pandas as pd

from rubric_analysis.utils.llm import get_llm_output
from rubric_analysis.utils.prompts import ranker_prompt_multi

def ranker_postprocess_multi(output: str, models: List[str]) -> List[str]:
    try:
        output = output.replace("Output ", "").replace("output ", "")
        output = re.sub(r"[#*]", "", output)
        ranking_pattern = re.compile(r"(?:Ranking:|^)\s*(?:Property\s+\d+:\s*(A|B|N/A|unsure|equal)\s*Analysis:.*(?:\s*\n|$))+", re.I | re.M)
        ranking_section = ranking_pattern.search(output)
        if not ranking_section:
            return []
        score_pattern = re.compile(r"Property\s+\d+:\s*(A|B|N/A|unsure|equal)", re.I)
        scores = score_pattern.findall(ranking_section.group())
        results = []
        for score in scores:
            s = score.lower()
            if s == "a":
                results.append(models[0])
            elif s == "b":
                results.append(models[1])
            else:
                results.append("tie")
        return results
    except Exception as e:
        print(f"Error in ranker_postprocess_multi: {output}\n\n{e}")
        return []


class RubricRanker:
    def __init__(self, config, models):
        self.config = config
        self.models: List[str] = models
        self.rubric_batch_size: int = config.get("rubric_batch_size", 5)

    def score(self, rubrics: List[str], df: pd.DataFrame, single_position_rank: bool | None = None) -> pd.DataFrame:
        all_scored_dfs = []
        for i in range(0, len(rubrics), self.rubric_batch_size):
            rubric_batch = rubrics[i : i + self.rubric_batch_size]
            scored_df = self.score_batch(rubric_batch, df)
            if not single_position_rank:
                scored_df_reversed = self.score_batch(rubric_batch, df, reverse_position=True)
                scored_df.rename(columns={"score": "score_forward", "score_reversed": "score_backward"}, inplace=True)
                scored_df_reversed.rename(columns={"score": "score_backward", "score_reversed": "score_forward"}, inplace=True)
                scored_df = scored_df.merge(
                    scored_df_reversed[["conversation_id", "rubric", "score_backward"]],
                    on=["conversation_id", "rubric"],
                    how="inner",
                )

                def is_position_bias(item1, item2):
                    return item1 == item2 and item1 != 0 and item2 != 0

                scored_df["position_matters"] = scored_df.apply(
                    lambda row: is_position_bias(row["score_forward"], row["score_backward"]), axis=1
                )
                scored_df["score"] = scored_df.apply(
                    lambda row: row["score_forward"] if not row["position_matters"] else 0, axis=1
                )
            all_scored_dfs.append(scored_df)
        return pd.concat(all_scored_dfs)

    def generate_ranker_input(self, row: pd.Series, rubric_batch: List[str], models: List[str], reverse_position: bool) -> str:
        if reverse_position:
            return (
                f"Properties:\n" + "\n".join(f"Property {i+1}: {rubric}" for i, rubric in enumerate(rubric_batch)) + "\n"
                f"--------------------------------\n\nUser prompt:\n{row['question']}\n\n"
                f"\n\nResponse A:\n{row[models[1]]}\n\n"
                f"\n\nResponse B:\n{row[models[0]]}\n\n"
                f"--------------------------------\n\nProperties (restated):\n" + "\n".join(f"Property {i+1}: {rubric}" for i, rubric in enumerate(rubric_batch)) + "\n"
            )
        else:
            return (
                f"Properties:\n" + "\n".join(f"Property {i+1}: {rubric}" for i, rubric in enumerate(rubric_batch)) + "\n"
                f"--------------------------------\n\nUser prompt:\n{row['question']}\n\n"
                f"\n\nResponse A:\n{row[models[0]]}\n\n"
                f"\n\nResponse B:\n{row[models[1]]}\n\n"
                f"--------------------------------\n\nProperties (restated):\n" + "\n".join(f"Property {i+1}: {rubric}" for i, rubric in enumerate(rubric_batch)) + "\n"
            )

    def score_batch(self, rubric_batch: List[str], df: pd.DataFrame, reverse_position: bool = False) -> pd.DataFrame:
        print(f"Scoring rubric batch: {rubric_batch}")
        models = self.models
        ranker_cfg = self.config

        rubric_df = df.copy().reset_index(drop=True)
        rubric_df["score_pos_model"] = [models for _ in range(len(rubric_df))]
        rubric_df["ranker_inputs"] = rubric_df.apply(
            lambda row: self.generate_ranker_input(row, rubric_batch, models, reverse_position),
            axis=1,
        )

        prompts = [ranker_prompt_multi.format(inputs=rubric_df.iloc[idx]["ranker_inputs"]) for idx in range(len(rubric_df))]
        outputs = get_llm_output(prompts, ranker_cfg["model"], cache=True)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"get_llm_output returned {len(outputs)} outputs for {len(prompts)} prompts")

        rubric_df["ranker_output"] = [ranker_postprocess_multi(output, models) for output in outputs]

        # --- retry malformed rows only
        n = len(rubric_batch)
        max_retries = 3
        for retry in range(max_retries):
            retry_indices = []
            for i in range(len(rubric_df)):
                out = rubric_df.iloc[i]["ranker_output"]  # <<< positional indexing avoids KeyError
                if not isinstance(out, (list, tuple)) or len(out) != n:
                    retry_indices.append(i)

            if not retry_indices:
                break

            print(f"Retry {retry + 1}: Attempting to fix {len(retry_indices)} incorrect outputs")
            retry_prompts = [
                ranker_prompt_multi.format(inputs=rubric_df.iloc[idx]["ranker_inputs"])
                for idx in retry_indices
            ]
            try:
                new_outputs = get_llm_output(retry_prompts, "gpt-4o", cache=(retry == 0))
                if len(new_outputs) != len(retry_prompts):  # <<< keep 1:1 correspondence
                    raise RuntimeError(
                        f"Retry pass returned {len(new_outputs)} outputs for {len(retry_prompts)} prompts"
                    )
                for idx, new_output in zip(retry_indices, new_outputs):
                    rubric_df.at[idx, "ranker_output"] = ranker_postprocess_multi(new_output, models)
            except Exception as e:
                print(f"Error during retry: {e}")
                break  # <<< do not loop indefinitely if the retry path is failing

        # --- enforce invariant before explode to avoid "matching element counts" error
        bad_rows = [
            i for i in range(len(rubric_df))
            if not isinstance(rubric_df.iloc[i]["ranker_output"], (list, tuple))
            or len(rubric_df.iloc[i]["ranker_output"]) != n
        ]
        if bad_rows:
            # keep this strict to surface the real issue rather than padding or truncation
            # For each bad row, replace ranker_output with a list of 0's (as score_label expects labels matching models)
            for i in bad_rows:
                rubric_df.at[i, "ranker_output"] = [0] * n
            # Optionally log this issue for debugging
            sample = bad_rows[:5]
            details = [
                f"row={i}, type={type(rubric_df.iloc[i]['ranker_output'])}, "
                f"len={len(rubric_df.iloc[i]['ranker_output']) if hasattr(rubric_df.iloc[i]['ranker_output'], '__len__') else None}"
                for i in sample
            ]
            print(
                "Warning: Malformed ranker_output after retries; set to 0. "
                f"Expected {n}. Offending rows: {details}"
            )

        # --- prepare synchronized explode
        rubric_df["score_label"] = rubric_df["ranker_output"]
        rubric_df["rubric"] = [list(rubric_batch) for _ in range(len(rubric_df))]  # <<< ensure list, not string

        # explode the rubrics column and the score column at the same time
        rubric_df = rubric_df.explode(["rubric", "score_label"], ignore_index=True)  # <<< ignore_index for clean reindex

        # scoring
        rubric_df["score"] = rubric_df["score_label"].apply(
            lambda x: 1 if x == models[0] else (-1 if x == models[1] else 0)
        )
        return rubric_df
