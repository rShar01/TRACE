import pandas as pd
import numpy as np
import argparse

from pathlib import Path

def print_file(path, is_reward_model=False):
    df = pd.read_csv(path)

    if "winner" not in df.columns:
        df["winner"] = "B"
    not_follow_instruct = 0
    decision_inconsistent = 0
    correct = 0
    wrong = 0
    for row in df.itertuples():
        if is_reward_model:
            if row.original_order is np.nan or row.swapped_order is np.nan:
                not_follow_instruct += 1
            elif row.original_order == row.winner:
                correct += 1
            else:
                wrong += 1
        else:
            if row.original_order is np.nan or row.swapped_order is np.nan:
                not_follow_instruct += 1
            elif row.original_order == row.swapped_order and row.original_order != "C":
                decision_inconsistent += 1
            elif row.original_order == "C" and row.swapped_order == "C":
                correct += 1
            elif row.winner == row.original_order :
                correct += 1
            else:
                wrong += 1
    total = len(df)
    percent_invalid = (decision_inconsistent + not_follow_instruct) / total
    accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0
    print(f"Total: {total}, inconsistent: {decision_inconsistent}, blank: {not_follow_instruct}, correct: {correct}, wrong: {wrong}")
    print(f"Invalid: {percent_invalid:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{total}, {decision_inconsistent}, {not_follow_instruct}, {correct}, {wrong}, {percent_invalid:.4f}, {accuracy:.4f}")

def main(args):
    p = Path(args.results_path)
    if p.is_file():
        print_file(p, args.is_reward_model)
    elif p.is_dir():
        for file in p.glob("**/*.csv"):
            try:
                print(f"Results for {file}:")
                print_file(file)
                print("--------------------------------------------------")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    else:
        print(f"Path {args.results_path} is not a valid file or directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--is_reward_model", action="store_true")
    args = parser.parse_args()
    main(args)