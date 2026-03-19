from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# from rubric_utils.llm import get_llm_output

def get_pref_score(preference: str, models: list):
    if preference == models[0]:
        return 1
    elif preference == models[1]:
        return -1
    else:
        raise ValueError(f"Invalid preference: {preference}")

def map_judge_preference(label: str, models: List[str]) -> str | None:
    if pd.isna(label):
        return None
    label = str(label).strip().upper()
    if label == "A":
        return models[0]
    if label == "B":
        return models[1]
    return None


def build_multi_rubric_alignment_training_data(
    rubric_df: pd.DataFrame,
    preference_df: pd.DataFrame,
) -> tuple:
    """
    Args:
        rubric_df: Long format DataFrame with columns [conversation_id, rubric, score]
        preference_df: DataFrame with columns [conversation_id, preference_col]
    
    Returns:
        tuple: (feature_df, X, y) where:
            - feature_df: Wide format DataFrame with one row per conversation, one column per rubric
            - X: Normalized feature matrix (n_conversations x n_rubrics)
            - y: Preference labels {-1, 1}
    """
    # Check required columns
    required = ["conversation_id", "rubric", "score"]
    missing = [c for c in required if c not in rubric_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    pref_required = ["conversation_id", "preference_feature"]
    if (pref_missing := [c for c in pref_required if c not in preference_df.columns]):
        raise ValueError(f"Missing preference columns: {pref_missing}")
    
    # Keep only rows with usable ids/labels before pivoting.
    df = rubric_df.loc[rubric_df["conversation_id"].notna(), required]
    pref_df = preference_df.loc[
        preference_df["conversation_id"].notna() & preference_df["preference_feature"].notna(),
        pref_required,
    ]

    # Pivot to wide format: conversation_id × rubric → one row per conversation
    feature_df = pd.pivot_table(
        df, 
        values="score", 
        index="conversation_id", 
        columns="rubric", 
        fill_value=0
    )

    # Build one preference label per conversation id and align to pivot index order.
    pref_by_conversation = (
        pref_df.drop_duplicates(subset="conversation_id")
        .set_index("conversation_id")["preference_feature"]
    )
    y_pref = pref_by_conversation.reindex(feature_df.index).to_numpy()
    
    # Filter out ties (0s) and missing labels - uninformative for learning.
    valid_mask = pd.notna(y_pref) & (y_pref != 0)
    feature_df = feature_df[valid_mask]
    y_pref = y_pref[valid_mask]

    if len(feature_df) == 0:
        raise ValueError("No valid training rows after aligning preferences to conversation_id")
    
    X = feature_df.to_numpy()
    
    # Normalize features (critical for comparable coefficients)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return feature_df, X, y_pref


def fit_multi_rubric_logit(
    rubric_df: pd.DataFrame,
    preference_df: pd.DataFrame,
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Fit multi-rubric logistic regression with bootstrap for uncertainty.
    
    Args:
        rubric_df: Long format DataFrame with rubric scores
        preference_df: DataFrame containing conversation_id and preference labels/features
        preference_col: Name of preference column in preference_df
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        DataFrame with one row per rubric showing coefficient and confidence intervals
    """
    # Build training data
    feature_df, X, y = build_multi_rubric_alignment_training_data(rubric_df, preference_df)
    # breakpoint()
    # Bootstrap
    n = len(X)
    rng = np.random.default_rng()
    bootstrap_coefs = {rubric: [] for rubric in feature_df.columns}
    
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        
        model = LogisticRegression(
            penalty="l2", 
            solver="liblinear", 
            random_state=42,
            fit_intercept=True
        )
        model.fit(Xb, yb)
        
        # Store coefficients (matching order of feature_df.columns)
        for i, rubric in enumerate(feature_df.columns):
            bootstrap_coefs[rubric].append(model.coef_[0, i])
    
    # Aggregate results
    rows = []
    for rubric in feature_df.columns:
        coefs = bootstrap_coefs[rubric]
        if len(coefs) == 0:
            rows.append({
                "rubric": rubric,
                "coef_preference": 0.0,
                "coef_std_preference": 0.0,
                "coef_lower_ci_preference": 0.0,
                "coef_upper_ci_preference": 0.0,
            })
        else:
            rows.append({
                "rubric": rubric,
                "coef_preference": float(np.mean(coefs)),
                "coef_std_preference": float(np.std(coefs)),
                "coef_lower_ci_preference": float(np.percentile(coefs, 2.5)),
                "coef_upper_ci_preference": float(np.percentile(coefs, 97.5)),
            })
    
    return pd.DataFrame(rows).sort_values("coef_preference", ascending=False)


# Backward-compatible aliases.
build_multi_vibe_alignment_training_data = build_multi_rubric_alignment_training_data
fit_multi_vibe_logit = fit_multi_rubric_logit
