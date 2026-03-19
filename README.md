# TRACE

TRACE (Tool for Rubric Analysis in Code Evaluation) is a framework for evaluating and interpreting LLM-based judges in realistic developer workflows. It measures how closely model judgments align with human preferences, with a focus on disagreement cases rather than just aggregate agreement. To explain divergence, TRACE automatically discovers interpretable rubric items from response differences, then quantifies how human and model judgments correlate with those rubrics.

## What TRACE provides

- Pairwise judge evaluation with answer-order swapping for consistency checks.
- Standardized accuracy and invalid-output analysis over preference datasets.
- Automatic rubric discovery from disagreement examples.
- Rubric-level coefficient analysis comparing human vs judge behavior.

## Evaluate a judge on a preference dataset

1. Configure an experiment
- Create an experiment config in the experiment_config/ folder.
- Add a corresponding model config in the model_configs/ folder.
- Configure API keys and experiment details.

2. Run judge preference evaluation
- Entry point: [judge_preference/evaluate_preference.py](judge_preference/evaluate_preference.py)
- Command:

```bash
python -m judge_preference.evaluate_preference --experiment_config experiment_config/<file>.yaml
```

This generates pairwise decisions in original and swapped answer order and saves outputs to the configured CSV.

3. Run rubric analysis
- Entry point: [rubric_analysis/rubric_judge.py](rubric_analysis/rubric_judge.py)
- Command:

```bash
python -m rubric_analysis.rubric_judge --experiment_config experiment_config/<file>.yaml
```

Outputs include:
- Rubric proposals and ranking files.
- A coefficients table comparing human and judge rubric loadings (with judge confidence intervals).
