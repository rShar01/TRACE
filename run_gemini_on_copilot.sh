# Install dependencies
uv venv --python 3.10
uv pip install -r requirements.txt

# Get judge preference results
python -m judge_preference.evaluate_preference \
    --experiment_config experiment_config/gemini-2.5-pro.yaml

# Run analysis of preference results with judge_preference/analysis.py

# Generate rubrics and fit models
python -m rubric_analysis.rubric_judge \
    --experiment_config experiment_config/gemini-2.5-pro.yaml