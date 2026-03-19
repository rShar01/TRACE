from datasets import load_dataset
import pandas as pd
import re
from pathlib import Path

# Heuristics for edit/complete prompts and code-like responses
CODE_KEYWORDS = [
    "def ",
    "class ",
    "function",
    "public ",
    "private ",
    "static ",
    "void ",
    "int ",
    "const ",
    "var ",
    "let ",
    "=>",
    "lambda",
    "#include",
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE TABLE",
    "BEGIN",
    "end",
    "{",
    ";",
    "::",
    "printf",
    "cout",
    "System.out",
    "import ",
    "from ",
    "using ",
    "package ",
]

EDIT_TERMS = [
    "complete",
    "finish",
    "implement",
    "fill in",
    "continue",
    "extend",
    "add",
    "rewrite",
    "refactor",
    "fix",
    "bug",
    "error",
    "debug",
    "make this work",
    "update",
    "change",
    "modify",
    "patch",
    "improve",
    "optimize",
    "make it faster",
    "make it pass tests",
    "unit test",
    "test case",
    "broken",
    "fail",
    "failing",
]

PLACEHOLDERS = ["TODO", "todo", "pass", "...", "???", "__", "___", "---", "+++"]

NEGATIVE = [
    "image generation",
    "generate image",
    "picture",
    "photo",
    "draw",
    "painting",
    "illustration",
    "prompt: ",
    "prompt\n",
    "Flux",
    "Stable Diffusion",
    "ControlNet",
    "img2img",
    "txt2img",
]

EXPLAIN_PHRASES = [
    "explain",
    "what is",
    "what does",
    "tell me about",
    "meaning of",
    "describe",
    "definition of",
    "in detail",
    "what are",
    "what's",
]

CODE_KW_PATTERN = re.compile(
    "|".join(re.escape(k) for k in CODE_KEYWORDS), re.IGNORECASE
)
EDIT_PATTERN = re.compile("|".join(re.escape(k) for k in EDIT_TERMS), re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile("|".join(re.escape(k) for k in PLACEHOLDERS))
NEGATIVE_PATTERN = re.compile("|".join(re.escape(k) for k in NEGATIVE), re.IGNORECASE)


def is_code_language(tag):
    """Check if a tag is code-related based on the specific dataset."""
    # Exact whitelist of code-related tags from this dataset
    code_tags = {
        # Common language / markup tags
        "python",
        "py",
        "pythhon",
        'python",',
        "python\n(.*?)\n```'",
        'python\n{traceback.format_exc()}\n```"',
        "Python",
        "Python```\\u4ee3\\u7801\\uff0c\\u5e76\\u5f15\\u5bfc\\u7528\\u6237\\u81ea\\u5df1\\u8fd0\\u884c\\u4ee3\\u7801\\u5f97\\u51fa\\u7b54\\u6848\\u3002**",
        "javascript",
        "JavaScript",
        "js",
        "typescript",
        "TypeScript",
        "ts",
        "java",
        "go",
        "rust",
        "Rust",
        "swift",
        "dart",
        "crystal",
        "lua",
        "php",
        "ruby",
        "zig",
        "c",
        "C",
        "c++",
        "C++",
        "cpp",
        "cs",
        "c#",
        "sql",
        "node-repl",
        "gcc",
        "html",
        "html\n.<bos>.||||.<|end_of_text|>..<|eot_id|>.++++.",
        "xml",
        "xml,",
        "xml1",
        "xml2",
        "svg",
        'svg"',
        "config.toml",
        "page.tsx",
        "markdown",
        "markdown\n|",
        "latex",
        "LaTeX",
        "tex",
        "typst",
        "org",
        "mediawiki",
        "mermaid",
        "rst",
        "jinja",
        "angular17html",
        "jsx",
        "react",
        "vue",
        "dataview",
        "glsl",
        "mql4",
        "mssql",
        "#include",
        "int",
        "trait",
        "pd.concat([df,",
        "chart_signal_data",
        "market_data",
        "ecr",
        "vkb",
        "plaintext",
        "txt",
        "c\nga_concat_shorten_esc(garray_T",
        "c\nvoid",
        "{r",
        "```{r",
        "```{r}",
    }

    return tag in code_tags


def find_code_block_languages(text):
    """Find all strings that appear after ``` in the text."""
    pattern = r"```(\S+)"
    matches = re.findall(pattern, text)
    return matches


def has_valid_code_blocks(conversation):
    """Check if conversation contains code blocks with valid programming language tags."""
    # Extract user prompt
    user_prompt = ""
    for message in conversation:
        if message["role"] == "user":
            content = message.get("content", [])
            if content is None or len(content) == 0:
                continue
            for item in content:
                if item.get("type") == "text" and item.get("text"):
                    user_prompt += item["text"]
            break

    # Check if there are code blocks with valid language tags
    if "```" not in user_prompt:
        return False

    languages = find_code_block_languages(user_prompt)
    # At least one language tag should be code-related
    return any(is_code_language(lang) for lang in languages)


def is_single_turn(conversation):
    """Check if conversation has exactly one user message and one assistant message."""
    if not conversation or len(conversation) != 2:
        return False

    roles = [msg["role"] for msg in conversation]
    return roles.count("user") == 1 and roles.count("assistant") == 1


def filter_fn(examples):
    """Filter for code examples with clear winners, single-turn conversations, and valid code blocks."""
    # Check basic conditions
    if not (
        examples["is_code"] == True
        and (examples["winner"] == "model_a" or examples["winner"] == "model_b")
    ):
        return False

    # Only keep evaluation_order == 1
    if examples.get("evaluation_order") != 1:
        return False

    # Check both conversations are single-turn
    if not (
        is_single_turn(examples["conversation_a"])
        and is_single_turn(examples["conversation_b"])
    ):
        return False

    # Check that conversations contain valid code blocks with recognized language tags
    if not has_valid_code_blocks(examples["conversation_a"]):
        return False

    # Ensure both assistant responses use the same programming language fences
    _, answer_a = extract_conversation_text(examples["conversation_a"])
    _, answer_b = extract_conversation_text(examples["conversation_b"])
    return responses_share_language(answer_a, answer_b)


def extract_text_from_content(content_list):
    """Extract text from content array structure."""
    if content_list is None or len(content_list) == 0:
        return ""

    # Content is a list of objects with 'type' and 'text' fields
    texts = []
    for item in content_list:
        if item.get("type") == "text" and item.get("text"):
            texts.append(item["text"])

    return "\n".join(texts)


def extract_conversation_text(conversation):
    """Extract user prompt and assistant response from conversation."""
    user_prompt = ""
    assistant_response = ""

    for message in conversation:
        if message["role"] == "user":
            user_prompt = extract_text_from_content(message.get("content", []))
        elif message["role"] == "assistant":
            assistant_response = extract_text_from_content(message.get("content", []))

    return user_prompt, assistant_response


def extract_code_languages(text: str) -> set:
    """Extract valid code language tags from fenced code blocks in the text."""
    if not text:
        return set()
    languages = set()
    for lang in find_code_block_languages(text):
        candidate = lang.strip()
        if is_code_language(candidate):
            languages.add(candidate)
    return languages


def responses_share_language(answer_a: str, answer_b: str) -> bool:
    """
    Ensure both assistant responses contain a code fence and at least one shared programming language.
    """
    langs_a = extract_code_languages(answer_a)
    langs_b = extract_code_languages(answer_b)
    if not langs_a or not langs_b:
        return False
    return bool(langs_a & langs_b)


def map_winner(winner):
    """Map winner field to expected format."""
    if winner == "model_a":
        return "A"
    elif winner == "model_b":
        return "B"
    else:
        raise ValueError("Unexpected winner value: {}".format(winner))


def load_kept_question_ids(csv_path=None):
    """
    Load question IDs to keep from a CSV file with a question_id column.
    If the path is None or the file is missing, return None so callers fall back to
    the original unfiltered behavior.
    """
    if csv_path is None:
        return None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if "question_id" not in df.columns:
        raise ValueError(f"Expected question_id column in {csv_path}")

    return set(df["question_id"].dropna().astype(str))


def filter_dataset_by_question_ids(dataset, kept_question_ids):
    """Optional filter to keep only rows whose id is in kept_question_ids."""
    if not kept_question_ids:
        return dataset

    kept_question_ids = {str(qid) for qid in kept_question_ids}
    return dataset.filter(lambda row: str(row["id"]) in kept_question_ids)


def is_edit_like_prompt(text: str) -> bool:
    """Heuristic for completion/edit style prompts."""
    if NEGATIVE_PATTERN.search(text):
        return False
    if any(phrase in text.lower() for phrase in EXPLAIN_PHRASES):
        return False
    has_edit = bool(EDIT_PATTERN.search(text))
    has_placeholder = bool(PLACEHOLDER_PATTERN.search(text))
    return has_edit or has_placeholder


def is_code_like_response(text: str) -> bool:
    """
    Check if assistant response has a code fence with an acceptable language tag
    and contains code-like tokens.
    """
    if "```" not in text:
        return False
    languages = find_code_block_languages(text)
    if not (languages and any(is_code_language(lang) for lang in languages)):
        return False
    # require code-y tokens to avoid pure prose fences
    return bool(CODE_KW_PATTERN.search(text))


def edit_filter_fn(example):
    """Filter for edit/complete style prompts with code-like assistant answers."""
    if not (
        example["is_code"] == True
        and (example["winner"] == "model_a" or example["winner"] == "model_b")
    ):
        return False

    # Only keep evaluation_order == 1
    if example.get("evaluation_order") != 1:
        return False

    if not (
        is_single_turn(example["conversation_a"])
        and is_single_turn(example["conversation_b"])
    ):
        return False

    user_prompt_a, assistant_response_a = extract_conversation_text(
        example["conversation_a"]
    )
    _, assistant_response_b = extract_conversation_text(example["conversation_b"])

    if not is_edit_like_prompt(user_prompt_a):
        return False
    if not (
        is_code_like_response(assistant_response_a)
        and is_code_like_response(assistant_response_b)
    ):
        return False
    if not responses_share_language(assistant_response_a, assistant_response_b):
        return False

    return True


def load_lmarena():
    data = pd.read_csv("data/arena-human-preference-140k_sample.csv")
    kept_ids = pd.read_csv("data/kept_question_ids.csv")
    data = data[data['question_id'].isin(kept_ids['question_id'])]
    return data

    # data = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    # data = data.shuffle(seed=13)
    # data = data.filter(filter_fn)

    # # Convert to pandas and process
    # df = data.to_pandas()

    # # Extract user prompts and responses from conversation structures
    # processed_rows = []
    # for idx, row in df.iterrows():
    #     user_prompt_a, answer_a = extract_conversation_text(row["conversation_a"])
    #     user_prompt_b, answer_b = extract_conversation_text(row["conversation_b"])

    #     # Both conversations should have the same user prompt
    #     processed_rows.append(
    #         {
    #             "question_id": row["id"],
    #             "user_instruction": user_prompt_a,  # or user_prompt_b, they should be the same
    #             "answer_a": answer_a,
    #             "answer_b": answer_b,
    #             "winner": map_winner(row["winner"]),
    #         }
    #     )

    # # Create new dataframe with processed data
    # processed_df = pd.DataFrame(processed_rows)

    # return processed_df


def load_lmarena_edit_like(kept_question_ids_path=None):
    """
    Load a stricter subset focused on completion/edit style prompts with code-like answers.
    """
    data = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    data = data.shuffle(seed=13)
    data = data.filter(edit_filter_fn, load_from_cache_file=False)
    kept_question_ids = load_kept_question_ids(kept_question_ids_path)
    data = filter_dataset_by_question_ids(data, kept_question_ids)

    df = data.to_pandas()

    processed_rows = []
    for idx, row in df.iterrows():
        user_prompt_a, answer_a = extract_conversation_text(row["conversation_a"])
        user_prompt_b, answer_b = extract_conversation_text(row["conversation_b"])

        processed_rows.append(
            {
                "question_id": row["id"],
                "user_instruction": user_prompt_a,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "winner": map_winner(row["winner"]),
            }
        )

    processed_df = pd.DataFrame(processed_rows)
    return processed_df