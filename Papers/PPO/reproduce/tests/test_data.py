from ppo_repro.data import (
    canonicalize_comparison_row,
    content_hash,
    format_prompt,
    parse_tldr_prompt,
    stable_group_select,
)


def test_prompt_round_trip():
    prompt = format_prompt("relationships", "A title", "A post\nwith lines")
    parsed = parse_tldr_prompt(prompt)
    assert parsed == {"subreddit": "relationships", "title": "A title", "post": "A post\nwith lines"}


def test_hash_normalizes_whitespace_and_case():
    assert content_hash(" Title ", "A\n post") == content_hash("title", "a post")


def test_comparison_choice_controls_direction():
    raw = {
        "info": {"id": "p1", "post": "post", "title": "title", "subreddit": "x"},
        "summaries": [
            {"text": "bad", "policy": "a"},
            {"text": "good", "policy": "b"},
        ],
        "choice": 1,
        "split": "train",
        "batch": "batch3",
    }
    row = canonicalize_comparison_row(raw)
    assert row["chosen"].strip() == "good"
    assert row["rejected"].strip() == "bad"


def test_group_selection_never_splits_post():
    rows = [
        {"content_hash": "a", "value": 1},
        {"content_hash": "a", "value": 2},
        {"content_hash": "b", "value": 3},
        {"content_hash": "c", "value": 4},
    ]
    selected = stable_group_select(rows, size=3, seed=42)
    for group in {row["content_hash"] for row in selected}:
        assert sum(row["content_hash"] == group for row in selected) == sum(
            row["content_hash"] == group for row in rows
        )

