from scripts.judge import parse_json_object


def test_parse_plain_json():
    result = parse_json_object(
        '{"winner":"A","factual_error_A":false,"factual_error_B":true,"reason":"A is faithful"}'
    )
    assert result["winner"] == "A"


def test_parse_fenced_json():
    result = parse_json_object(
        '```json\n{"winner":"tie","factual_error_A":false,"factual_error_B":false,"reason":"similar"}\n```'
    )
    assert result["winner"] == "tie"

