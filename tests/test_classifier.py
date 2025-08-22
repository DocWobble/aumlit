from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from classifier import parse_reason


def test_error_rules():
    rules_path = Path(__file__).parent.parent / "rules" / "error_rules.yaml"
    cases = yaml.safe_load(rules_path.read_text())
    for case in cases:
        msg = case["msg"]
        expected = case["update"]
        assert parse_reason(msg) == expected
