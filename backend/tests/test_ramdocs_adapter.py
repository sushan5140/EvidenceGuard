import json

from app.research.ramdocs import load_ramdocs


def test_load_ramdocs_format(tmp_path):
    path = tmp_path / "ramdocs.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "What is the capital?",
                "documents": [
                    {"text": "Canberra is the capital of Australia.", "type": "correct", "answer": "Canberra"},
                    {"text": "Sydney is the capital of Australia.", "type": "misinfo", "answer": "Sydney"},
                ],
                "gold_answers": ["Canberra"],
                "wrong_answers": ["Sydney"],
                "disambig_entity": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cases = load_ramdocs(path)
    assert len(cases) == 1
    assert cases[0].gold_answers == ["Canberra"]
    assert cases[0].documents[1]["type"] == "misinfo"
