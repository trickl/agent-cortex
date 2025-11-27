import json

from llmflow.core.agent import _prepare_tool_result_content


def _serialize(payload):
    return json.dumps(payload, ensure_ascii=False)


def test_prepare_tool_result_content_trims_qlty_payload():
    payload = {
        "success": True,
        "data": [
            {"id": "issue-1", "attributes": {"title": "Unused import"}},
            {"id": "issue-2", "attributes": {"title": "Complexity"}},
            {"id": "issue-3", "attributes": {"title": "Bandit"}},
        ],
        "meta": {"hasMore": True},
        "source": "https://api.qlty.sh/example",
    }

    trimmed_str = _prepare_tool_result_content("qlty_list_issues", _serialize(payload))
    trimmed = json.loads(trimmed_str)

    assert trimmed["issue_count"] == 3
    assert len(trimmed["data"]) == 1
    assert trimmed["data_truncated"] is True
    assert trimmed["data"][0]["id"] == "issue-1"
    assert trimmed["additional_issue_ids"] == ["issue-2", "issue-3"]
    assert "Trimmed" in trimmed["note"]


def test_prepare_tool_result_content_handles_no_issues():
    payload = {"success": True, "data": [], "meta": {}}

    trimmed_str = _prepare_tool_result_content("qlty_list_issues", _serialize(payload))
    trimmed = json.loads(trimmed_str)

    assert trimmed["issue_count"] == 0
    assert trimmed["data"] == []
    assert trimmed["data_truncated"] is False
    assert "no issues" in trimmed["note"].lower()


def test_prepare_tool_result_content_passthrough_for_other_tools():
    original = _serialize({"success": True, "value": 5})

    result = _prepare_tool_result_content("other_tool", original)

    assert result == original
