"""Helper utilities shared across Agent implementations."""

import json
from typing import Any, Dict, Optional


def _prepare_tool_result_content(tool_name: Optional[str], content: str) -> str:
    """Trim large tool payloads before adding them to conversation history."""

    if tool_name != "qlty_list_issues":
        return content
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content
    data = payload.get("data")
    if not isinstance(data, list):
        return content

    issue_count = len(data)
    trimmed_data = data[:1] if issue_count else []
    additional_ids = [
        issue.get("id")
        for issue in data[1:]
        if isinstance(issue, dict) and issue.get("id")
    ]
    summary: Dict[str, Any] = {
        "issue_count": issue_count,
        "data": trimmed_data,
        "data_truncated": issue_count > len(trimmed_data),
        "additional_issue_ids": additional_ids,
    }
    if issue_count == 0:
        summary["note"] = "Qlty API returned no issues for the requested filters."
    elif issue_count > 1:
        summary["note"] = (
            "Trimmed Qlty response to the first issue to keep the context concise."
        )
    else:
        summary["note"] = "Qlty API returned a single issue."

    meta = payload.get("meta")
    if isinstance(meta, dict) and meta:
        summary["meta"] = meta
    source = payload.get("source")
    if isinstance(source, str) and source.strip():
        summary["source"] = source

    return json.dumps(summary, ensure_ascii=False)
