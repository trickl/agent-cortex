from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from llmflow.tools import tool_qlty


class DummyResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.encoding = "utf-8"
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            error = requests.HTTPError(f"status {self.status_code}")
            error.response = self
            raise error

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_qlty_list_issues_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QLTY_API_TOKEN", raising=False)
    result = tool_qlty.qlty_list_issues("acmeco", "awesome-project")
    assert result["success"] is False
    assert "token" in result["error"].lower()


def test_qlty_list_issues_rejects_non_ascii_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLTY_API_TOKEN", "qlty…truncated")
    result = tool_qlty.qlty_list_issues("acmeco", "awesome-project")
    assert result["success"] is False
    assert "unsupported" in result["error"].lower()


def test_qlty_list_issues_auto_paginate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLTY_API_TOKEN", "token123")

    responses: List[Dict[str, Any]] = [
        {"data": [{"id": "1"}], "meta": {"hasMore": True}},
        {"data": [{"id": "2"}], "meta": {"hasMore": False}},
    ]
    calls: List[Dict[str, Any]] = []

    def fake_get(url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: float):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return DummyResponse(responses.pop(0))

    monkeypatch.setattr(tool_qlty.requests, "get", fake_get)

    result = tool_qlty.qlty_list_issues(
        owner_key_or_id="acmeco",
        project_key_or_id="awesome-project",
        auto_paginate=True,
        page_limit=1,
        include_raw_pages=True,
    )

    assert result["success"] is True
    assert result["pages_fetched"] == 2
    assert len(result["data"]) == 2
    assert result["meta"].get("hasMore") is False
    assert len(result["raw_pages"]) == 2
    assert calls[0]["params"].get("page[offset]") is None
    assert calls[1]["params"].get("page[offset]") == 1


def test_request_page_handles_utf8_content_when_server_claims_latin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLTY_API_TOKEN", "token123")

    payload = {"data": [{"id": "1", "title": "Lint issue with ellipsis …"}], "meta": {}}

    class WeirdResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.encoding = "latin-1"
            self.content = json.dumps(payload).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: float):
        return WeirdResponse()

    monkeypatch.setattr(tool_qlty.requests, "get", fake_get)

    result = tool_qlty._request_page(
        url="https://api.qlty.sh/test",
        headers={},
        params={},
        timeout=30.0,
        max_retries=0,
        retry_delay=0.1,
    )

    assert result["data"][0]["title"].endswith("…")


def test_qlty_get_first_issue_returns_single_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_list(owner_key_or_id: str, project_key_or_id: str, **kwargs: Any) -> Dict[str, Any]:
        captured["owner"] = owner_key_or_id
        captured["project"] = project_key_or_id
        captured.update(kwargs)
        return {
            "success": True,
            "data": [
                {"id": "issue-1", "attributes": {"title": "Lint error"}},
                {"id": "issue-2", "attributes": {"title": "Other"}},
            ],
            "meta": {"hasMore": True},
            "source": "https://api.qlty.sh/test",
        }

    monkeypatch.setattr(tool_qlty, "qlty_list_issues", fake_list)

    result = tool_qlty.qlty_get_first_issue("acme", "project")

    assert result["success"] is True
    assert result["issue"]["id"] == "issue-1"
    assert result["issue_found"] is True
    assert captured["categories"] == ["lint"]
    assert captured["statuses"] == ["open"]
    assert captured["page_limit"] == 1
    assert captured["auto_paginate"] is False


def test_qlty_get_first_issue_handles_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_list(owner_key_or_id: str, project_key_or_id: str, **kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "data": [], "meta": {}, "source": "src"}

    monkeypatch.setattr(tool_qlty, "qlty_list_issues", fake_list)

    result = tool_qlty.qlty_get_first_issue("acme", "project", categories=["lint"], statuses=["open"])  # overrides respected

    assert result["success"] is True
    assert result["issue"] is None
    assert result["issue_found"] is False
    assert "no issues" in result["message"].lower()


def test_qlty_get_first_issue_propagates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_list(owner_key_or_id: str, project_key_or_id: str, **kwargs: Any) -> Dict[str, Any]:
        return {"success": False, "error": "bad token"}

    monkeypatch.setattr(tool_qlty, "qlty_list_issues", fake_list)

    result = tool_qlty.qlty_get_first_issue("acme", "project")

    assert result["success"] is False
    assert result["error"] == "bad token"