"""Tools for interacting with the Qlty REST API."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

from llmflow.tools.tool_decorator import register_tool

_DEFAULT_BASE_URL = "https://api.qlty.sh"
_DEFAULT_PAGE_LIMIT = 50
_TOOL_TAGS = ["qlty", "issues", "project_management"]
_SINGLE_ISSUE_TAGS = _TOOL_TAGS + ["qlty_single_issue"]


def _failure(message: str, status_code: Optional[int] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"success": False, "error": message}
    if status_code is not None:
        payload["status_code"] = status_code
    return payload


def _dedupe(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    ordered: List[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered or None


def _resolve_base_url(base_url: Optional[str]) -> str:
    return (base_url or os.getenv("QLTY_API_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")


def _resolve_token(token: Optional[str], token_env_var: str) -> Optional[str]:
    if token:
        return token
    return os.getenv(token_env_var)


def _validate_token_charset(token: str) -> Optional[str]:
    try:
        token.encode("latin-1")
        return None
    except UnicodeEncodeError as exc:
        snippet = token[max(exc.start - 3, 0) : exc.end + 3]
        return (
            "Qlty API token contains unsupported characters. Ensure you copied the full token "
            "without smart quotes or ellipsis (problem segment: "
            f"{snippet!r})."
        )


def _build_filter_query(
    categories: Optional[Sequence[str]],
    levels: Optional[Sequence[str]],
    statuses: Optional[Sequence[str]],
    tools: Optional[Sequence[str]],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if (deduped := _dedupe(categories)):
        params["category"] = json.dumps(deduped)
    if (deduped := _dedupe(levels)):
        params["level"] = json.dumps(deduped)
    if (deduped := _dedupe(statuses)):
        params["status"] = json.dumps(deduped)
    if (deduped := _dedupe(tools)):
        params["tool"] = json.dumps(deduped)
    return params


def _decode_response_content(content: bytes, preferred_encoding: Optional[str]) -> str:
    if not content:
        return ""

    candidates = []
    if preferred_encoding:
        candidates.append(preferred_encoding)
    candidates.extend(["utf-8", "latin-1"])

    for encoding in candidates:
        if not encoding:
            continue
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    return content.decode("utf-8", errors="replace")


def _request_page(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    timeout: float,
    max_retries: int,
    retry_delay: float,
) -> Dict[str, Any]:
    attempt = 0
    backoff = retry_delay
    while True:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            response.raise_for_status()

            preferred_encoding = response.encoding
            if preferred_encoding:
                preferred_encoding = preferred_encoding.lower()
            if not preferred_encoding or preferred_encoding in {"", "latin-1", "iso-8859-1"}:
                preferred_encoding = "utf-8"

            body_text = _decode_response_content(response.content, preferred_encoding)
            try:
                return json.loads(body_text)
            except json.JSONDecodeError as exc:
                snippet = body_text[:200]
                raise ValueError(
                    f"Failed to parse Qlty API response: {exc.msg} (pos {exc.pos}). Snippet: {snippet}"
                ) from exc
        except requests.RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2


@register_tool(tags=_TOOL_TAGS)
def qlty_list_issues(
    owner_key_or_id: str,
    project_key_or_id: str,
    categories: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[str]] = None,
    statuses: Optional[Sequence[str]] = None,
    tools: Optional[Sequence[str]] = None,
    page_limit: Optional[int] = None,
    page_offset: Optional[int] = None,
    auto_paginate: bool = False,
    max_pages: int = 5,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    token_env_var: str = "QLTY_API_TOKEN",
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    include_raw_pages: bool = False,
) -> Dict[str, Any]:
    """Retrieve a list of Qlty issues for a project.

    Args:
        owner_key_or_id: Workspace key/ID that owns the project.
        project_key_or_id: Project key/ID to pull issues from.
        categories: Optional category filters (e.g., "lint", "vulnerability").
        levels: Optional severity filters (e.g., "low", "medium", "high").
        statuses: Optional status filters (defaults to open issues when omitted).
        tools: Optional list of tool identifiers (e.g., "eslint", "typescript").
        page_limit: Page size for each request (defaults to 50 when auto_paginating).
        page_offset: Starting offset for the first request.
        auto_paginate: When True, keep fetching until the API reports no more data
            or until ``max_pages`` has been reached. Totals are aggregated.
        max_pages: Maximum number of API pages to fetch while auto-paginating.
        base_url: Override the Qlty API base URL (defaults to https://api.qlty.sh or
            ``QLTY_API_BASE_URL`` env var).
        token: Provide a Qlty API token directly. When omitted, ``token_env_var``
            is consulted.
        token_env_var: Environment variable to read the token from when ``token``
            is not supplied. Defaults to ``QLTY_API_TOKEN``.
        timeout: Request timeout in seconds.
        max_retries: Number of times to retry transient failures before giving up.
        retry_delay: Base delay (seconds) between retries; doubled on each attempt.
        include_raw_pages: When True, include the raw page payloads for debugging.
    """

    resolved_token = _resolve_token(token, token_env_var)
    if not resolved_token:
        return _failure(
            "Missing Qlty API token. Provide the 'token' argument or set the "
            f"environment variable '{token_env_var}'."
        )

    charset_error = _validate_token_charset(resolved_token)
    if charset_error:
        return _failure(charset_error)

    resolved_base = _resolve_base_url(base_url)
    url = f"{resolved_base}/gh/{owner_key_or_id}/projects/{project_key_or_id}/issues"

    headers = {
        "Authorization": f"Bearer {resolved_token}",
        "Accept": "application/json",
    }

    params = _build_filter_query(categories, levels, statuses, tools)
    if page_limit is not None:
        params["page[limit]"] = page_limit
    if page_offset is not None:
        params["page[offset]"] = page_offset

    effective_limit = page_limit or _DEFAULT_PAGE_LIMIT
    results: List[Any] = []
    raw_pages: List[Dict[str, Any]] = []
    pages_fetched = 0
    offset = page_offset or 0
    meta: Dict[str, Any] = {}

    try:
        while True:
            page_params = dict(params)
            if auto_paginate and "page[limit]" not in page_params:
                page_params["page[limit]"] = effective_limit
            if offset:
                page_params["page[offset]"] = offset

            payload = _request_page(
                url,
                headers,
                page_params,
                timeout,
                max_retries,
                retry_delay,
            )
            data = payload.get("data") or []
            meta = payload.get("meta") or {}
            results.extend(data)
            if include_raw_pages:
                raw_pages.append({
                    "params": page_params,
                    "data": data,
                    "meta": meta,
                })

            pages_fetched += 1
            if not auto_paginate:
                break

            has_more = bool(meta.get("hasMore"))
            if not has_more:
                break

            if max_pages and pages_fetched >= max_pages:
                break

            increment = page_params.get("page[limit]") or len(data) or effective_limit
            offset = (offset or 0) + int(increment)

        response_payload: Dict[str, Any] = {
            "success": True,
            "data": results,
            "meta": meta,
            "pages_fetched": pages_fetched,
            "source": url,
        }
        if include_raw_pages:
            response_payload["raw_pages"] = raw_pages
        return response_payload
    except requests.HTTPError as exc:  # pragma: no cover - exercised in tests via mock
        status = exc.response.status_code if exc.response is not None else None
        detail = None
        try:
            if exc.response is not None:
                detail = exc.response.json()
        except ValueError:
            detail = exc.response.text if exc.response is not None else None
        message = str(exc)
        if detail:
            message = f"{message} | Details: {detail}"
        return _failure(message, status_code=status)
    except requests.RequestException as exc:
        return _failure(f"Failed to reach Qlty API: {exc}")
    except ValueError as exc:
        return _failure(f"Unexpected Qlty API response: {exc}")


@register_tool(tags=_SINGLE_ISSUE_TAGS)
def qlty_get_first_issue(
    owner_key_or_id: str,
    project_key_or_id: str,
    categories: Optional[Sequence[str]] = None,
    statuses: Optional[Sequence[str]] = None,
    levels: Optional[Sequence[str]] = None,
    tools: Optional[Sequence[str]] = None,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    token_env_var: str = "QLTY_API_TOKEN",
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """Return only the first Qlty issue that matches the provided filters.

    This tool wraps :func:`qlty_list_issues` but enforces a single-item response so the
    agent never needs to ingest the full issue list into the LLM context. By default
    it targets open lint issues, which keeps the integration tests focused on the
    appropriate category while still allowing overrides when needed.
    """

    effective_categories = list(categories) if categories is not None else ["lint"]
    effective_statuses = list(statuses) if statuses is not None else ["open"]

    list_response = qlty_list_issues(
        owner_key_or_id=owner_key_or_id,
        project_key_or_id=project_key_or_id,
        categories=effective_categories,
        levels=levels,
        statuses=effective_statuses,
        tools=tools,
        page_limit=1,
        page_offset=None,
        auto_paginate=False,
        base_url=base_url,
        token=token,
        token_env_var=token_env_var,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    if not list_response.get("success"):
        return list_response

    issues = list_response.get("data") or []
    first_issue = issues[0] if issues else None

    result: Dict[str, Any] = {
        "success": True,
        "issue": first_issue,
        "issue_found": bool(first_issue),
        "filters": {
            "owner": owner_key_or_id,
            "project": project_key_or_id,
            "categories": effective_categories,
            "statuses": effective_statuses,
            "levels": list(levels) if levels is not None else None,
            "tools": list(tools) if tools is not None else None,
        },
        "meta": list_response.get("meta"),
        "source": list_response.get("source"),
    }

    if not first_issue:
        result["message"] = "No issues matched the provided filters."

    return result
