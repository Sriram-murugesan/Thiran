"""
tools/web/search.py
===================
Implements web.search intent.

Strategy: Opens the default browser with the search engine's query URL.
Supports Google and DuckDuckGo (both privacy-respecting fallbacks available).
"""

from __future__ import annotations

import urllib.parse
import webbrowser
from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField

_SEARCH_URLS: dict[str, str] = {
    "google": "https://www.google.com/search?q={}",
    "duckduckgo": "https://duckduckgo.com/?q={}",
}


class WebSearchTool(BaseTool):

    @property
    def name(self) -> str:
        return "web.search"

    @property
    def description(self) -> str:
        return "Search the web using Google or DuckDuckGo and open results in the browser."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="query",
                type_hint="string",
                required=True,
                description="The search query to look up.",
            ),
            SlotField(
                name="engine",
                type_hint="string",
                required=False,
                description="Which search engine to use.",
                default="google",
                allowed_values=["google", "duckduckgo"],
            ),
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)
        query: str = str(slots["query"])
        engine: str = str(slots.get("engine", "google")).lower()

        url_template = _SEARCH_URLS.get(engine, _SEARCH_URLS["google"])
        url = url_template.format(urllib.parse.quote_plus(query))

        engine_label = engine.capitalize()
        log.info("web.search.open", engine=engine, url=url)

        opened = webbrowser.open(url)

        if not opened:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I couldn't open the browser. Please check your browser settings.",
                error_code="BROWSER_OPEN_FAILED",
                duration_ms=0,
            )

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=f"Searching {engine_label} for: {query}.",
            data={"query": query, "engine": engine, "url": url},
            duration_ms=0,
        )
