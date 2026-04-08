"""
tools/youtube/play.py
=====================
Implements youtube.play and youtube.search intents.

Strategy: 
- youtube.search: Opens YouTube in the default browser with a search URL.
- youtube.play: Scrapes YouTube search results to locate the first video ID
  and opens its direct watch URL in the default browser to trigger autoplay.
"""

from __future__ import annotations

import re
import subprocess
import urllib.parse
import urllib.request
from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField


def _open_brave(url: str) -> bool:
    """Open a URL in the Brave browser. Returns True on success."""
    try:
        subprocess.Popen(
            ["brave-browser", "--new-window", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        return False


class _YouTubeBaseTool(BaseTool):
    """Shared URL-building logic for YouTube tools."""

    _BASE_URL = "https://www.youtube.com/results"

    def _build_url(self, query: str) -> str:
        params = urllib.parse.urlencode({"search_query": query})
        return f"{self._BASE_URL}?{params}"

    async def _open_youtube(
        self,
        query: str,
        request_id: str,
        action_label: str,
    ) -> ToolResult:
        log = self._log(request_id)
        url = self._build_url(query)
        log.info("youtube.open", action=action_label, url=url)

        opened = _open_brave(url)

        if not opened:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I couldn't open YouTube. Please check your browser settings.",
                error_code="BROWSER_OPEN_FAILED",
                duration_ms=0,
            )

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=f"{action_label} YouTube for: {query}.",
            data={"url": url, "query": query},
            duration_ms=0,
        )


class YouTubePlayTool(_YouTubeBaseTool):

    @property
    def name(self) -> str:
        return "youtube.play"

    @property
    def description(self) -> str:
        return "Autoplay a YouTube video by finding the best match and launching it directly."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="query",
                type_hint="string",
                required=True,
                description="The video title, channel, or topic to play.",
            )
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)
        query = str(slots["query"])
        
        # Scrape to find direct video URL for autoplay
        search_url = self._build_url(query)
        target_url = None
        
        try:
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urllib.request.urlopen(req, timeout=5.0).read().decode('utf-8')
            video_ids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", html)
            if video_ids:
                target_url = f"https://www.youtube.com/watch?v={video_ids[0]}&autoplay=1"
        except Exception as e:
            log.warning("youtube.play.scrape_failed", error=str(e))
            
        # Fallback to search-page if parsing fails
        if not target_url:
            log.warning("youtube.play.fallback_to_search")
            return await self._open_youtube(
                query=query,
                request_id=request_id,
                action_label="Playing (Fallback)",
            )

        log.info("youtube.play.opening_direct", url=target_url, query=query)
        opened = _open_brave(target_url)

        if not opened:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I couldn't open YouTube. Please check your browser settings.",
                error_code="BROWSER_OPEN_FAILED",
                duration_ms=0,
            )

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=f"Playing {query} on YouTube.",
            data={"url": target_url, "query": query},
            duration_ms=0,
        )


class YouTubeSearchTool(_YouTubeBaseTool):

    @property
    def name(self) -> str:
        return "youtube.search"

    @property
    def description(self) -> str:
        return "Search YouTube and open results in the browser."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="query",
                type_hint="string",
                required=True,
                description="The search term to look up on YouTube.",
            )
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        return await self._open_youtube(
            query=str(slots["query"]),
            request_id=request_id,
            action_label="Searching",
        )
