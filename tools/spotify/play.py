"""
tools/spotify/play.py
=====================
Implements the spotify.play intent.

Action: Opens Spotify Desktop App explicitly targeting a song, causing auto-play.
Strategy: Constructs a search query, hits DuckDuckGo HTML parser for the first 
valid open.spotify.com/track/{ID} signature, and passes `spotify:track:{ID}` 
to the operating system to auto-start playback.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import webbrowser
from typing import Any

from core.models import ToolResult
from tools.base import BaseTool, SlotField


class SpotifyPlayTool(BaseTool):

    @property
    def name(self) -> str:
        return "spotify.play"

    @property
    def description(self) -> str:
        return "Play a song, artist, album, or playlist on Spotify using direct URI protocols."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="query",
                type_hint="string",
                required=True,
                description="The song title, artist name, album, or playlist to play.",
            ),
            SlotField(
                name="artist",
                type_hint="string",
                required=False,
                description="Artist name to narrow the search.",
                default=None,
            ),
            SlotField(
                name="album",
                type_hint="string",
                required=False,
                description="Album name to narrow the search.",
                default=None,
            ),
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)

        query: str = str(slots["query"])
        artist: str | None = slots.get("artist")
        album: str | None = slots.get("album")

        # Build a human-readable display string for TTS
        display_parts = [query]
        if artist:
            display_parts.append(f"by {artist}")
        if album:
            display_parts.append(f"from {album}")
        display = " ".join(display_parts)

        # Build the Spotify search term
        search_term = query
        if artist:
            search_term = f"{artist} {query}"
        if album:
            search_term = f"{search_term} {album}"

        # 1. Scrape to fetch direct Track ID for auto-play hook
        target_url = None
        search_engine_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote('site:open.spotify.com/track ' + search_term)}"
        
        try:
            req = urllib.request.Request(search_engine_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            html = urllib.request.urlopen(req, timeout=5.0).read().decode('utf-8')
            # Look for the first track link
            matches = re.findall(r"open\.spotify\.com/track/([a-zA-Z0-9]+)", html)
            if matches:
                # Fallback to web browser since native OS desktop client isn't installed
                target_url = f"https://open.spotify.com/track/{matches[0]}?autoplay=true"
        except Exception as e:
            log.warning("spotify.play.scrape_failed", error=str(e))

        # 2. Fallback to basic search hook if scraper is blocked
        if not target_url:
            log.warning("spotify.play.fallback_to_search")
            encoded = urllib.parse.quote_plus(search_term)
            target_url = f"https://open.spotify.com/search/{encoded}"

        log.info("spotify.play.opening", url=target_url, display=display)
        opened = webbrowser.open(target_url)

        if not opened:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I couldn't open Spotify. Please check your system settings.",
                error_code="BROWSER_OPEN_FAILED",
                duration_ms=0,
            )

        action_word = "Playing" if "open.spotify.com/track/" in target_url else "Searching"

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=f"{action_word} {display} on Spotify.",
            data={"url": target_url, "query": query, "artist": artist, "album": album},
            duration_ms=0,
        )

    async def _dry_run_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        query: str = str(slots["query"])
        artist: str | None = slots.get("artist")
        
        display_parts = [query]
        if artist:
            display_parts.append(f"by {artist}")
        display = " ".join(display_parts)

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            simulated=True,
            message=f"Simulation: Right now, I would open Spotify to play {display}.",
            duration_ms=0,
        )
