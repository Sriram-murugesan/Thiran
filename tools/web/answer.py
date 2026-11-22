"""
tools/web/answer.py
===================
Implements web.answer intent for answering conversational questions.

Strategy: Uses DuckDuckGo to pull top text snippets and passes them to
the Groq LLM to yield a concise spoken answer.
"""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp
from ddgs import DDGS

from core.models import ToolResult
from tools.base import BaseTool, SlotField


class WebAnswerTool(BaseTool):

    @property
    def name(self) -> str:
        return "web.answer"

    @property
    def description(self) -> str:
        return "Search the web to answer general knowledge questions. Returns a concise spoken answer."

    @property
    def slot_fields(self) -> list[SlotField]:
        return [
            SlotField(
                name="query",
                type_hint="string",
                required=True,
                description="The direct question to answer.",
            ),
        ]

    async def _execute_impl(
        self,
        slots: dict[str, Any],
        request_id: str,
    ) -> ToolResult:
        log = self._log(request_id)
        query: str = str(slots["query"])

        log.info("web.answer.searching", query=query)

        try:
            # DuckDuckGo search is slightly blocking, but usually very fast.
            # Running inside an executor is better for strict non-blocking loops,
            # but for a quick script it works well.
            import asyncio
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, lambda: list(DDGS().text(query, max_results=3)))
        except Exception as e:
            log.error("web.answer.search_error", error=str(e))
            results = []

        if not results:
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I couldn't find an answer for that right now.",
                error_code="SEARCH_FAILED",
                duration_ms=0,
            )

        snippets = []
        for r in results:
            if "body" in r:
                snippets.append(r["body"])
        
        context_text = "\n".join(snippets)

        log.info("web.answer.summarizing")
        try:
            summary = await self._summarize(query, context_text)
            if not summary:
                raise ValueError("Empty summary from LLM.")
        except Exception as e:
            log.error("web.answer.summarize_error", error=str(e))
            return ToolResult(
                request_id=request_id,
                tool_name=self.name,
                success=False,
                message="I found some results, but had trouble summarizing them.",
                error_code="SUMMARIZE_FAILED",
                duration_ms=0,
            )

        return ToolResult(
            request_id=request_id,
            tool_name=self.name,
            success=True,
            message=summary,
            data={"query": query, "snippets": snippets},
            duration_ms=0,
        )

    async def _summarize(self, query: str, context: str) -> str:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "I need a Groq API key to summarize search results."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        system_prompt = (
            "You are a helpful voice assistant. Answer the user's question based "
            "ONLY on the provided search snippets. Keep your answer highly concise, "
            "conversational, and easy to speak out loud (max 1 or 2 sentences). "
            "Do not use markdown, emojis, or URLs. If the snippets don't contain the answer, "
            "just say you don't know based on the search."
        )

        user_content = f"Question: {query}\n\nSearch Snippets:\n{context}"

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
            "max_tokens": 150,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"].strip()
