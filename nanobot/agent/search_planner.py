"""Search-Enhanced Planning module (net-planner.md Module 8).

Implements a pre-planning web search augmentation layer that wraps TaskPlanner:

Sub-module 8.1 - SearchDecider:    LLM-based decision whether to search (no tools)
Sub-module 8.2 - SearchAgent:      Agentic search loop using AgentRunner + web tools,
                                    letting the LLM decide how to call web_search / web_fetch
Sub-module (inline) - plan/replan: Injects search results into TaskPlanner context

The chain is fully non-invasive: failures at any stage degrade gracefully to
plain planning without blocking the main flow.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.planner import TaskPlanner
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMProvider


# Tools that must NOT be available to the search agent (side effects / destructive)
# NOTE: "exec" is intentionally NOT excluded here — shell commands like curl/wget
# are the backbone of skills-based web search (e.g. weather, custom search skills).
# The SearchAgent's system prompt restricts it to read-only information gathering.
_SEARCH_AGENT_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "notebook_edit",  # file modification
    "message",                                     # send message to channel
    "spawn",                                       # create subagents
    "cron",                                        # schedule jobs
    "my",                                          # self-modification
})

# Web-search tool names grouped by backend
_WEB_TOOL_NAMES: frozenset[str] = frozenset({"web_search", "web_fetch"})
_BROWSER_TOOL_NAMES: frozenset[str] = frozenset({"browser_search", "browser_fetch"})
_ALL_WEB_SEARCH_TOOLS: frozenset[str] = _WEB_TOOL_NAMES | _BROWSER_TOOL_NAMES


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchEnhancedPlanResult:
    """Carries the outcome of the full search-enhanced planning pipeline."""

    plan_text: str | None               # Final plan (None if planning failed)
    search_triggered: bool = False      # Whether the search decision was TRIGGER
    search_keywords: list[str] = field(default_factory=list)
    purified_info: str | None = None    # Search agent output injected into plan; None if skipped/failed
    search_failed: bool = False         # True when search was triggered but failed (circuit broken)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SearchEnhancedPlanner:
    """Wraps TaskPlanner with pre-planning web search augmentation.

    The search step uses a small agentic loop (AgentRunner + web tools)
    rather than directly invoking WebSearchTool — the LLM autonomously
    decides which queries to run and how to synthesise the results.

    Usage::

        sep = SearchEnhancedPlanner(planner=planner, ...)
        result = await sep.plan(task="...", progress_callback=bus_fn)
        plan_text = result.plan_text
    """

    def __init__(
        self,
        planner: "TaskPlanner",
        provider: "LLMProvider",
        model: str,
        max_tool_result_chars: int,
        tools: "ToolRegistry",
        max_results: int = 5,
        timeout: int = 30,
        max_purified_chars: int = 2000,
        search_on_replan: bool = False,
        web_search_backend: str = "auto",
    ) -> None:
        from nanobot.agent.runner import AgentRunner

        self._planner = planner
        self._provider = provider
        self._model = model
        self._max_tool_result_chars = max_tool_result_chars
        self._tools = tools
        self._max_results = max_results
        self._timeout = timeout
        self._max_purified_chars = max_purified_chars
        self._search_on_replan = search_on_replan
        self._web_search_backend = web_search_backend.strip().lower() if web_search_backend else "auto"
        self._runner = AgentRunner(provider)

    # ------------------------------------------------------------------
    # Sub-module 8.1: SearchDecider (pure LLM, no tools)
    # ------------------------------------------------------------------

    async def decide(
        self,
        task: str,
        existing_info: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Decide whether web search is needed for *task*.

        A single no-tool LLM call reads the task and returns either
        ``TRIGGER: kw1, kw2`` or ``SKIP``.

        Returns:
            (should_search, keywords) — keywords is empty when should_search=False.
        """
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.runner import AgentRunSpec
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.utils.prompt_templates import render_template

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        system_prompt = render_template(
            "agent/search_decision_system.md",
            time_ctx=time_ctx,
            has_existing_info=bool(existing_info),
        )

        user_content = task
        if existing_info:
            user_content = (
                f"[Already available search info]\n{existing_info}\n\n"
                f"[Task]\n{task}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            result = await self._runner.run(
                AgentRunSpec(
                    initial_messages=messages,
                    tools=ToolRegistry(),   # no tools — pure decision
                    model=self._model,
                    max_iterations=1,
                    max_tool_result_chars=self._max_tool_result_chars,
                    error_message=None,
                )
            )
        except Exception as exc:
            logger.warning("SearchDecider: LLM call failed ({}), defaulting to SKIP", exc)
            return False, []

        if result.stop_reason == "error" or result.final_content is None:
            logger.warning("SearchDecider: no response, defaulting to SKIP")
            return False, []

        raw = result.final_content.strip()
        first_line = raw.split("\n", 1)[0].strip().upper()

        if first_line.startswith("SKIP"):
            logger.info("SearchDecider: decision=SKIP")
            return False, []

        if first_line.startswith("TRIGGER"):
            # Parse keywords from "TRIGGER: kw1, kw2, ..."
            rest = raw.split(":", 1)[1].strip() if ":" in raw else task[:80]
            keywords = [k.strip() for k in rest.split(",") if k.strip()][:5]
            if not keywords:
                keywords = [task[:80]]
            logger.info("SearchDecider: decision=TRIGGER keywords={}", keywords)
            return True, keywords

        # Fallback: scan full body
        if "TRIGGER" in raw.upper():
            keywords = [task[:80]]
            return True, keywords

        logger.info("SearchDecider: parse fallback → SKIP")
        return False, []

    # ------------------------------------------------------------------
    # Sub-module 8.2: SearchAgent (agentic loop with available tools)
    # ------------------------------------------------------------------

    async def _run_search_agent(
        self,
        task: str,
        keywords: list[str],
    ) -> str | None:
        """Run a small agentic loop that gathers information using available tools.

        Uses a read-safe subset of the main agent's tool registry (web_search,
        web_fetch, exec, read_file, glob, grep, MCP tools, etc.).  Destructive
        tools (write_file, message, spawn, cron, my) are excluded.  exec is
        included but its per-command timeout is capped to _EXEC_TIMEOUT_CAP so
        a single hanging curl cannot block the entire loop.

        The LLM decides which queries/fetches to execute and produces a concise,
        task-relevant summary as its final response.

        Returns:
            The agent's final summary string, or None on failure/timeout.
        """
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.runner import AgentRunSpec
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.tools.shell import ExecTool
        from nanobot.utils.prompt_templates import render_template

        # The per-command timeout cap for exec inside SearchAgent.
        # Prevents a single curl/wget without --max-time from consuming the
        # entire SearchAgent budget (default exec timeout is 60 s).
        _EXEC_TIMEOUT_CAP = 150

        # Build a read-safe registry: copy all tools except destructive ones
        safe_tools = ToolRegistry()
        for name in self._tools.tool_names:
            if name not in _SEARCH_AGENT_EXCLUDED_TOOLS:
                tool = self._tools.get(name)
                if tool is None:
                    continue
                # Cap exec timeout so a single hanging curl does not block the
                # entire SearchAgent loop.  Clone the tool with a lower timeout
                # rather than mutating the shared instance.
                if isinstance(tool, ExecTool) and tool.timeout > _EXEC_TIMEOUT_CAP:
                    tool = ExecTool(
                        timeout=_EXEC_TIMEOUT_CAP,
                        working_dir=tool.working_dir,
                        deny_patterns=tool.deny_patterns,
                        allow_patterns=tool.allow_patterns,
                        restrict_to_workspace=tool.restrict_to_workspace,
                        sandbox=tool.sandbox,
                        path_append=tool.path_append,
                        allowed_env_keys=tool.allowed_env_keys,
                    )
                safe_tools.register(tool)

        # Apply web-search backend preference: decide which tool set to expose
        # based on self._web_search_backend ("auto" | "browser" | "web").
        registered = frozenset(safe_tools.tool_names)
        has_browser = bool(registered & _BROWSER_TOOL_NAMES)
        has_web = bool(registered & _WEB_TOOL_NAMES)

        if self._web_search_backend == "browser":
            # Headless-browser tools only
            excluded_web = _WEB_TOOL_NAMES
            backend_used = "browser" if has_browser else "(none — browser tools not registered)"
        elif self._web_search_backend == "web":
            # HTTP-API tools only
            excluded_web = _BROWSER_TOOL_NAMES
            backend_used = "web" if has_web else "(none — web tools not registered)"
        else:
            # auto: prefer browser if available, fall back to web
            if has_browser:
                excluded_web = _WEB_TOOL_NAMES
                backend_used = "browser (auto)"
            else:
                excluded_web = _BROWSER_TOOL_NAMES
                backend_used = "web (auto)"

        # Re-build safe_tools excluding the deselected web tool set
        if excluded_web & registered:  # only if there's actually something to remove
            filtered = ToolRegistry()
            for name in safe_tools.tool_names:
                if name not in excluded_web:
                    t = safe_tools.get(name)
                    if t is not None:
                        filtered.register(t)
            safe_tools = filtered

        logger.info(
            "SearchAgent: web_search_backend={} (config={}, has_browser={}, has_web={})",
            backend_used, self._web_search_backend, has_browser, has_web,
        )

        if len(safe_tools) == 0:
            logger.info("SearchAgent: no tools available in registry, skipping agentic search")
            return None

        available_names = safe_tools.tool_names

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        system_prompt = render_template(
            "agent/search_agent_system.md",
            time_ctx=time_ctx,
            task=task,
            keywords_str=", ".join(keywords),
            max_chars=self._max_purified_chars,
            available_tools=available_names,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Gather information to help plan this task:\n\n{task}\n\n"
                    f"Focus on: {', '.join(keywords)}"
                ),
            },
        ]

        logger.info(
            "SearchAgent: starting (keywords={}, tools={}, timeout={}s)",
            keywords, available_names, self._timeout,
        )

        try:
            result = await asyncio.wait_for(
                self._runner.run(
                    AgentRunSpec(
                        initial_messages=messages,
                        tools=safe_tools,
                        model=self._model,
                        max_iterations=8,           # small agentic loop — enough for 2–4 searches
                        max_tool_result_chars=self._max_tool_result_chars,
                        concurrent_tools=False,     # serialise requests
                        error_message=None,
                    )
                ),
                timeout=float(self._timeout),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "SearchAgent: timed out after {}s for keywords={}",
                self._timeout, keywords,
            )
            return None
        except Exception as exc:
            logger.warning("SearchAgent: unexpected error: {}", exc)
            return None

        if result.stop_reason == "error" or result.final_content is None:
            logger.warning(
                "SearchAgent: agent finished with stop_reason={}", result.stop_reason
            )
            return None

        content = result.final_content.strip()
        if not content or content == "NO_USEFUL_INFO":
            logger.info("SearchAgent: no useful info found")
            return None

        # Cap to configured limit
        if len(content) > self._max_purified_chars:
            content = content[: self._max_purified_chars]

        logger.info(
            "SearchAgent: search complete ({} tool calls, {} chars)",
            len(result.tools_used), len(content),
        )
        return content

    # ------------------------------------------------------------------
    # Main orchestration: decide → search agent → plan
    # ------------------------------------------------------------------

    async def plan(
        self,
        task: str,
        channel: str | None = None,
        chat_id: str | None = None,
        context_summary: str | None = None,
        existing_search_info: str | None = None,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> SearchEnhancedPlanResult:
        """Full search-enhanced planning pipeline.

        Steps:
            1. decide() — should we search? (no tools, pure LLM)
            2. _run_search_agent() — agentic web search with web_search/web_fetch
            3. planner.plan() — generate plan, injecting search results as context
        """

        async def _notify(msg: str) -> None:
            if progress_callback:
                try:
                    await progress_callback(msg)
                except Exception:
                    pass

        search_triggered = False
        search_keywords: list[str] = []
        search_info: str | None = None
        search_failed = False

        # --- Sub-module 8.1: Decide ---
        await _notify("🤔 Deciding whether to search the web...")
        try:
            search_triggered, search_keywords = await self.decide(
                task, existing_info=existing_search_info
            )
        except Exception as exc:
            logger.warning("SearchDecider: unexpected error ({}), skip search", exc)
            search_triggered = False

        if not search_triggered:
            await _notify("⏩ Skipping web search (not needed for this task)")
        else:
            kw_preview = ", ".join(search_keywords[:3])
            await _notify(f"🔍 **Searching the web** for: {kw_preview}")

            # --- Sub-module 8.2: Agentic search ---
            try:
                search_info = await self._run_search_agent(task, search_keywords)
            except Exception as exc:
                logger.warning("SearchAgent: unexpected error: {}", exc)
                search_info = None

            if search_info is None:
                search_failed = True
                await _notify("⚠️ Web search failed, proceeding without additional information")
            else:
                await _notify("✨ Search information retrieved")

        # --- Delegate to TaskPlanner ---
        await _notify("⏳ Generating execution plan...")
        plan_text = await self._planner.plan(
            task=task,
            channel=channel,
            chat_id=chat_id,
            context_summary=context_summary,
            search_info=search_info,
        )

        return SearchEnhancedPlanResult(
            plan_text=plan_text,
            search_triggered=search_triggered,
            search_keywords=search_keywords,
            purified_info=search_info,
            search_failed=search_failed,
        )

    async def replan(
        self,
        task: str,
        previous_plan: str,
        feedback: str,
        mode: str = "local",
        channel: str | None = None,
        chat_id: str | None = None,
        context_summary: str | None = None,
        existing_search_info: str | None = None,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> SearchEnhancedPlanResult:
        """Search-enhanced replanning.

        For LOCAL_REPLAN, search is skipped (reuse existing info).
        For GLOBAL_REPLAN with search_on_replan=True, re-runs the full search
        pipeline before calling planner.replan() with fresh context.
        """

        async def _notify(msg: str) -> None:
            if progress_callback:
                try:
                    await progress_callback(msg)
                except Exception:
                    pass

        search_triggered = False
        search_keywords: list[str] = []
        search_info: str | None = existing_search_info  # default: reuse cached
        search_failed = False

        if mode == "global" and self._search_on_replan:
            # Re-run search decision to get fresh information
            try:
                search_triggered, search_keywords = await self.decide(
                    task, existing_info=None  # force fresh decision
                )
            except Exception as exc:
                logger.warning("SearchDecider (replan): error ({}), skip", exc)

            if search_triggered:
                kw_preview = ", ".join(search_keywords[:3])
                await _notify(f"🔍 **Re-searching the web** for: {kw_preview}")
                fresh: str | None = None
                try:
                    fresh = await self._run_search_agent(task, search_keywords)
                except Exception as exc:
                    logger.warning("SearchAgent (replan): unexpected error: {}", exc)

                if fresh is None:
                    search_failed = True
                    await _notify("⚠️ Web re-search failed, using previous information")
                else:
                    search_info = fresh
                    await _notify("✨ Fresh search information retrieved")
            else:
                await _notify("⏩ Skipping web re-search (not needed)")

        new_plan = await self._planner.replan(
            task=task,
            previous_plan=previous_plan,
            feedback=feedback,
            mode=mode,
            channel=channel,
            chat_id=chat_id,
            context_summary=context_summary,
            search_info=search_info,
        )

        return SearchEnhancedPlanResult(
            plan_text=new_plan,
            search_triggered=search_triggered,
            search_keywords=search_keywords,
            purified_info=search_info,
            search_failed=search_failed,
        )
