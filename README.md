# EasyClaw

EasyClaw is developed based on [Nanobot](https://github.com/HKUDS/nanobot), with the following key features:

- **Browser-Enhanced Planning** — Plans on any task even with a relatively weak model by leveraging real-time web search.
- **DAG-Based Subagent Pools** — Directed Acyclic Graph scheduling with configurable recursion depths.
- **ReAct Worker & Re-planner** — Self-improves the planner DAG based on environment feedback.
- **Enhanced Memory Consolidation & Auto-Skill Learning** — Automatic skill distillation mechanism inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Architecture

```
   ╔═════════════════════════════════════════════════════════════════╗
   ║                    EasyClaw  Runtime Stack                      ║
   ╠═════════════════════════════════════════════════════════════════╣
   │  ①  Public Support Layer                                        │
   │     Config Center · State Hub · Monitoring · Exception Guard    │
   │     Tool Pool  (incl. Web Search Tools)                        │
   ├─────────────────────────────────────────────────────────────────┤
   │  ②  Web-Enhanced Decision Layer                       ★ NEW    │
   │     Search Decision Engine · Retrieval Executor                │
   │     Information Purifier & Injector                            │
   ├─────────────────────────────────────────────────────────────────┤
   │  ③  Closed-Loop Control Layer                         ★ NEW    │
   │     Feedback Collector · Task Evaluator · Re-planning Engine   │
   ├─────────────────────────────────────────────────────────────────┤
   │  ④  Planning & Scheduling Layer                                  │
   │     Top-Level Master Planner · Recursive Depth Controller      │
   ├─────────────────────────────────────────────────────────────────┤
   │  ⑤  Execution Engine Layer                                       │
   │     Reactive Event Bus (ReAct Core) · DAG Dependency Scheduler │
   ├─────────────────────────────────────────────────────────────────┤
   │  ⑥  Execution Unit Layer                                         │
   │     SubAgent Pool · Recursive Decomposition (Configurable)     │
   ╠════════════════════ ▼ async · observes history ═══════════════════╣
   │  ⑦  Skill Distillation Layer                          ★ NEW    │
   │     Dream:  cron 2h · /dream  →  dreamed-* skills             │
   │     Hermes: ×10 calls · /skill-autogen  →  hermes-* skills    │
   │     ↑  Skills fed back into ① Tool Pool                        │
   ╚═════════════════════════════════════════════════════════════════╝
```

## Quick Start

1. Copy the example config:
  Configure your models and apis, then
   ```bash
   cp config.json.example ~/.nanobot/config.json
   ```
2. Start the gateway:
   ```bash
   nanobot gateway
   ```

## Skill Distillation

EasyClaw supports two orthogonal skill distillation modes that can run independently or simultaneously.

### Mode 1 — Dream  *(Memory Consolidation)*

Consolidates conversation history into long-term memory files (`MEMORY.md`, `SOUL.md`, `USER.md`) and extracts reusable skills prefixed with `dreamed-`.

| Trigger | Condition | Notification |
|---|---|---|
| `/dream` command | Always | `Dream completed in Xs.` / `Dream: nothing to process.` |
| Cron auto (every 2 h) | When changes exist | `[Dream] Memory consolidation complete. MEMORY / SOUL / USER.md updated.` |

- **Default state:** enabled
- **Skill output:** `workspace/skills/dreamed-<name>/SKILL.md`

### Mode 2 — Hermes-Autogen  *(Tool-Usage Distillation)*

Watches tool-call history and distills recurring patterns into reusable skills prefixed with `hermes-`, inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent).

| Trigger | Condition | Notification |
|---|---|---|
| `/skill-autogen` command | Always | `Skill-Autogen completed in Xs: a new skill was created...` |
| Threshold auto (every 10 tool calls) | When a new skill is identified | `[Skill-Autogen] Background distillation complete. New skill saved to workspace/skills/` |

- **Default state:** disabled — enable via `skill_autogen.enable = true` in config
- **Skill output:** `workspace/skills/hermes-<name>/SKILL.md`

> **Note:** All channel notifications are routed to the last active session.
> Silent (log-only) when no active session exists.

## Task Persistence

| Command | Description |
|---|---|
| `/tasks` | List currently running subagents and pending interrupted tasks |
| `/tasks resume <id>` | Re-spawn a subagent task; re-deliver a main-flow task to the bus |
| `/tasks discard <id>` | Remove from registry; mark main-flow task as errored |

**Key behaviors after service restart:**

- Within 3 seconds, a notification is pushed to each affected channel listing all interrupted tasks with `resume`/`discard` options.
- `/tasks resume <id>`: Subagent restarts from the beginning (original task description); main-flow task resumes from the persisted `runtime_checkpoint`.
- `/tasks discard <id>`: Clears the persistence record; subagent is dropped entirely; main-flow task leaves an error marker in the session.

## Documentation

- [Nanobot Extended Docs](readme.nanobot.md)
- [Upstream Nanobot](https://github.com/HKUDS/nanobot)

## Configuration

Copy `config.json.example` to `~/.nanobot/config.json` and edit as needed.
All keys below live under `agents.defaults` unless noted otherwise.

### Plan-and-Solve + DAG Subagents

| Key | Type | Default | Description |
|---|---|---|---|
| `plan_and_solve` | bool | `true` | Enable ReAct→Plan-and-Solve upgrade; the agent builds a DAG before execution |
| `max_subagent_depth` | int | `2` | Maximum recursion depth for nested SubAgent spawning (0 = flat, no nesting) |

### Search-Enhanced Planning  *(② Web-Enhanced Decision Layer)*

| Key | Type | Default | Description |
|---|---|---|---|
| `search_enhanced_planning.enable` | bool | `true` | Gate for the whole web-search pre-planning pipeline |
| `search_enhanced_planning.max_results` | int | `5` | Max search results fetched per planning query |
| `search_enhanced_planning.timeout` | int | `120` | Seconds before a single search call times out |
| `search_enhanced_planning.search_on_replan` | bool | `false` | Re-run web search each time the planner triggers a re-plan cycle |
| `search_enhanced_planning.max_purified_chars` | int | `2000` | Max characters of distilled search content injected into the planning prompt |
| `tools.web.enable` | bool | `false` | **Must be `true`** to activate web search tools used by this layer |
| `tools.web.search.provider` | string | `"duckduckgo"` | Search backend: `duckduckgo` \| `bing` \| `google` \| `searxng` |
| `tools.web.search.apiKey` | string | `""` | API key for providers that require one |
| `tools.web.search.maxResults` | int | `5` | Hard cap on results returned by the search tool |
| `tools.web.search.timeout` | int | `30` | Per-request timeout for the search tool (seconds) |

### Skill Distillation — Dream  *(⑦ Mode 1)*

| Key | Type | Default | Description |
|---|---|---|---|
| `dream.intervalH` | float | `2` | Hours between automatic Dream cron runs (set to `0` to disable cron) |
| `dream.modelOverride` | string\|null | `null` | Use a different model for Dream; falls back to agent default when `null` |
| `dream.maxBatchSize` | int | `20` | Max conversation turns processed per Dream run |
| `dream.maxIterations` | int | `15` | Max LLM iterations allowed inside a single Dream run |
| `dream.annotateLineAges` | bool | `true` | Annotate each memory line with its age to guide future consolidation |

### Skill Distillation — Hermes-Autogen  *(⑦ Mode 2)*

| Key | Type | Default | Description |
|---|---|---|---|
| `skill_autogen.enable` | bool | `false` | Enable Hermes-Autogen; **disabled by default** — must opt in explicitly |
| `skill_autogen.nudge_interval` | int | `10` | Tool-call count threshold that triggers a background distillation run |
| `skill_autogen.max_iterations` | int | `8` | Max LLM iterations allowed inside a single Hermes-Autogen run |
