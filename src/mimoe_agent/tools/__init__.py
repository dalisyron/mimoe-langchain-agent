"""Tool registry: ``build_tools(settings)`` assembles the agent's tools.

Imports are deferred into the function so importing this package never pulls in every
tool module (and so partially written modules cannot break unrelated imports).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from mimoe_agent.config import Settings
    from mimoe_agent.mimoe import MimoeClient

TOOL_NAMES: tuple[str, ...] = (
    "run_python",
    "list_files",
    "read_file",
    "search_files",
    "calculator",
    "now",
    "mimoe_status",
    "git",
)


def build_tools(settings: Settings, client: MimoeClient | None = None) -> list[BaseTool]:
    """Return the eight tools in the order the model sees them.

    ``run_python`` runs code as the user (approval-gated by the agent middleware); the
    workspace tools are jailed to ``settings.workspace``; ``mimoe_status`` reads the engine
    through ``client`` (a fresh ``MimoeClient`` is created when none is given).
    """
    from mimoe_agent.tools.run_python import make_run_python
    from mimoe_agent.tools.system import calculator, make_git, make_mimoe_status, now
    from mimoe_agent.tools.workspace import Workspace, make_workspace_tools

    if client is None:
        from mimoe_agent.mimoe import MimoeClient

        client = MimoeClient(settings.base_url, settings.api_key)

    ws = Workspace(settings.workspace)
    tools: list[BaseTool] = [
        make_run_python(
            ws, allow_network=settings.allow_network, approval=not settings.auto_approve
        ),
        *make_workspace_tools(ws),
        calculator,
        now,
        make_mimoe_status(client),
        make_git(ws),
    ]
    assert [t.name for t in tools] == list(TOOL_NAMES), [t.name for t in tools]
    return tools
