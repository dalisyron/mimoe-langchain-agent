"""Command-line entry points: the REPL (default command) and `serve`."""

from __future__ import annotations

import typer

from mimoe_agent import __version__

app = typer.Typer(
    help="A private local assistant running on the mimOE Studio endpoint.",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Start the interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        typer.echo(f"mimoe-agent {__version__}: REPL arrives in step D1.")


@app.command()
def serve(port: int = typer.Option(8000, help="Port on 127.0.0.1 for the web UI.")) -> None:
    """Run the FastAPI server and the web chat UI."""
    typer.echo(f"serve arrives in step D2 (port {port}).")
