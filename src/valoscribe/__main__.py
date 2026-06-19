"""
ValoscribeCLI entry point using Typer.

Organized command structure:
    valoscribe detect <subcommand>     - Detection commands
    valoscribe extract <subcommand>    - Extraction commands
    valoscribe orchestrate <subcommand> - Orchestration commands
    valoscribe <utility command>       - Utility commands (download, read, crop, etc.)
"""

import typer

from valoscribe.commands import detect, extract, orchestrate, scrape, utils

app = typer.Typer(
    name="valoscribe",
    help="Computer vision pipeline for analyzing Valorant esports VODs",
    add_completion=False,
    no_args_is_help=True,
)

# Add command groups
app.add_typer(detect.app, name="detect")
app.add_typer(extract.app, name="extract")
app.add_typer(orchestrate.app, name="orchestrate")

# Add utility commands at top level
app.command()(utils.download)
app.command()(utils.read)
app.command()(utils.crop)
app.command()(utils.process)
app.command()(utils.analyze)
app.command(name="split-metadata")(utils.split_metadata)
app.command()(utils.version)
app.command(name="scrape-vlr")(scrape.scrape_vlr)


if __name__ == "__main__":
    app()
