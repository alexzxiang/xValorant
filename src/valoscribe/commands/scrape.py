"""CLI commands for VLR.gg scraping."""

import json
from pathlib import Path

import typer

from valoscribe.scraper import scrape_match
from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


def scrape_vlr(
    match_url: str = typer.Argument(..., help="VLR.gg match URL"),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file path (default: print to stdout)",
    ),
    pretty: bool = typer.Option(
        True, "--pretty/--compact", help="Pretty print JSON output"
    ),
    timeout: int = typer.Option(
        10, "--timeout", "-t", help="Request timeout in seconds"
    ),
):
    """
    Scrape match metadata from VLR.gg.

    Extracts per-map information including:
    - VOD URLs (YouTube links)
    - Player names, teams, and agents
    - Starting sides (attack/defense)

    Example:
        valoscribe scrape-vlr "https://www.vlr.gg/542272/nrg-vs-fnatic-..."
        valoscribe scrape-vlr <url> -o match_data.json
    """
    try:
        typer.echo(f"Scraping match: {match_url}")

        # Scrape the match
        match_data = scrape_match(match_url, timeout=timeout)

        # Format output
        indent = 2 if pretty else None
        json_output = json.dumps(match_data, indent=indent)

        # Write to file or stdout
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_output)
            typer.echo(f"Match data saved to {output}")

            # Print summary
            typer.echo(f"\nMatch: {' vs '.join(match_data['teams'])}")
            typer.echo(f"Maps: {len(match_data['maps'])}")
            for map_data in match_data["maps"]:
                # Count total players from both teams
                total_players = sum(len(team['players']) for team in map_data['teams'])
                typer.echo(
                    f"  Map {map_data['map_number']}: {map_data['map_name']} "
                    f"- {total_players} players"
                )
        else:
            # Print to stdout
            typer.echo(json_output)

    except Exception as e:
        log.error(f"Scraping failed: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
