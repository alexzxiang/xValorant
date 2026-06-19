"""
Generate per-map metadata.json files from a VLR.gg match ID.

Scrapes team names, player rosters, agent picks, and starting sides, then
writes one metadata.json per map into the output directory structure:

    <output_dir>/<team1>_vs_<team2>/map<N>_<mapname>/metadata.json

Usage:
    python scripts/generate_match_metadata.py 684610
    python scripts/generate_match_metadata.py 684610 --output-dir matches
    python scripts/generate_match_metadata.py https://www.vlr.gg/684610/team-a-vs-team-b-...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from valoscribe.scraper.vlr_scraper import VLRScraper


def match_url_from_id(match_id: str) -> str:
    return f"https://www.vlr.gg/{match_id}?game=all&tab=overview"


def slugify(name: str) -> str:
    """Lowercase, spaces to underscores, strip non-alphanumeric."""
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_").replace("-", "_"))


def build_metadata(map_data: dict, match_url: str) -> dict:
    """Convert the scraper's map_data dict into the metadata.json schema."""
    teams_out = []
    players_out = []

    for team in map_data["teams"]:
        teams_out.append({
            "name": team["name"],
            "starting_side": team["starting_side"],
        })
        for player in team["players"]:
            players_out.append({
                "name": player["name"],
                "team": team["name"],
                "agent": player["agent"].lower(),
            })

    return {
        "teams": teams_out,
        "players": players_out,
        "map": map_data["map_name"],
        "map_number": map_data["map_number"],
        "match_url": match_url,
        "vod_url": map_data.get("vod_url"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate metadata.json files from VLR match")
    parser.add_argument(
        "match",
        help="VLR match ID (e.g. 684610) or full VLR match URL",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Root output directory (default: current directory). "
             "Creates <team1>_vs_<team2>/map<N>_<mapname>/ subdirs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print metadata without writing files",
    )
    args = parser.parse_args()

    # Resolve match URL
    if args.match.startswith("http"):
        match_url = args.match
    else:
        match_url = match_url_from_id(args.match)

    print(f"Scraping: {match_url}")

    scraper = VLRScraper()
    result = scraper.scrape_match(match_url)

    teams = result["teams"]
    if len(teams) >= 2:
        series_slug = f"{slugify(teams[0])}_vs_{slugify(teams[1])}"
    else:
        series_slug = "match"

    out_root = args.output_dir or Path(".")

    print(f"\nMatch: {teams[0] if teams else '?'} vs {teams[1] if len(teams) > 1 else '?'}")
    print(f"Maps found: {len(result['maps'])}")
    print(f"Series dir: {series_slug}/\n")

    for map_data in result["maps"]:
        # Skip unplayed maps (deciders in a sweep have no player data)
        total_players = sum(len(t["players"]) for t in map_data["teams"])
        if total_players < 10:
            print(f"  Map {map_data['map_number']}: {map_data['map_name']} — skipped (no player data, likely unplayed decider)\n")
            continue

        metadata = build_metadata(map_data, result["match_url"])

        map_slug = f"map{map_data['map_number']}_{slugify(map_data['map_name'])}"
        map_dir = out_root / series_slug / map_slug

        print(f"  Map {map_data['map_number']}: {map_data['map_name']}")
        for team in metadata["teams"]:
            print(f"    {team['name']} starts {team['starting_side']}")
        for p in metadata["players"]:
            print(f"    {p['name']} ({p['team']}) — {p['agent']}")
        if map_data.get("vod_url"):
            print(f"    VOD: {map_data['vod_url']}")
        print()

        if not args.dry_run:
            map_dir.mkdir(parents=True, exist_ok=True)
            out_path = map_dir / "metadata.json"
            out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"    Wrote: {out_path}")

    if args.dry_run:
        print("(dry-run — no files written)")
    else:
        print(f"\nDone. Metadata written under: {out_root / series_slug}/")


if __name__ == "__main__":
    main()
