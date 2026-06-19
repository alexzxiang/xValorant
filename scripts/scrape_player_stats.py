"""
Scrape per-player performance stats from a VLR.gg event stats page.

Usage:
    python scripts/scrape_player_stats.py 2765
    python scripts/scrape_player_stats.py 2765 --output data/player_stats/masters_london_2026.json
    python scripts/scrape_player_stats.py https://www.vlr.gg/event/stats/2765/valorant-masters-london-2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from valoscribe.scraper.vlr_scraper import VLRScraper


def main():
    parser = argparse.ArgumentParser(description="Scrape VLR.gg event player stats")
    parser.add_argument("event", help="VLR event ID (e.g. 2765) or full event stats URL")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path. Defaults to data/player_stats/<event_id>.json")
    args = parser.parse_args()

    event = args.event.strip()
    event_id = event.split("/event/stats/")[-1].split("/")[0] if "/event/" in event else event

    output_path = Path(args.output) if args.output else Path(f"data/player_stats/{event_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scraper = VLRScraper()
    stats = scraper.scrape_event_stats(event)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Scraped {len(stats)} players -> {output_path}")

    # Preview top 5 by rating
    top5 = sorted(stats.items(), key=lambda x: x[1].get("rating", 0), reverse=True)[:5]
    print("\nTop 5 by rating:")
    for name, s in top5:
        print(f"  {name:<20} ({s['team']:<6}) rating={s['rating']:.2f}  ACS={s['acs']:.0f}  CL%={s['cl_pct']*100:.0f}%")


if __name__ == "__main__":
    main()
