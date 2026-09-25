"""Tiny sales report app used as demo material for the agent."""

import csv
from pathlib import Path

from utils import fmt_money

DATA = Path(__file__).resolve().parent.parent / "sales.csv"


def load_rows(path: Path = DATA) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def revenue_by_region(rows: list[dict[str, str]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        totals[row["region"]] = totals.get(row["region"], 0.0) + float(row["revenue"])
    return totals


def main() -> None:
    rows = load_rows()
    # TODO: sort regions by revenue instead of alphabetically
    for region, total in sorted(revenue_by_region(rows).items()):
        print(f"{region:<6} {fmt_money(total)}")
    print(f"total  {fmt_money(sum(float(r['revenue']) for r in rows))}")


if __name__ == "__main__":
    main()
