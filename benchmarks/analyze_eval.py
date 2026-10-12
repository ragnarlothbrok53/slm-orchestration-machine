"""Analyse eval_agent.py results: paired significance between modes, and where agents go wrong.

    uv run python benchmarks/analyze_eval.py [results/eval-agent]

Writes ``analysis.json`` next to the results. With 42 problems, only large differences mean
anything, so modes are compared pairwise on the *same* problems (exact McNemar test).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from slom.tools import evaluate_arithmetic

TEMPLATES = [
    "price x qty + fee",
    "give away, buy bags",
    "two-leg distance",
    "boxes - sold",
    "percent discount",
    "weekly savings - gift",
]


def mcnemar(rows: dict[str, dict[str, dict[str, str]]], a: str, b: str) -> dict[str, float]:
    only_b = sum(q[a]["correct"] == "0" and q[b]["correct"] == "1" for q in rows.values())
    only_a = sum(q[a]["correct"] == "1" and q[b]["correct"] == "0" for q in rows.values())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_exact": round(p, 4)}


def expressions(row: dict[str, str]) -> list[str]:
    """Calculator expressions in the transcript (stored as escaped JSON, possibly truncated)."""
    return re.findall(r'expression\\?":\s*\\?"([^"\\]+)', row.get("transcript") or "")


def value(expr: str) -> float | None:
    try:
        return float(evaluate_arithmetic(expr))
    except Exception:
        return None


def main() -> None:
    folder = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results/eval-agent"
    )
    flat = list(csv.DictReader((folder / "rows.csv").open(encoding="utf-8")))
    rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in flat:
        rows[r["id"]][r["mode"]] = r
    modes = list(dict.fromkeys(r["mode"] for r in flat))

    analysis: dict[str, object] = {
        "paired": {f"{a} vs {b}": mcnemar(rows, a, b) for a, b in combinations(modes, 2)},
        "per_template": {
            name: {
                m: sum(
                    int(q[m]["correct"]) for i, q in rows.items() if int(i.split("-")[1]) % 6 == t
                )
                for m in modes
            }
            for t, name in enumerate(TEMPLATES)
        },
    }
    for mode in (m for m in modes if m.startswith("agent")):
        wrong = [q[mode] for q in rows.values() if q[mode]["correct"] == "0"]
        computed_right = answered_own_value = 0
        for r in wrong:
            values = [value(e) for e in expressions(r)]
            computed_right += any(v == float(r["expected"]) for v in values if v is not None)
            answered_own_value += (
                bool(values)
                and values[-1] is not None
                and r["got"] != ""
                and abs(values[-1] - float(r["got"])) < 1e-6
            )
        analysis[f"{mode}_errors"] = {
            "wrong": len(wrong),
            "a_tool_call_computed_the_right_value": computed_right,
            "answer_equals_its_own_expression": answered_own_value,
        }
    (folder / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
