"""Generate the arithmetic word-problem set used by eval_agent.py (deterministic: fixed seed).

Every problem needs 2-3 operations on 2-3 digit numbers and has an exact integer answer. This is a
deliberately narrow task: it measures whether the orchestration gets a small model to *use a tool
correctly*, not general reasoning or agent ability.

    uv run python benchmarks/data/make_arith.py > benchmarks/data/arith.jsonl
"""

from __future__ import annotations

import json
import random

NAMES = ["Priya", "Arjun", "Meera", "Rahul", "Ananya", "Vikram", "Sara", "Kabir"]
ITEMS = ["pens", "notebooks", "mangoes", "tickets", "bottles", "chairs"]


def problems(rng: random.Random) -> list[tuple[str, int]]:
    n, item = rng.choice(NAMES), rng.choice(ITEMS)
    a, b, c, d = (rng.randint(12, 99) for _ in range(4))
    big = rng.randint(120, 999)
    pct = rng.choice([10, 20, 25, 40, 50])
    price = rng.randint(3, 40) * 20  # discounts below come out whole
    return [
        (
            f"A shop sells {item} at {a} rupees each. {n} buys {b} {item} and pays a {c} rupee "
            f"delivery fee. How many rupees does {n} pay in total?",
            a * b + c,
        ),
        (
            f"{n} has {big} marbles, gives away {a}, then buys {b} bags of {c} marbles each. "
            f"How many marbles does {n} have now?",
            big - a + b * c,
        ),
        (
            f"A train travels at {a} km/h for {b} hours and then at {c} km/h for {d} hours. "
            "How many kilometres does it travel in total?",
            a * b + c * d,
        ),
        (
            f"A box holds {a} apples. A farm fills {b} boxes and then sells {big} apples. "
            "How many apples are left?",
            a * b - big,
        ),
        (
            f"A jacket costs {price} rupees and is discounted by {pct} percent. "
            "What is the sale price in rupees?",
            price * (100 - pct) // 100,
        ),
        (
            f"{n} saves {a} rupees every week for {b} weeks and then spends {c} rupees on a gift. "
            "How many rupees are left?",
            a * b - c,
        ),
    ]


def main() -> None:
    rng = random.Random(7)
    for i in range(42):  # 7 per template, each with freshly drawn numbers
        question, answer = problems(rng)[i % 6]
        print(json.dumps({"id": f"arith-{i:02}", "question": question, "answer": answer}))


if __name__ == "__main__":
    main()
