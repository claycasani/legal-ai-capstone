import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-train", default="training/lora_train.jsonl")
    parser.add_argument("--output-eval", default="training/lora_eval.jsonl")
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    rows = []
    seen = set()
    for input_path in args.inputs:
        with Path(input_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    row.get("instruction", "").strip(),
                    row.get("context", "").strip()[:400],
                    row.get("response", "").strip()[:400],
                )
                if key in seen:
                    continue
                rows.append(row)
                seen.add(key)

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    eval_size = min(args.eval_size, max(0, len(rows) // 5))
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]

    write_jsonl(Path(args.output_train), train_rows)
    write_jsonl(Path(args.output_eval), eval_rows)

    print(f"Wrote {len(train_rows)} train examples to {args.output_train}")
    print(f"Wrote {len(eval_rows)} eval examples to {args.output_eval}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
