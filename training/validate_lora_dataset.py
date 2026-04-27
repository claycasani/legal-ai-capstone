import argparse
import json
from pathlib import Path

REQUIRED_KEYS = {"instruction", "context", "response"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    path = Path(args.dataset)
    count = 0
    errors = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSON: {exc}")
                continue

            missing = REQUIRED_KEYS - set(row)
            if missing:
                errors.append(f"Line {line_number}: missing keys {sorted(missing)}")
            for key in REQUIRED_KEYS:
                if key in row and not str(row[key]).strip():
                    errors.append(f"Line {line_number}: empty {key}")
            if "## Supporting Citations" not in row.get("response", ""):
                errors.append(f"Line {line_number}: response missing Supporting Citations section")
            if "[Chunk 1]" not in row.get("response", ""):
                errors.append(f"Line {line_number}: response missing [Chunk 1] citation")

    if errors:
        for error in errors[:25]:
            print(error)
        if len(errors) > 25:
            print(f"...and {len(errors) - 25} more errors")
        raise SystemExit(1)

    print(f"Validated {count} examples in {path}")


if __name__ == "__main__":
    main()
