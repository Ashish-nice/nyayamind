"""Print the number of documents in every NyayaMind Vespa schema."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

app = c.vespa_app()
print(f"{'Schema':<50} {'Documents':>10}")
print("=" * 62)
total = 0
for schema in c.all_schemas():
    try:
        response = app.query(body={"yql": f"select * from sources {schema} where true", "hits": 0})
        count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
        print(f"{schema:<50} {count:>10}")
        total += count
    except Exception as e:  # noqa: BLE001
        print(f"{schema:<50} {'ERROR':>10}  ({e})")
print("=" * 62)
print(f"{'TOTAL':<50} {total:>10}")
