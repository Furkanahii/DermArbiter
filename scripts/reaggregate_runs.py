"""Re-score completed SCIN runs with the current Borda rule, without model calls.

  python scripts/reaggregate_runs.py runs/pilot runs/pilot_reaggregated

Aggregated panel answers never enter a prompt (the moderator sees the raw panels),
so recomputing the final from the stored proposals is exact. Copies SRC to DST and
rewrites `final`/`status` for vote3, vote6, isolated_refine, debate, role_vote and
role_debate records. Failed cases stay failed: a call that errored has no output
to re-score. Prints every changed final.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dermarbiter.research.text_benchmark import borda_vote  # noqa: E402

AGGREGATED = {"vote3": "initial_proposals", "vote6": "initial_proposals",
              "role_vote": "initial_proposals", "isolated_refine": "revised_proposals",
              "debate": "revised_proposals", "role_debate": "revised_proposals"}


def reaggregate(record: dict) -> dict:
    key = AGGREGATED.get(record["config"]["method"])
    if record["task"] != "dx" or key is None or record["status"] == "error":
        return record
    proposals = record[key]
    proposals = list(proposals.values()) if isinstance(proposals, dict) else proposals
    final = borda_vote(proposals)
    return {**record, "final": final, "status": "abstain" if final["answer"] is None else "ok"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    a = p.parse_args(argv)
    if a.dst.exists():
        p.error(f"{a.dst} exists")
    shutil.copytree(a.src, a.dst)
    changed = 0
    for f in sorted(a.dst.rglob("predictions.jsonl")):
        records = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        new = [reaggregate(r) for r in records]
        for old, rec in zip(records, new):
            if (old.get("final") or {}).get("answer") != (rec.get("final") or {}).get("answer"):
                changed += 1
                print(f"{f.parent.relative_to(a.dst)} {rec['case_id']}: "
                      f"{old['final']['answer']} -> {rec['final']['answer']}")
        f.write_text("".join(json.dumps(r) + "\n" for r in new))
    print(f"changed finals: {changed}")


if __name__ == "__main__":
    main()
