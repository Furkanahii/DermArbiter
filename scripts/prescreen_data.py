"""Automated pre-screen before the human data review (does not replace it).

  python scripts/prescreen_data.py data/text_benchmark/*.jsonl

Flags, per case:
  name_in_text      SCIN: a word (>=5 letters) from a candidate condition name
                    appears in the patient text (possible label leak).
  image_cue         MCQ: the stem refers to an image or figure.
  positional_option MCQ: an option depends on option order ("all of the above",
                    "A and B"), which option shuffling would break.
  duplicate_options MCQ: two options with the same text.
  choices_mismatch  MCQ: the stem embeds "Answer Choices: (A) ..." that disagree
                    with the options field.
Prints counts and flagged case ids per file; exit code 1 if anything is flagged.
Regexes over-flag on purpose; a human decides.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

IMAGE = re.compile(r"\b(image|picture|photo(graph)?|figure|fig\.|depicted|illustrated|shown (below|here|in))\b", re.I)
POSITIONAL = re.compile(r"\b(all of the above|none of the above|all of these|none of these|"
                        r"both (of )?(the )?above|[A-J] (and|&|or) [A-J])\b", re.I)
CHOICES = re.compile(r"\(([A-J])\)\s*(.+?)(?=\s*\([A-J]\)|$)", re.S)


def flags(case: dict, name_words: set[str]) -> list[str]:
    out = []
    if "case_text" in case:  # SCIN-style differential case
        text = case["case_text"].lower()
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in name_words):
            out.append("name_in_text")
        return out
    q, opts = case["question"], case["options"]
    if IMAGE.search(q):
        out.append("image_cue")
    if any(POSITIONAL.search(v) for v in opts.values()):
        out.append("positional_option")
    if len({v.strip().casefold() for v in opts.values()}) < len(opts):
        out.append("duplicate_options")
    if "Answer Choices:" in q:
        embedded = dict((k, v.strip()) for k, v in CHOICES.findall(q.split("Answer Choices:")[-1]))
        if any(embedded.get(k, "").casefold() != v.strip().casefold() for k, v in opts.items()):
            out.append("choices_mismatch")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", type=Path)
    a = p.parse_args(argv)
    any_flag = False
    for path in a.datasets:
        cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # "rash" is the patient's own description category in SCIN, not a label
        words = {w for c in cases for n in c.get("candidates", [])
                 for w in re.findall(r"[a-z]{5,}", n.lower())} - {"rash"}
        flagged: dict[str, list[str]] = {}
        for c in cases:
            for f in flags(c, words):
                flagged.setdefault(f, []).append(c["case_id"])
        any_flag |= bool(flagged)
        print(f"{path.name}: {len(cases)} cases, " +
              (", ".join(f"{k} {len(v)}" for k, v in sorted(flagged.items())) or "no flags"))
        for k, ids in sorted(flagged.items()):
            print(f"  {k}: {' '.join(ids[:20])}{' ...' if len(ids) > 20 else ''}")
    return 1 if any_flag else 0


if __name__ == "__main__":
    sys.exit(main())
