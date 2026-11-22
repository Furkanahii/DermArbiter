"""Prepare candidates from a local, pinned MedMCQA validation parquet.

Requires pyarrow. Output is NOT a clinically reviewed or accepted test cohort.
Preserves the published question/options; excludes explanations from model input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dermarbiter.research.text_benchmark import VERSION, validate_cases


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-sha256", default="b768a1ea34afc9f80d3106d9b21f80fa8a00ec450a1f6cd641af72ca9e591021")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    file_hash = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    if file_hash != args.expected_sha256:
        raise ValueError("Unexpected validation file hash; verify source revision before converting")
    rows = pq.read_table(args.parquet).to_pylist()
    selected = [r for r in rows if r.get("subject_name") == "Skin" and r.get("choice_type") == "single"]
    cases = []
    for row in selected:
        if type(row["cop"]) is not int or row["cop"] not in range(4):
            raise ValueError("Missing/invalid published answer key; do not use public test placeholders")
        cid = "medmcqa:" + row["id"]
        cases.append({"schema_version": VERSION, "case_id": cid, "source_id": cid,
            "source_url": f"https://huggingface.co/datasets/openlifescienceai/medmcqa/tree/{args.revision}",
            "license": "Apache-2.0 (dataset card)", "source_revision": args.revision,
            "source_file_sha256": file_hash,
            "original_split": "validation", "split": "dev", "modality": "text",
            "text_origin": "published_exam_question", "gold_origin": "published_answer_key",
            "data_status": "pilot_candidate", "question": row["question"],
            "options": dict(zip("ABCD", [row[k] for k in ("opa", "opb", "opc", "opd")])),
            "answer": "ABCD"[row["cop"]],
            "limitations": ["No new clinician review; published key may contain errors or outdated content.",
                "Question ID used for grouping; original patient/exam grouping unavailable.",
                "Public benchmark contamination unknown; never claim unseen clinical validation."]})
    validate_cases(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        for case in cases:
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"n_candidates": len(cases), "status": "pilot_candidate", "split": "dev"}))


if __name__ == "__main__":
    main()
