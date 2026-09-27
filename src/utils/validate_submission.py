"""Official Submission Validator Script for Business Entity Resolution.

Standalone validation utility enforcing all competition constraints:
1. File format (tab-separated .tsv)
2. Correct header columns
3. Exact row count matching test Source 1 entities
4. Subtab/comma-separated format with no duplicates
5. Valid S2- / S3- entity IDs only
6. Subset constraint (matching_results IDs must exist in candidate_pairs)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Set, Tuple


def validate_submission(matching_tsv: Path, candidate_tsv: Path, test_dir: Path) -> bool:
    """Validate submission files against official competition rules."""
    test_s1_path = test_dir / "test_source1.tsv"
    if not test_s1_path.exists():
        test_s1_path = test_dir / "test_s1.tsv"

    if not test_s1_path.exists():
        print(f"Error: Cannot find test_source1.tsv in {test_dir}")
        return False

    # Read ground-truth test S1 IDs
    s1_test_ids: Set[str] = set()
    with open(test_s1_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_test_ids.add(row["entity_id"].strip())

    print(f"Loaded {len(s1_test_ids):,} test Source 1 entity IDs.")

    # 1. Audit matching_results.tsv
    print(f"Validating {matching_tsv}...")
    matching_s1_seen: Set[str] = set()
    matched_pairs: Set[Tuple[str, str]] = set()

    with open(matching_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            print(f"FAIL [Rule 1]: Incorrect header in matching_results.tsv: {reader.fieldnames}")
            return False

        for line_num, row in enumerate(reader, start=2):
            sid = row.get("source1_entity_id", "").strip()
            if not sid:
                print(f"FAIL [Rule 2]: Empty source1_entity_id at line {line_num}")
                return False
            if sid not in s1_test_ids:
                print(f"FAIL [Rule 2]: Unknown source1_entity_id '{sid}' at line {line_num}")
                return False
            if sid in matching_s1_seen:
                print(f"FAIL [Rule 3]: Duplicate source1_entity_id '{sid}' at line {line_num}")
                return False
            matching_s1_seen.add(sid)

            raw_m = row.get("matched_entity_ids", "").strip()
            if raw_m:
                m_list = [x.strip() for x in raw_m.split(",")]
                if len(m_list) != len(set(m_list)):
                    print(f"FAIL [Rule 4]: Duplicate matched IDs in list for '{sid}': {m_list}")
                    return False
                for m in m_list:
                    if not (m.startswith("S2-") or m.startswith("S3-")):
                        print(f"FAIL [Rule 5]: Invalid matched ID prefix '{m}' for '{sid}' (must be S2- or S3-)")
                        return False
                    matched_pairs.add((sid, m))

    if len(matching_s1_seen) != len(s1_test_ids):
        print(f"FAIL [Rule 6]: Missing entities in matching_results.tsv: expected {len(s1_test_ids)}, got {len(matching_s1_seen)}")
        return False

    # 2. Audit candidate_pairs.tsv
    print(f"Validating {candidate_tsv}...")
    candidate_s1_seen: Set[str] = set()
    candidate_pairs: Set[Tuple[str, str]] = set()

    with open(candidate_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "candidate_entity_ids"]:
            print(f"FAIL [Rule 1]: Incorrect header in candidate_pairs.tsv: {reader.fieldnames}")
            return False

        for line_num, row in enumerate(reader, start=2):
            sid = row.get("source1_entity_id", "").strip()
            if not sid:
                print(f"FAIL [Rule 2]: Empty source1_entity_id at line {line_num}")
                return False
            if sid not in s1_test_ids:
                print(f"FAIL [Rule 2]: Unknown source1_entity_id '{sid}' at line {line_num}")
                return False
            if sid in candidate_s1_seen:
                print(f"FAIL [Rule 3]: Duplicate source1_entity_id '{sid}' at line {line_num}")
                return False
            candidate_s1_seen.add(sid)

            raw_c = row.get("candidate_entity_ids", "").strip()
            if raw_c:
                c_list = [x.strip() for x in raw_c.split(",")]
                if len(c_list) != len(set(c_list)):
                    print(f"FAIL [Rule 4]: Duplicate candidate IDs in list for '{sid}': {c_list}")
                    return False
                for c in c_list:
                    if not (c.startswith("S2-") or c.startswith("S3-")):
                        print(f"FAIL [Rule 5]: Invalid candidate ID prefix '{c}' for '{sid}'")
                        return False
                    candidate_pairs.add((sid, c))

    if len(candidate_s1_seen) != len(s1_test_ids):
        print(f"FAIL [Rule 6]: Missing entities in candidate_pairs.tsv: expected {len(s1_test_ids)}, got {len(candidate_s1_seen)}")
        return False

    # 3. Audit Subset Rule
    for sid, m in matched_pairs:
        if (sid, m) not in candidate_pairs:
            print(f"FAIL [Rule 7]: Matched pair ('{sid}', '{m}') in matching_results.tsv is NOT in candidate_pairs.tsv!")
            return False

    print("\n=======================================================")
    print("  PASS: Submission output files are 100% compliant!")
    print("=======================================================\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Submission Validator Utility")
    parser.add_argument("--matching", type=Path, required=True, help="Path to output/matching_results.tsv")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=Path, required=True, help="Path to raw test directory")
    args = parser.parse_args()

    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
