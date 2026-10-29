"""Regression gate for the manually labeled demonstration policies."""
import json
import sys

def verify(report):
    current=report["summary"]
    held=report["holdout"]["verified_summary"]
    cases={x["id"]:x for x in report["cases"]}
    thresholds={
        "p04 booking answer is grounded": cases["p04"]["passed"],
        "p05 credentials answer is grounded": cases["p05"]["passed"],
        "original recall": current["recall_at_3"] >= .8636,
        "original abstention": current["abstention_accuracy"] >= .875,
        "heldout recall": held["recall_at_3"] >= .625,
        "heldout abstention": held["abstention_accuracy"] >= .8333,
        "heldout pass rate": held["case_pass_rate"] >= .6429,
    }
    for label,passed in thresholds.items():
        print(("PASS" if passed else "FAIL")+": "+label)
    return all(thresholds.values())

if __name__=="__main__":
    with open(sys.argv[1],encoding="utf8") as f:
        report=json.load(f)
    if not verify(report):
        raise SystemExit("Quality regression: do not deploy this staging commit.")
