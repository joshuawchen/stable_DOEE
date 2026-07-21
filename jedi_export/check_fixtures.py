#!/usr/bin/env python3
"""Parity check: doee_to_yaml.validate_spec must accept and reject exactly
the fixtures in fixtures.json, which mirror the C++ validate() unit test.

    python3 check_fixtures.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doee_to_yaml as D  # noqa: E402

fails = []
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures.json")) as f:
    cases = json.load(f)["cases"]

for case in cases:
    problems = D.validate_spec(case["spec"])
    got = "reject" if problems else "accept"
    ok = got == case["expect"]
    detail = f" ({problems[0]})" if problems else ""
    print(f"  {'PASS' if ok else 'FAIL'}  {case['name']}: {got}{detail}")
    if not ok:
        fails.append(case["name"])

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
