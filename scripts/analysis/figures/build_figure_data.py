#!/usr/bin/env python
"""Merge the per-figure JSONs into the single dataset the artifact embeds.

Fails loudly on a missing figure rather than emitting a partial dataset — the failure mode this
folder structure exists to prevent.
"""
import glob, json, os, sys
OUT = "results/analysis/figures"
EXPECT = {2:"fig02_factor_ranking",3:"fig03_dose_response",4:"fig04_olfactory_ladders",
          5:"fig05_consequence_chain",6:"fig06_response_targeting",7:"fig07_proximity",
          8:"fig08_peri_damage",9:"fig09_nociception_by_origin",
          10:"fig10_dwell_by_injury_time",11:"fig11_near_lethal",
          12:"fig12_dwell_by_nutrition_time",13:"fig13_ten_agents",
          14:"fig14_variance_decomposition",15:"fig15_nociception_all_agents",
          16:"fig16_modulator_ratio",17:"fig17_modulator_variance",
          18:"fig18_internal_state_dependence",19:"fig19_internal_state_table"}

def main():
    merged, missing = {}, []
    for num, name in sorted(EXPECT.items()):
        p = f"{OUT}/{name}.json"
        if not os.path.exists(p):
            missing.append(f"figure {num}: {p}"); continue
        merged[name] = json.load(open(p))
    if missing:
        print("MISSING FIGURE DATA — refusing to build a partial dataset:")
        for m in missing: print("   ", m)
        print(f"\nrun: python scripts/analysis/figures/run_all.py")
        return 1
    dest = f"{OUT}/all_figures.json"
    json.dump(merged, open(dest, "w"), default=float)
    print(f"merged {len(merged)} figures -> {dest}  ({os.path.getsize(dest):,} bytes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
