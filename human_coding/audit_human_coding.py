#!/usr/bin/env python3
"""Reproduce the six-coder reliability audit from anonymised JSON exports."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
INPUT = HERE / "exports"
CODING = HERE
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)

CODES = pd.read_csv(CODING / "coding_frame.csv")["code"].tolist()
SAMPLE = pd.read_csv(CODING / "gold_sample.csv", dtype={"seg_id": str})
FILES = {c: INPUT / f"coder_{c}.json" for c in "ABCDEF"}


def load_coder(coder: str, path: Path) -> pd.DataFrame:
    obj = json_load(path)
    rows = pd.DataFrame(obj["rows"])
    rows = rows.rename(columns={"text (Vietnamese)": "text", "question": "field_desc"})
    rows["seg_id"] = rows["seg_id"].astype(str).str.strip()
    rows["primary_code"] = rows["primary_code"].fillna("").astype(str).str.strip().str.upper()
    rows["coder_id"] = coder
    return rows


def json_load(path: Path):
    import json
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fleiss_kappa(matrix: pd.DataFrame) -> float:
    counts = np.column_stack([(matrix == code).sum(axis=1).to_numpy() for code in CODES])
    n = counts.sum(axis=1)
    if not np.all(n == n[0]):
        raise ValueError("Fleiss' kappa requires a constant number of raters per item")
    n0 = int(n[0])
    p_item = ((counts * counts).sum(axis=1) - n0) / (n0 * (n0 - 1))
    p_bar = p_item.mean()
    p_cat = counts.sum(axis=0) / counts.sum()
    p_e = (p_cat * p_cat).sum()
    return float((p_bar - p_e) / (1 - p_e))


def nominal_alpha(matrix: pd.DataFrame) -> float:
    observed_num = observed_den = 0.0
    totals = {c: 0 for c in CODES}
    total_n = 0
    for _, row in matrix.iterrows():
        vals = [x for x in row.tolist() if isinstance(x, str) and x in CODES]
        n = len(vals)
        if n < 2:
            continue
        cnt = pd.Series(vals).value_counts()
        observed_num += n * n - float((cnt * cnt).sum())
        observed_den += n * (n - 1)
        for code, value in cnt.items():
            totals[code] += int(value)
        total_n += n
    do = observed_num / observed_den
    de = sum(v * (total_n - v) for v in totals.values()) / (total_n * (total_n - 1))
    return float(1 - do / de)


def bootstrap_metric(matrix: pd.DataFrame, fn, n_boot: int = 200, seed: int = 20260926):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        take = rng.integers(0, len(matrix), len(matrix))
        try:
            val = fn(matrix.iloc[take].reset_index(drop=True))
            if np.isfinite(val):
                vals.append(val)
        except (ValueError, ZeroDivisionError):
            pass
    return np.percentile(vals, [2.5, 97.5]) if vals else [np.nan, np.nan]


# Pre-specified senior adjudication of the nine ontology-compatible cases that
# lacked a strict majority. Two irreducibly multi-theme responses are excluded.
ADJUDICATION = {
    "S009": {"decision": "EXCLUDE_SINGLE_LABEL", "gold_code": "", "secondary_code": "SOCIAL|ECON",
             "rationale": "Two explicit and co-equal reasons (family and livelihood); no defensible primary theme."},
    "S038": {"decision": "INCLUDE", "gold_code": "OTHER", "secondary_code": "",
             "rationale": "States that the area already has everything; no requested substantive change."},
    "S048": {"decision": "INCLUDE", "gold_code": "MOBIL", "secondary_code": "INFRA",
             "rationale": "Reason for non-use concerns access to public transport; parking infrastructure is secondary."},
    "S075": {"decision": "INCLUDE", "gold_code": "OTHER", "secondary_code": "",
             "rationale": "Requests no additional change and names no substantive theme."},
    "S097": {"decision": "INCLUDE", "gold_code": "OTHER", "secondary_code": "",
             "rationale": "Declines to request a change and names no substantive theme."},
    "S099": {"decision": "EXCLUDE_SINGLE_LABEL", "gold_code": "", "secondary_code": "ECON|MOBIL",
             "rationale": "Two explicit and co-equal location reasons (livelihood and public transport); no defensible primary theme."},
    "S121": {"decision": "INCLUDE", "gold_code": "ECON", "secondary_code": "",
             "rationale": "The phrase refers to the economic and commercial environment, not environmental quality."},
    "S159": {"decision": "INCLUDE", "gold_code": "OTHER", "secondary_code": "",
             "rationale": "Too vague to identify a substantive reason or theme."},
    "S171": {"decision": "INCLUDE", "gold_code": "MOBIL", "secondary_code": "",
             "rationale": "Walking discomfort is stated specifically as an access barrier to public transport."},
}


def main() -> None:
    loaded = {coder: load_coder(coder, path) for coder, path in FILES.items()}
    sample_ids = set(SAMPLE["seg_id"])
    audit_rows = []
    for coder, frame in loaded.items():
        found = set(frame["seg_id"])
        invalid = int((~frame["primary_code"].isin(CODES) & frame["primary_code"].ne("")).sum())
        merged = SAMPLE[["seg_id", "instrument", "text"]].merge(
            frame[["seg_id", "instrument", "text", "primary_code"]], on="seg_id", how="left",
            suffixes=("_sample", "_coder"),
        )
        audit_rows.append({
            "coder_id": coder,
            "expected_n": len(SAMPLE),
            "rows_in_export": len(frame),
            "unique_seg_ids": frame["seg_id"].nunique(),
            "coded_n": int(frame["primary_code"].isin(CODES).sum()),
            "missing_code_n": int(frame["primary_code"].eq("").sum()),
            "invalid_code_n": invalid,
            "duplicate_seg_id_n": int(frame["seg_id"].duplicated().sum()),
            "missing_seg_id_n": len(sample_ids - found),
            "extra_seg_id_n": len(found - sample_ids),
            "text_mismatch_n": int((merged["text_sample"].fillna("") != merged["text_coder"].fillna("")).sum()),
            "instrument_mismatch_n": int((merged["instrument_sample"].fillna("") != merged["instrument_coder"].fillna("")).sum()),
        })
    pd.DataFrame(audit_rows).to_csv(OUT / "coder_completeness_audit.csv", index=False, encoding="utf-8-sig")

    matrix = SAMPLE[["seg_id", "instrument", "field", "field_desc", "text"]].copy()
    for coder, frame in loaded.items():
        temp = frame[["seg_id", "primary_code"]].drop_duplicates("seg_id").rename(columns={"primary_code": coder})
        matrix = matrix.merge(temp, on="seg_id", how="left")
        matrix[coder] = matrix[coder].where(matrix[coder].isin(CODES), np.nan)

    pair_rows = []
    for a, b in itertools.combinations(FILES, 2):
        pair = matrix[[a, b]].dropna()
        kappa = cohen_kappa_score(pair[a], pair[b], labels=CODES)
        lo, hi = bootstrap_metric(pair, lambda z, a=a, b=b: cohen_kappa_score(z[a], z[b], labels=CODES))
        pair_rows.append({
            "coder_1": a, "coder_2": b, "n": len(pair),
            "raw_agreement": float((pair[a] == pair[b]).mean()),
            "cohen_kappa": float(kappa), "ci_low": float(lo), "ci_high": float(hi),
        })
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(OUT / "pairwise_agreement.csv", index=False, encoding="utf-8-sig")

    ratings = matrix[list(FILES)]
    complete = ratings.dropna()
    fk = fleiss_kappa(complete)
    fk_lo, fk_hi = bootstrap_metric(complete, fleiss_kappa)
    ka = nominal_alpha(ratings)
    ka_lo, ka_hi = bootstrap_metric(ratings, nominal_alpha)

    consensus_rows = []
    for _, row in matrix.iterrows():
        vals = row[list(FILES)].dropna()
        counts = vals.value_counts()
        top_n = int(counts.iloc[0])
        leaders = sorted(counts[counts == top_n].index.tolist())
        n = len(vals)
        unique = len(leaders) == 1
        strict_majority = unique and top_n > n / 2
        if unique and top_n == n:
            strength = "unanimous"
        elif strict_majority and top_n >= 5:
            strength = "very_strong_majority"
        elif strict_majority:
            strength = "strict_majority"
        elif unique:
            strength = "plurality_only"
        else:
            strength = "tie"
        consensus_rows.append({
            "seg_id": row["seg_id"], "n_raters": n, "top_count": top_n,
            "agreement_share": top_n / n,
            "consensus_code": leaders[0] if unique else "",
            "strict_majority": strict_majority,
            "needs_adjudication": not strict_majority,
            "strength": strength,
            "tied_codes": "|".join(leaders) if not unique else "",
        })
    consensus = pd.DataFrame(consensus_rows)
    consensus.to_csv(OUT / "consensus_by_segment.csv", index=False, encoding="utf-8-sig")

    annotated = matrix.merge(consensus, on="seg_id")
    annotated.to_csv(OUT / "human_ratings_with_consensus.csv", index=False, encoding="utf-8-sig")
    annotated.loc[annotated["needs_adjudication"], [
        "seg_id", "instrument", "field_desc", "text", *FILES.keys(), "top_count", "agreement_share", "tied_codes"
    ]].to_csv(OUT / "adjudication_queue.csv", index=False, encoding="utf-8-sig")

    gold = annotated.loc[annotated["strict_majority"], ["seg_id", "consensus_code", "top_count", "n_raters", "agreement_share"]]
    gold = gold.rename(columns={"consensus_code": "gold_code"})
    gold.to_csv(OUT / "consensus_gold_standard.csv", index=False, encoding="utf-8-sig")

    adj_rows = []
    for seg_id, decision in ADJUDICATION.items():
        src = annotated.loc[annotated.seg_id == seg_id].iloc[0]
        adj_rows.append({"seg_id": seg_id, "field_desc": src.field_desc, "text": src.text, **decision})
    adjudicated = pd.DataFrame(adj_rows)
    adjudicated.to_csv(OUT / "adjudication_decisions_substantive_9.csv", index=False, encoding="utf-8-sig")

    majority_gold = gold[["seg_id", "gold_code"]].copy()
    majority_gold["label_source"] = "strict_majority"
    adjudicated_included = adjudicated.loc[adjudicated.decision == "INCLUDE", ["seg_id", "gold_code"]].copy()
    adjudicated_included["label_source"] = "senior_adjudication"
    final_gold = pd.concat([majority_gold, adjudicated_included], ignore_index=True).sort_values("seg_id")
    final_gold.to_csv(OUT / "final_gold_standard_single_label_130.csv", index=False, encoding="utf-8-sig")
    adjudicated.loc[adjudicated.decision == "EXCLUDE_SINGLE_LABEL"].to_csv(
        OUT / "excluded_multitheme_single_label_2.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Fleiss' kappa (complete cases): {fk:.3f} [{fk_lo:.3f}, {fk_hi:.3f}]")
    print(f"Krippendorff alpha: {ka:.3f} [{ka_lo:.3f}, {ka_hi:.3f}]")
    print(f"Final gold standard: {len(final_gold)} segments")


if __name__ == "__main__":
    main()
