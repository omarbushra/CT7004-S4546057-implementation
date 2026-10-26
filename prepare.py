"""Load IEEE-CIS, derive entity keys, preprocess, split temporally.

Implements Sections 3.5 (entity extraction) and 3.7.1 (preprocessing and
splitting) of the dissertation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (DATA, DATA_FRACTION, SEED, SPLIT, run_profile,
                    save_json, save_table, set_seed)

DEVICE_FIELDS = ["DeviceInfo", "DeviceType", "id_30", "id_31", "id_33"]


def load_raw() -> pd.DataFrame:
    """Join transaction and identity tables on TransactionID.

    Left join, deliberately: the identity table holds 144,233 records against
    590,540 transactions, so an inner join would silently discard three
    quarters of the data and bias the fraud rate (Section 3.5).
    """
    tx = pd.read_csv(DATA / "train_transaction.csv")
    idf = pd.read_csv(DATA / "train_identity.csv")
    # Kaggle ships the test-side identity file with hyphens (id-01); normalise.
    idf.columns = [c.replace("-", "_") for c in idf.columns]
    df = tx.merge(idf, on="TransactionID", how="left")
    print(f"transactions={len(tx):,}  identity={len(idf):,}  "
          f"identity coverage={len(idf)/len(tx):.1%}")
    return df


def add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Derive card and device entity identifiers.

    card1 alone is insufficient: 13,553 distinct values against 590,540
    transactions, so it cannot represent individual accounts. The composite key
    follows Section 3.5 - card1 + addr1 + a normalised account-age field
    derived from D1 and TransactionDT. D1 is days since the card began its
    relationship with the issuer, so (transaction day - D1) is approximately
    constant per account and acts as an account-opening date.

    This heuristic is widely used in applied work on this dataset but has NOT
    been validated against ground-truth account identities, which are not
    released. Treat it as an approximation; card1_only is kept so the ablation
    in Section 3.5 can be run.
    """
    df = df.copy()
    day = df["TransactionDT"] / (24 * 60 * 60)
    account_open = np.floor(day - df["D1"].fillna(-999))

    def key(s: pd.Series) -> pd.Series:
        """Integer-like column to a string key, with a sentinel for missing.

        pandas 3 returns StringDtype from .astype(str) and keeps NA as NA, so
        concatenating a column that contains NA silently yields NA for the
        whole row. Left unhandled that would collapse every transaction with a
        missing addr1 (~11% of the data) into one enormous hub node - the same
        spurious structure Section 3.5 rejects for merchants.
        """
        return s.astype("Int64").astype("string").fillna("NA")

    card_id = key(df["card1"]) + "_" + key(df["addr1"]) + "_" + key(account_open)

    # A transaction with no card1 cannot be attributed to an account at all.
    # Give each one its own singleton id so they stay separate rather than
    # fusing into a single fictitious account.
    no_card = df["card1"].isna()
    singleton = "NOCARD_" + pd.Series(df.index, index=df.index).astype("string")
    df["card_id"] = card_id.where(~no_card, singleton)
    df["card_id_ablation"] = key(df["card1"]).where(~no_card, singleton)

    # Device fingerprint. str.cat is vectorised; a row-wise join over 590k rows
    # is both slow and, under pandas 3, type-unsafe.
    parts = [df[c].astype("string").fillna("nan") for c in DEVICE_FIELDS]
    fp = parts[0]
    for part in parts[1:]:
        fp = fp.str.cat(part, sep="|", na_rep="nan")

    has_identity = df[DEVICE_FIELDS].notna().any(axis=1)
    df["device_id"] = fp.where(has_identity, pd.NA)
    df["has_device"] = has_identity.astype(int)
    return df


def preprocess(df: pd.DataFrame, max_missing: float = 0.90) -> tuple[pd.DataFrame, list[str]]:
    """Clean, encode and scale. Returns (frame, feature column names).

    Columns missing in more than `max_missing` of rows are dropped; several
    identity fields exceed 99% missing after the masking Vesta applied before
    release, and keeping them adds noise without signal.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    drop = ["TransactionID", "isFraud", "card_id", "card_id_ablation",
            "device_id", "has_device"]
    feat = [c for c in df.columns if c not in drop]

    missing = df[feat].isna().mean()
    feat = [c for c in feat if missing[c] <= max_missing]

    # pandas 3 returns StringDtype rather than object for text, so an
    # `== object` test silently classifies string columns as numeric.
    # Missingness is itself signal in this dataset: transactions with a missing
    # addr1 carry roughly five times the base fraud rate. The graph encodes that
    # structurally, through the NA sentinel in the card key, so the baselines
    # must receive it explicitly as well. Otherwise any GNN advantage partly
    # reflects information the baselines were denied, and the like-for-like
    # claim in Section 3.6 does not hold.
    informative = [c for c in feat if 0.01 <= missing[c] <= max_missing]
    for c in informative:
        df[f"{c}_isna"] = df[c].isna().astype(float)
    feat = feat + [f"{c}_isna" for c in informative]
    print(f"missingness indicators added={len(informative)}")

    cat = [c for c in feat if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in feat if c not in cat]

    for c in cat:                       # frequency encoding: no target leakage
        df[c] = df[c].map(df[c].value_counts(normalize=True)).fillna(0.0)

    df[num] = df[num].fillna(df[num].median())

    from sklearn.preprocessing import StandardScaler
    df[feat] = StandardScaler().fit_transform(df[feat].astype(float))

    print(f"features retained={len(feat)} (dropped {len(missing)-len(feat)} "
          f"over {max_missing:.0%} missing)")
    return df, feat


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test by time, not at random.

    Section 3.7.1. A deployed model predicts on future transactions, so a
    random split would let information about later transactions inform
    predictions about earlier ones. In a graph setting this compounds, because
    edges cross the boundary - which is why graph.py additionally masks
    message passing (Section 3.5).
    """
    n = len(df)
    a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    split = np.array(["train"] * n, dtype=object)
    split[a:b] = "val"
    split[b:] = "test"
    df = df.copy()
    df["split"] = split

    rates = df.groupby("split")["isFraud"].agg(["size", "mean"])
    print(rates.rename(columns={"size": "n", "mean": "fraud_rate"}))
    return df


def main() -> None:
    set_seed(SEED)
    print("[1/4] loading")
    df = load_raw()

    if DATA_FRACTION < 1.0:
        # earliest N% by time, so the temporal structure survives
        df = df.sort_values("TransactionDT").head(int(len(df) * DATA_FRACTION))
        print(f"DATA_FRACTION={DATA_FRACTION}: using earliest {len(df):,} transactions")

    print("[2/4] entity keys")
    df = add_entity_keys(df)

    print("[3/4] preprocessing")
    df, feat = preprocess(df)

    print("[4/4] temporal split")
    df = temporal_split(df)

    stats = {
        "n_transactions": int(len(df)),
        "n_fraud": int(df["isFraud"].sum()),
        "fraud_rate": round(float(df["isFraud"].mean()), 5),
        "n_card_entities": int(df["card_id"].nunique()),
        "n_card_entities_ablation": int(df["card_id_ablation"].nunique()),
        "n_device_entities": int(df["device_id"].nunique(dropna=True)),
        "device_coverage": round(float(df["has_device"].mean()), 4),
        "n_features": len(feat),
    }
    stats["run_profile"] = run_profile()
    save_json(stats, "04_02_dataset_statistics")

    sizes = df.groupby("card_id").size()
    save_table(sizes.describe().to_frame("transactions_per_card").round(2),
               "04_03_card_entity_size_distribution", index=True)

    df.to_parquet(DATA / "prepared.parquet", index=False)
    pd.Series(feat).to_csv(DATA / "feature_columns.csv", index=False, header=False)
    print(f"\nprepared.parquet written: {len(df):,} rows x {len(feat)} features")


if __name__ == "__main__":
    main()
