"""Dataset analytics for the dashboard and retention insights.

Everything here is computed from data/Churn_Modelling.csv. These are
associations in one historical dataset, not proof of what causes churn.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

AGE_BINS = [0, 29, 39, 49, 59, 200]
AGE_LABELS = ["Under 30", "30 to 39", "40 to 49", "50 to 59", "60 and over"]

# Dimensions shown in segment analysis: (column label, how to derive it)
DIMENSIONS = ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard", "Age band", "Balance"]
DIMENSION_TITLES = {
    "Geography": "Country", "Gender": "Gender", "NumOfProducts": "Number of products",
    "IsActiveMember": "Active member", "HasCrCard": "Has credit card",
    "Age band": "Age band", "Balance": "Account balance",
}


def with_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Age band"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    out["Balance"] = np.where(out["Balance"] > 0, "Has balance", "Zero balance")
    out["IsActiveMember"] = out["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    out["HasCrCard"] = out["HasCrCard"].map({1: "Has card", 0: "No card"})
    out["NumOfProducts"] = out["NumOfProducts"].astype(str) + " product(s)"
    return out


def overall_stats(df: pd.DataFrame) -> dict[str, float]:
    churned = int(df["Exited"].sum())
    return {"customers": int(len(df)), "churned": churned,
            "retained": int(len(df) - churned), "churn_rate": churned / len(df)}


def segment_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate per segment with lift versus the overall churn rate."""
    seg = with_segments(df)
    overall = seg["Exited"].mean()
    rows = []
    for dim in DIMENSIONS:
        grouped = seg.groupby(dim)["Exited"].agg(["count", "sum"])
        for label, row in grouped.iterrows():
            rate = row["sum"] / row["count"]
            rows.append({"Dimension": dim, "Segment": str(label), "Customers": int(row["count"]),
                         "Churned": int(row["sum"]), "ChurnRate": float(rate),
                         "Lift": float(rate / overall)})
    return pd.DataFrame(rows)


def segment_chart_data(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    seg = with_segments(df)
    table = (seg.groupby(dimension)["Exited"].agg(Customers="count", Churned="sum").reset_index()
             .rename(columns={dimension: "Segment"}))
    table["Retained"] = table["Customers"] - table["Churned"]
    table["ChurnRate"] = table["Churned"] / table["Customers"]
    if dimension == "Age band":
        table["Segment"] = pd.Categorical(table["Segment"], AGE_LABELS, ordered=True)
        table = table.sort_values("Segment")
    return table


def customer_segments(customer: dict[str, Any]) -> dict[str, str]:
    """Map a validated customer to the segment label it falls into per dimension."""
    age_band = AGE_LABELS[int(np.digitize(customer["Age"], [30, 40, 50, 60]))]
    return {
        "Geography": customer["Geography"],
        "Gender": customer["Gender"],
        "NumOfProducts": f"{customer['NumOfProducts']} product(s)",
        "IsActiveMember": "Active" if customer["IsActiveMember"] else "Inactive",
        "HasCrCard": "Has card" if customer["HasCrCard"] else "No card",
        "Age band": age_band,
        "Balance": "Has balance" if customer["Balance"] > 0 else "Zero balance",
    }


def _overall_rate(rates: pd.DataFrame) -> float:
    """Overall churn rate, taken from a single dimension (each dimension covers every customer once)."""
    one = rates[rates.Dimension == rates.Dimension.iloc[0]]
    return float(one["Churned"].sum() / one["Customers"].sum())


def customer_signals(customer: dict[str, Any], rates: pd.DataFrame, min_customers: int = 100,
                     min_gap_points: float = 3.0, limit: int = 4) -> list[dict[str, Any]]:
    """Historical churn rate of the segments this customer belongs to.

    Only segments with enough customers and a clear gap from the overall rate
    are returned. Sorted by size of the gap.
    """
    overall = _overall_rate(rates)
    out = []
    for dim, segment in customer_segments(customer).items():
        match = rates[(rates.Dimension == dim) & (rates.Segment == segment)]
        if match.empty:
            continue
        row = match.iloc[0]
        gap = (row.ChurnRate - overall) * 100
        if row.Customers >= min_customers and abs(gap) >= min_gap_points:
            out.append({"Dimension": dim, "Segment": segment, "ChurnRate": float(row.ChurnRate),
                        "Customers": int(row.Customers), "GapPoints": float(gap),
                        "Direction": "higher" if gap > 0 else "lower"})
    out.sort(key=lambda r: abs(r["GapPoints"]), reverse=True)
    return out[:limit]


def what_if_variants(customer: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Single-field changes a bank could plausibly influence. Used to probe model sensitivity."""
    variants = []
    if not customer["IsActiveMember"]:
        variants.append(("If the customer became an active member", {**customer, "IsActiveMember": 1}))
    if customer["NumOfProducts"] == 1:
        variants.append(("If the customer held 2 products", {**customer, "NumOfProducts": 2}))
    elif customer["NumOfProducts"] >= 3:
        variants.append(("If the customer held 2 products", {**customer, "NumOfProducts": 2}))
    return variants


def retention_plays(rates: pd.DataFrame, top: int = 4) -> pd.DataFrame:
    """Largest segments with clearly elevated churn, for the retention insights page."""
    candidates = rates[(rates.Customers >= 300) & (rates.Lift >= 1.2)].copy()
    candidates["ExcessChurners"] = candidates["Churned"] - candidates["Customers"] * _overall_rate(rates)
    return candidates.sort_values("ExcessChurners", ascending=False).head(top).reset_index(drop=True)
