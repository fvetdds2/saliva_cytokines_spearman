# streamlit_app.py
# Option B: open ExcelFile directly (no caching)
# Computes Spearman correlations: SFR_1 vs cytokines (from Excel column BJ onward), split by Group_ID

import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Prefer SciPy for exact Spearman; fall back if missing
try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# -------------------------------
# Helpers
# -------------------------------
def excel_col_to_index(label: str) -> int:
    """Excel letters ('A', 'AA', 'BJ', ...) -> 0-based index."""
    label = label.strip().upper()
    n = 0
    for c in label:
        if not ('A' <= c <= 'Z'):
            raise ValueError(f"Invalid Excel column label: {label}")
        n = n * 26 + (ord(c) - 64)
    return n - 1

def p_adjust(pvals: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """
    Multiple-testing correction. Supported:
    'fdr_bh' (BH/FDR), 'bonferroni', 'holm', 'fdr_by'.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    msk = np.isfinite(p)
    if msk.sum() == 0:
        return q
    pv = p[msk]
    m = pv.size
    order = np.argsort(pv)
    pv_sorted = pv[order]

    if method == "bonferroni":
        q_sorted = np.minimum(pv_sorted * m, 1.0)
    elif method == "holm":
        q_sorted = np.minimum.accumulate((pv_sorted[::-1] * np.arange(1, m + 1))[::-1])
        q_sorted = np.minimum(q_sorted, 1.0)
    elif method == "fdr_bh":
        ranks = np.arange(1, m + 1)
        q_sorted = pv_sorted * m / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
    elif method == "fdr_by":
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        ranks = np.arange(1, m + 1)
        q_sorted = pv_sorted * m * c_m / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
    else:
        raise ValueError(f"Unknown method '{method}'")

    out = np.full_like(p, np.nan, dtype=float)
    out_idx = np.where(msk)[0][order]
    out[out_idx] = q_sorted
    return out

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks for ties (fallback if SciPy missing)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks

def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0.0:
        return np.nan
    return float((x * y).sum() / denom)

def _spearman_no_scipy(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho via rank correlation; p-value via normal approx (n>=10)."""
    n = x.size
    if n < 3:
        return (np.nan, np.nan)
    rx, ry = _rankdata(x), _rankdata(y)
    rho = _pearsonr(rx, ry)
    if np.isnan(rho) or n < 10:
        return (rho, np.nan)
    z = rho * np.sqrt(n - 1)
    # two-sided p from normal tail
    p = float(np.math.erfc(abs(z) / np.sqrt(2.0)))
    return (rho, p)

def compute_spearman_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    start_col_label: str = "BJ",
    fdr_method: str = "fdr_bh"
) -> tuple[pd.DataFrame, list[str]]:
    # basic checks
    for col in (group_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found.")

    # identify cytokine columns from BJ onward
    start_idx = excel_col_to_index(start_col_label)
    if start_idx >= len(df.columns):
        raise ValueError(f"Start column '{start_col_label}' (index {start_idx}) is beyond last column.")
    candidate_cols = list(df.columns[start_idx:])
    numeric_mask = df[candidate_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=0)
    cytokine_cols = [c for c in candidate_cols if bool(numeric_mask.get(c, False))]
    if not cytokine_cols:
        raise ValueError(f"No numeric cytokine columns found from '{start_col_label}' onward.")

    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    rows = []
    for group, gdf in df.groupby(group_col, dropna=False):
        x = pd.to_numeric(gdf[target_col], errors="coerce")
        for cyto in cytokine_cols:
            y = pd.to_numeric(gdf[cyto], errors="coerce")
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n < 3:
                rho, p = np.nan, np.nan
            else:
                if HAVE_SCIPY:
                    rho, p = spearmanr(x[mask], y[mask])
                else:
                    rho, p = _spearman_no_scipy(x[mask].to_numpy(), y[mask].to_numpy())
            rows.append(
                {"Group_ID": group, "Cytokine": cyto, "N_pairs": n, "Spearman_rho": rho, "p_value": p}
            )

    res = pd.DataFrame(rows)

    # adjust p-values within each group
    def add_adj(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["q_value"] = p_adjust(sub["p_value"].values, method=fdr_method)
        return sub

    res = res.groupby("Group_ID", group_keys=False).apply(add_adj)
    res = res[["Group_ID", "Cytokine", "N_pairs", "Spearman_rho", "p_value", "q_value"]]
    return res, cytokine_cols

def make_group_heatmap(df_group: pd.DataFrame, target_col: str, group_value) -> io.BytesIO:
    """Return a PNG buffer for the group's rho heatmap."""
    pivot = df_group.set_index("Cytokine")["Spearman_rho"].to_frame()
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks([0])
    ax.set_xticklabels(["rho"])
    ax.set_title(f"Spearman rho: SFR_1 vs Cytokines\nGroup: {group_value}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman rho")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Spearman Correlations by Group", layout="wide")
st.title("Spearman Correlation: SFR_1 vs Cytokines (from Excel column BJ onward) by Group")

with st.sidebar:
    st.header("Settings")
    group_col = st.text_input("Group column", value="Group_ID")
    target_col = st.text_input("Target column", value="SFR_1")
    start_col_label = st.text_input("Start cytokine column (Excel letter)", value="BJ")
    fdr_method = st.selectbox("Multiple testing correction",
                              ["fdr_bh", "bonferroni", "holm", "fdr_by"], index=0)
    show_top_n = st.number_input("Top hits per group", min_value=1, max_value=100, value=10, step=1)

# Prefer a local default file if present
default_candidates = [Path("UO1_149.xlsx"), Path("/mnt/data/UO1_149.xlsx")]
existing_default = next((p for p in default_candidates if p.exists()), None)

source = st.radio(
    "Choose data source",
    ["Use local UO1_149.xlsx", "Upload Excel"],
    index=0 if existing_default else 1,
    horizontal=True
)

file_bytes: bytes | None = None
if source == "Use local UO1_149.xlsx":
    if existing_default is None:
        st.warning("Couldn't find **UO1_149.xlsx**. Please upload a file instead.")
        source = "Upload Excel"
    else:
        file_bytes = existing_default.read_bytes()

if source == "Upload Excel":
    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        file_bytes = uploaded.read()

if file_bytes is None:
    st.info("Provide an Excel file to begin.")
    st.stop()

# ---- OPTION B: open ExcelFile directly (no caching) ----
try:
    xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

sheet_name = st.selectbox("Choose worksheet", options=xl.sheet_names, index=0)
try:
    df = xl.parse(sheet_name)
except Exception as e:
    st.error(f"Failed to parse sheet '{sheet_name}': {e}")
    st.stop()

st.caption(f"Loaded **{sheet_name}** with shape {df.shape[0]} rows × {df.shape[1]} columns.")
with st.expander("Preview data (first 25 rows)"):
    st.dataframe(df.head(25), use_container_width=True)

# Compute
if st.button("Run Spearman analysis", type="primary"):
    try:
        results, cytokine_cols = compute_spearman_by_group(
            df, group_col=group_col, target_col=target_col,
            start_col_label=start_col_label, fdr_method=fdr_method
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.success(f"Detected {len(cytokine_cols)} cytokine columns starting from '{start_col_label}'.")

    st.subheader("All results")
    st.dataframe(
        results.sort_values(["Group_ID", "q_value", "p_value"], na_position="last"),
        use_container_width=True,
    )

    # Download CSV
    csv_buf = io.StringIO()
    results.to_csv(csv_buf, index=False)
    st.download_button(
        "Download results as CSV",
        data=csv_buf.getvalue(),
        file_name="spearman_SFR1_cytokines_by_group.csv",
        mime="text/csv",
    )

    # Top hits per group
    st.subheader(f"Top {show_top_n} associations per group (by q-value, then p-value)")
    top = (
        results.dropna(subset=["q_value"])
        .sort_values(["Group_ID", "q_value", "p_value", "Spearman_rho"])
        .groupby("Group_ID", as_index=False)
        .head(show_top_n)
    )
    st.dataframe(top, use_container_width=True)

    # Heatmaps + ZIP
    st.subheader("Per-group heatmaps (Spearman ρ)")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for group_val, gdf in results.groupby("Group_ID"):
            img_buf = make_group_heatmap(gdf, target_col=target_col, group_value=group_val)
            st.image(img_buf, caption=f"Group: {group_val}", use_column_width=True)
            zf.writestr(f"spearman_heatmap_{str(group_val).replace(' ', '_')}.png", img_buf.getvalue())
    zip_buf.seek(0)
    st.download_button(
        "Download all heatmaps (ZIP)",
        data=zip_buf,
        file_name="spearman_heatmaps.zip",
        mime="application/zip",
    )

with st.expander("Notes & assumptions"):
    st.markdown(
        """
- From Excel column **BJ** to the end is treated as cytokines; non-numeric columns are ignored automatically.
- Spearman correlation per **Group_ID** against **SFR_1** using pairwise complete observations.
- Multiple-testing correction applied **within each group** (BH/FDR default).
- Requires **N ≥ 3** pairwise non-missing samples to compute a correlation; otherwise ρ and *p* are NaN.
"""
    )
