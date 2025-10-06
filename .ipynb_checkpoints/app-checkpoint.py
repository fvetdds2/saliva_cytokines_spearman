# streamlit_app.py  (no statsmodels needed)
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

st.set_page_config(page_title="Spearman Correlations by Group", layout="wide")

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
    Multiple-testing correction (vectorized).
    Supported methods: 'fdr_bh' (Benjamini–Hochberg), 'bonferroni', 'holm', 'fdr_by'.
    Returns adjusted p-values (q-values for FDR methods).
    NaNs are preserved.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    q = np.full_like(p, np.nan, dtype=float)

    mask = np.isfinite(p)
    if mask.sum() == 0:
        return q

    pv = p[mask]
    m = pv.size
    order = np.argsort(pv)
    pv_sorted = pv[order]
    q_sorted = np.empty_like(pv_sorted)

    if method == "bonferroni":
        q_sorted = np.minimum(pv_sorted * m, 1.0)
    elif method == "holm":
        # Holm–Bonferroni (step-down)
        q_sorted = np.minimum.accumulate((pv_sorted[::-1] * np.arange(1, m + 1))[::-1])
        q_sorted = np.minimum(q_sorted, 1.0)
    elif method == "fdr_bh":
        # Benjamini–Hochberg (step-up)
        ranks = np.arange(1, m + 1)
        q_sorted = pv_sorted * m / ranks
        # enforce monotonicity
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
    elif method == "fdr_by":
        # Benjamini–Yekutieli (conservative under dependence)
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        ranks = np.arange(1, m + 1)
        q_sorted = pv_sorted * m * c_m / ranks
        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
        q_sorted = np.minimum(q_sorted, 1.0)
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Unsort back to original mask positions
    q_vals = np.full_like(p, np.nan, dtype=float)
    q_vals_idx = np.where(mask)[0][order]
    q_vals[q_vals_idx] = q_sorted
    return q_vals

@st.cache_data(show_spinner=False)
def load_excel(file_bytes) -> pd.ExcelFile:
    return pd.ExcelFile(file_bytes, engine="openpyxl")

def compute_spearman_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    start_col_label: str = "BJ",
    fdr_method: str = "fdr_bh"
) -> tuple[pd.DataFrame, list[str]]:
    # Checks
    for col in (group_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data.")

    # Find cytokine columns by position (from BJ to end)
    start_idx = excel_col_to_index(start_col_label)
    if start_idx >= len(df.columns):
        raise ValueError(
            f"Start column '{start_col_label}' (index {start_idx}) is beyond the last column ({len(df.columns)-1})."
        )
    candidate_cols = list(df.columns[start_idx:])

    # Keep only columns that have at least some numeric values
    numeric_mask = df[candidate_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=0)
    cytokine_cols = [c for c in candidate_cols if numeric_mask.get(c, False)]
    if not cytokine_cols:
        raise ValueError(f"No numeric cytokine columns found from '{start_col_label}' onward.")

    # Ensure target numeric
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
                rho, p = spearmanr(x[mask], y[mask])
            rows.append(
                {
                    "Group_ID": group,
                    "Cytokine": cyto,
                    "N_pairs": n,
                    "Spearman_rho": rho,
                    "p_value": p,
                }
            )

    res = pd.DataFrame(rows)

    # FDR/Family-wise correction within each group
    def add_adj(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        sub["q_value"] = p_adjust(sub["p_value"].values, method=fdr_method)
        return sub

    res = res.groupby("Group_ID", group_keys=False).apply(add_adj)
    res = res[["Group_ID", "Cytokine", "N_pairs", "Spearman_rho", "p_value", "q_value"]]
    return res, cytokine_cols

def make_group_heatmap(df_group: pd.DataFrame, target_col: str, group_value) -> io.BytesIO:
    """Return a PNG image buffer for the group's rho heatmap."""
    pivot = df_group.set_index("Cytokine")["Spearman_rho"].to_frame()
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks([0])
    ax.set_xticklabels(["rho"])
    ax.set_title(f"Spearman rho: {target_col} vs Cytokines\nGroup: {group_value}")
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
st.title("Spearman Correlation: SFR_1 vs Cytokines (from Excel column BJ onward) by Group")

with st.sidebar:
    st.header("Settings")
    group_col = st.text_input("Group column", value="Group_ID")
    target_col = st.text_input("Target column", value="SFR_1")
    start_col_label = st.text_input("Start cytokine column (Excel letter)", value="BJ")
    fdr_method = st.selectbox(
        "Multiple testing correction",
        ["fdr_bh", "bonferroni", "holm", "fdr_by"],  # all implemented here
        index=0
    )
    show_top_n = st.number_input("Top hits per group to display", min_value=1, max_value=100, value=10, step=1)

uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file to begin (e.g., your UO1 data sheet).")
    st.stop()

# Load workbook & pick sheet
try:
    xl = load_excel(uploaded)
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

# Optionally preview data
with st.expander("Preview data (first 25 rows)"):
    st.dataframe(df.head(25), use_container_width=True)

# Compute
run_btn = st.button("Run Spearman analysis", type="primary")
if not run_btn:
    st.stop()

try:
    results, cytokine_cols = compute_spearman_by_group(
        df, group_col=group_col, target_col=target_col, start_col_label=start_col_label, fdr_method=fdr_method
    )
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Done. Detected {len(cytokine_cols)} cytokine columns starting from '{start_col_label}'.")

# Show results table
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

# Top hits by group
st.subheader(f"Top {show_top_n} associations per group (by q-value, then p-value)")
top = (
    results.dropna(subset=["q_value"])
    .sort_values(["Group_ID", "q_value", "p_value", "Spearman_rho"])
    .groupby("Group_ID", as_index=False)
    .head(show_top_n)
)
st.dataframe(top, use_container_width=True)

# Heatmaps
st.subheader("Per-group heatmaps (Spearman ρ)")
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for group_val, gdf in results.groupby("Group_ID"):
        img_buf = make_group_heatmap(gdf, target_col=target_col, group_value=group_val)
        st.image(img_buf, caption=f"Group: {group_val}", use_column_width=True)
        # Add to ZIP
        zf.writestr(f"spearman_heatmap_{str(group_val).replace(' ', '_')}.png", img_buf.getvalue())
zip_buf.seek(0)

st.download_button(
    "Download all heatmaps (ZIP)",
    data=zip_buf,
    file_name="spearman_heatmaps.zip",
    mime="application/zip",
)

# Footnotes / Notes
with st.expander("Notes & assumptions"):
    st.markdown(
        """
- Treats everything from Excel column **BJ** to the end as cytokines, automatically keeping columns that contain numeric values.
- Computes **Spearman correlation** between `SFR_1` and each cytokine **within each `Group_ID`** using pairwise complete observations.
- Multiple-testing correction is applied **within each group** (select BH/FDR, Bonferroni, Holm, or BY).
- Requires at least **N ≥ 3** pairwise non-missing samples to compute a correlation; otherwise ρ and *p* are reported as NaN.
- If your cytokines end before the last column, using 'BJ' still works; non-numeric trailing columns are ignored automatically.
"""
    )
