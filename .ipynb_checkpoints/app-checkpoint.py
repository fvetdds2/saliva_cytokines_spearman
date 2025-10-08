# streamlit run app.py
import io
import numpy as np, pandas as pd
import streamlit as st

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------- helpers ----------
def excel_idx(label: str) -> int:
    n = 0
    for c in label.strip().upper(): n = n*26 + (ord(c)-64)
    return n-1

def p_adjust(p, method="fdr_bh"):
    p = np.asarray(p, float); out = np.full_like(p, np.nan, float)
    m = np.isfinite(p)
    if not m.any(): return out
    pv = p[m]; order = np.argsort(pv); k = pv.size; ranks = np.arange(1, k+1)
    if method == "bonferroni":
        adj = np.minimum(pv[order]*k, 1.0)
    elif method == "holm":
        adj = np.minimum.accumulate((pv[order][::-1]*ranks)[::-1]).clip(max=1)
    elif method == "fdr_by":
        c_m = np.sum(1/np.arange(1, k+1)); adj = (pv[order]*k*c_m/ranks)
        adj = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1)
    else:  # fdr_bh
        adj = (pv[order]*k/ranks)
        adj = np.minimum.accumulate(adj[::-1])[::-1].clip(max=1)
    out[np.where(m)[0][order]] = adj
    return out

def spearman_fast(x, y):
    x = pd.to_numeric(x, errors="coerce"); y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna(); n = int(m.sum())
    if n < 3: return n, np.nan, np.nan
    if HAVE_SCIPY: rho, p = spearmanr(x[m], y[m])
    else:
        rx, ry = x[m].rank(), y[m].rank()
        rx, ry = rx - rx.mean(), ry - ry.mean()
        denom = np.sqrt((rx**2).sum()*(ry**2).sum())
        rho = float((rx*ry).sum()/denom) if denom else np.nan
        p = np.nan
    return n, rho, p

# ---------- UI ----------
st.set_page_config(page_title="Spearman by Group", layout="wide")
st.title("Spearman: SFR_1 vs Cytokines (by Group)")

with st.sidebar:
    group_col  = st.text_input("Group column", "Group_ID")
    target_col = st.text_input("Target column", "SFR_1")
    start_label = st.text_input("Start cytokine column (Excel letter)", "BJ")
    fdr = st.selectbox("Multiple testing correction", ["fdr_bh","bonferroni","holm","fdr_by"], 0)
    top_n = st.number_input("Top hits per group", 1, 100, 10)

src = st.radio("Data source", ["Upload Excel (.xlsx)","Local UO1_149.xlsx"], index=0, horizontal=True)
file_bytes = None
if src == "Local UO1_149.xlsx":
    from pathlib import Path
    p = Path("UO1_149.xlsx")
    if p.exists(): file_bytes = p.read_bytes()
    else: st.warning("Local UO1_149.xlsx not found. Upload instead.")
if file_bytes is None:
    up = st.file_uploader("Upload Excel", type=["xlsx"])
    if up: file_bytes = up.read()
if not file_bytes:
    st.info("Upload/choose a file to continue."); st.stop()

# read excel
import pandas as pd, io as _io
try:
    xl = pd.ExcelFile(_io.BytesIO(file_bytes), engine="openpyxl")
except Exception as e:
    st.error(f"Failed to read Excel: {e}"); st.stop()
sheet = st.selectbox("Worksheet", xl.sheet_names, index=0)
df = xl.parse(sheet)
st.caption(f"Loaded **{sheet}**: {df.shape[0]} rows × {df.shape[1]} cols.")
st.dataframe(df.head(25), use_container_width=True)

# compute
if st.button("Run analysis", type="primary"):
    for c in (group_col, target_col):
        if c not in df.columns: st.error(f"Missing column: {c}"); st.stop()
    s_idx = excel_idx(start_label)
    if s_idx >= len(df.columns): st.error(f"Start '{start_label}' is beyond last column."); st.stop()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    cand = list(df.columns[s_idx:])
    numeric_any = df[cand].apply(pd.to_numeric, errors="coerce").notna().any()
    cyto_cols = [c for c in cand if bool(numeric_any.get(c, False))]
    if not cyto_cols: st.error("No numeric cytokine columns from start column."); st.stop()
    st.success(f"Detected {len(cyto_cols)} cytokines from '{start_label}' onward.")

    rows = []
    for gval, gdf in df.groupby(group_col, dropna=False):
        for cy in cyto_cols:
            n, rho, p = spearman_fast(gdf[target_col], gdf[cy])
            rows.append(dict(Group_ID=gval, Cytokine=cy, N_pairs=n, Spearman_rho=rho, p_value=p))
    res = pd.DataFrame(rows)

    # within-group FDR
    res = (res.groupby("Group_ID", group_keys=False)
             .apply(lambda s: s.assign(q_value=p_adjust(s["p_value"].values, fdr)))
             .reset_index(drop=True))
    res = res[["Group_ID","Cytokine","N_pairs","Spearman_rho","p_value","q_value"]]

    st.subheader("All results")
    st.dataframe(res.sort_values(["Group_ID","q_value","p_value"]), use_container_width=True)

    # Top-N per group
    st.subheader(f"Top {top_n} per group")
    top = (res.dropna(subset=["q_value"])
             .sort_values(["Group_ID","q_value","p_value","Spearman_rho"])
             .groupby("Group_ID", as_index=False).head(top_n))
    st.dataframe(top, use_container_width=True)

    # CSV download
    csv = io.StringIO(); res.to_csv(csv, index=False)
    st.download_button("Download CSV", data=csv.getvalue(),
        file_name="spearman_SFR1_cytokines_by_group.csv", mime="text/csv")

st.caption("Spearman per Group_ID vs SFR_1; cytokines from Excel col ‘BJ’ onward; FDR within group; N≥3 required.")
