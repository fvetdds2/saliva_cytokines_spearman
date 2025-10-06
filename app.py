# streamlit_app.py  (preloads UO1_149.xlsx if available)
import io, os, zipfile
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --- keep the helper funcs from the previous app unchanged (excel_col_to_index, p_adjust, load_excel, compute_spearman_by_group, make_group_heatmap) ---

st.set_page_config(page_title="Spearman Correlations by Group", layout="wide")
st.title("Spearman Correlation: SFR_1 vs Cytokines (from Excel column BJ onward) by Group")

with st.sidebar:
    st.header("Settings")
    group_col = st.text_input("Group column", value="Group_ID")
    target_col = st.text_input("Target column", value="SFR_1")
    start_col_label = st.text_input("Start cytokine column (Excel letter)", value="BJ")
    fdr_method = st.selectbox("Multiple testing correction",
                              ["fdr_bh", "bonferroni", "holm", "fdr_by"], index=0)
    show_top_n = st.number_input("Top hits per group", 1, 100, 10, 1)

# --- NEW: prefer local file named UO1_149.xlsx if found ---
default_path_candidates = [
    Path("UO1_149.xlsx"),
    Path("/mnt/data/UO1_149.xlsx"),
    Path("/mnt/data/UO1_149.xls"),
]
existing_default = next((p for p in default_path_candidates if p.exists()), None)

source = st.radio("Choose data source", ["Use local UO1_149.xlsx", "Upload Excel"],
                  index=0 if existing_default else 1, horizontal=True)

uploaded = None
file_bytes = None

if source == "Use local UO1_149.xlsx":
    if not existing_default:
        st.warning("Couldn’t find **UO1_149.xlsx**. Please upload your file instead.")
        source = "Upload Excel"
    else:
        file_bytes = existing_default.read_bytes()
else:
    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        file_bytes = uploaded.read()

if file_bytes is None:
    st.info("Provide an Excel file to begin.")
    st.stop()

# Load workbook & pick sheet
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
with st.expander("Preview (first 25 rows)"):
    st.dataframe(df.head(25), use_container_width=True)

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
    st.dataframe(results.sort_values(["Group_ID", "q_value", "p_value"], na_position="last"),
                 use_container_width=True)

    # Download CSV
    csv_buf = io.StringIO(); results.to_csv(csv_buf, index=False)
    st.download_button("Download CSV", csv_buf.getvalue(),
                       file_name="spearman_SFR1_cytokines_by_group.csv", mime="text/csv")

    # Top hits
    st.subheader(f"Top {show_top_n} associations per group")
    top = (results.dropna(subset=["q_value"])
           .sort_values(["Group_ID", "q_value", "p_value", "Spearman_rho"])
           .groupby("Group_ID", as_index=False).head(show_top_n))
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
    st.download_button("Download all heatmaps (ZIP)", zip_buf, "spearman_heatmaps.zip", "application/zip")

with st.expander("Notes & assumptions"):
    st.markdown(
        "- From Excel column **BJ** to the end is treated as cytokines; non-numeric columns are skipped.\n"
        "- Spearman correlation per **Group_ID** against **SFR_1** with pairwise complete observations.\n"
        "- Multiple-testing correction applied **within each group** (BH/FDR default). N<3 → NaN.\n"
    )
