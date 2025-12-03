import streamlit as st
import pandas as pd
import numpy as np
import json, os, time
import matplotlib.pyplot as plt

LOG_PATH = "logs/adst_coord.log"
MODEL_PATH = "trained_model.npy"

st.set_page_config(page_title="ADST Dashboard", layout="wide", page_icon="🛡️")

# Custom CSS for "Tech" look
st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #303030;
        padding: 10px;
        border-radius: 5px;
    }
    .stAlert {
        background-color: #0E1117;
        border: 1px solid #303030;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# JS trick: background refresh ONLY data
# ---------------------------------------
refresh_rate = 2000  # 2 seconds

st.markdown(
    f"""
    <script>
    function refreshData() {{
        var btn = document.getElementById("data_refresh_button");
        if (btn) {{ btn.click(); }}
    }}
    setInterval(refreshData, {refresh_rate});
    </script>
    """,
    unsafe_allow_html=True
)

# Hidden button to refresh ONLY data
refresh_button = st.button("Refresh Data", key="data_refresh_button", help="hidden")


# ---------------------------------------
# Helper functions
# ---------------------------------------

def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-2000:]
    parsed = []
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except:
            pass
    return pd.DataFrame(parsed)

def load_model_stats():
    if not os.path.exists(MODEL_PATH):
        return None
    arr = np.load(MODEL_PATH)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "norm": float(np.linalg.norm(arr)),
    }


# ---------------------------------------
# Static UI (never refreshes)
# ---------------------------------------

st.title("🚀 ADST 2.0 Federated Learning Dashboard")
st.caption(f"Real-time monitoring • Refresh every {refresh_rate/1000}s")

metric_box = st.empty()
workers_box = st.empty()
charts_box = st.empty()
logs_box = st.empty()

# ---------------------------------------
# Data refresh section (ONLY this part reruns)
# ---------------------------------------

df = load_logs()

if df.empty:
    st.info("Waiting for first logs…")
    st.stop()

# --------- Top Metrics ---------
with metric_box.container():
    col1, col2, col3, col4 = st.columns(4)

    # Epoch
    epoch_rows = df[df["event"].str.contains("epoch_sent|epoch_rotate", na=False)]
    current_epoch = int(epoch_rows["epoch"].max()) if not epoch_rows.empty else 0
    col1.metric("Current Epoch", current_epoch)

    # Active Workers
    worker_ids = sorted(df[df["event"] == "worker_hello"]["worker"].unique().tolist())
    col2.metric("Active Workers", len(worker_ids))

    # Privacy Status
    col4.metric("Privacy Mode", "🛡️ Active", delta="DP + SecAgg")

    # Model stats
    stats = load_model_stats()
    if stats:
        col3.metric("Mean Weight", f"{stats['mean']:.6f}")
    else:
        col3.write("Model not saved yet")


# --------- Worker Table ---------
with workers_box.container():
    st.subheader("👥 Active Workers")
    st.write(df[df["event"] == "worker_hello"][["worker", "ts"]].drop_duplicates())


# --------- Charts ---------
with charts_box.container():
    st.subheader("📉 Training Curves (Compact)")

    agg = df[df["event"] == "aggregated_gradient"]
    val = df[df["event"] == "validation_accuracy"]

    colA, colB, colC = st.columns(3)

    if not agg.empty:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
        ax.plot(agg["epoch"], agg["mean_gradient"], marker="o", markersize=4)
        ax.set_title("Mean Gradient", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        colA.pyplot(fig)

    if not val.empty:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
        ax.plot(val["epoch"], val["acc"], marker="o", markersize=4, color="green")
        ax.set_title("Validation Accuracy", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        colB.pyplot(fig)

    if not agg.empty and "grad_norm" in agg.columns:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
        ax.plot(agg["epoch"], agg["grad_norm"], marker="o", markersize=4, color="orange")
        ax.set_title("Gradient Norm (L2)", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Norm", fontsize=8)
        colC.pyplot(fig)


# --------- Logs ---------
with logs_box.container():
    st.subheader("📄 Recent Logs")
    st.dataframe(df.tail(80), height=250)
