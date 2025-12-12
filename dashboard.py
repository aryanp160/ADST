import streamlit as st
import pandas as pd
import numpy as np
import json, os, time
import matplotlib.pyplot as plt

LOG_PATH = "logs/adst_coord.log"
MODEL_PATH = "trained_model.npy"

st.set_page_config(page_title="ADST Mission Control", layout="wide", page_icon="🛡️")

# ---------------------------------------
# Custom CSS for "Sci-Fi/Tech" look
# ---------------------------------------
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #111;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0,255,200,0.05);
    }
    .stMetric label { color: #888; }
    .stMetric .css-1wivap2 { color: #00ffc8; } /* Metric value */

    /* Cards */
    .worker-card {
        background-color: #1a1a2e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 5px;
    }
    .status-active { color: #00ff00; font-weight: bold; }
    .status-inactive { color: #ff4444; font-weight: bold; }
    
    /* Logs */
    .log-entry {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        padding: 2px 0;
        border-bottom: 1px solid #222;
    }
    .log-sys { color: #00bfff; }
    .log-wrk { color: #aaaaaa; }
    .log-err { color: #ff5555; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# JS Refresh Logic
# ---------------------------------------
refresh_rate = 1000  # 1 second for smoother feel
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
refresh_button = st.button("Refresh", key="data_refresh_button", help="hidden")

# ---------------------------------------
# Helpers
# ---------------------------------------
def load_logs():
    if not os.path.exists(LOG_PATH): return pd.DataFrame()
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-3000:]
    parsed = []
    for ln in lines:
        try:
            parsed.append(json.loads(ln))
        except: pass
    if not parsed: return pd.DataFrame()
    df = pd.DataFrame(parsed)
    # Ensure TS is datetime
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    return df

def get_worker_status(df, worker_id, current_time):
    # Check last activity for this worker
    w_logs = df[df["worker"] == worker_id]
    if w_logs.empty: return "Offline", "grey"
    
    last_act = w_logs["ts"].max()
    seconds_ago = (current_time - last_act).total_seconds()
    
    if seconds_ago < 5: return "Active", "#00ff00"
    elif seconds_ago < 15: return "Idle", "#ffff00"
    else: return "Offline", "#ff4444"

# ---------------------------------------
# UI Layout
# ---------------------------------------
st.title("🛡️ ADST Federated Learning System")
st.markdown("### Secure Aggregation • Differential Privacy • Real-Time")

placeholder = st.empty()

# ---------------------------------------
# Render Loop
# ---------------------------------------
with placeholder.container():
    df = load_logs()
    
    if df.empty:
        st.warning("Waiting for system logs...")
        st.stop()

    now = pd.Timestamp.now(tz="UTC")

    # --- Header Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. Epoch
    epoch_rows = df[df["event"].str.contains("epoch_sent|epoch_rotate", na=False)]
    curr_epoch = int(epoch_rows["epoch"].max()) if not epoch_rows.empty else 0
    col1.metric("Current Epoch", f"{curr_epoch}/3")
    
    # 2. Status
    active_mask = (now - df["ts"]) < pd.Timedelta(seconds=10)
    recent_activity = len(df[active_mask])
    sys_status = "training" if recent_activity > 0 else "idle"
    col2.metric("System Status", sys_status.upper(), delta="Live" if sys_status=="training" else "Waiting")

    # 3. Aggregations
    total_aggs = len(df[df["event"]=="aggregated_gradient"])
    col3.metric("Global Updates", total_aggs)
    
    # 4. DP Noise
    col4.metric("Privacy Guarantee", "ε-DP", "Active")

    st.divider()

    # --- Network Grid ---
    st.subheader("🌐 Network Status")
    
    # Known workers usually 1-4 for this demo
    worker_ids = [1, 2, 3, 4] 
    cols = st.columns(len(worker_ids))
    
    for i, wid in enumerate(worker_ids):
        status, color = get_worker_status(df, wid, now)
        with cols[i]:
            st.markdown(f"""
            <div class="worker-card" style="border-color: {color}; box-shadow: 0 0 5px {color};">
                <h3>Worker {wid}</h3>
                <p style="color:{color}; font-size:1.2em;">● {status}</p>
                <div style="font-size:0.8em; color:#888;">ID: site{wid}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Charts & Visualization ---
    st.divider()
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📉 Training Performance")
        agg = df[df["event"] == "aggregated_gradient"]
        val = df[df["event"] == "validation_accuracy"]
        
        if not agg.empty:
            # Dual axis plot
            fig, ax1 = plt.subplots(figsize=(8,3), dpi=100)
            fig.patch.set_facecolor('#0E1117')
            ax1.set_facecolor('#0E1117')
            
            # Gradient Norm
            ax1.plot(agg["epoch"], agg["grad_norm"], color="#00ffc8", marker="o", label="Gradient Norm")
            ax1.set_xlabel("Epoch", color="white")
            ax1.set_ylabel("Norm", color="#00ffc8")
            ax1.tick_params(axis='x', colors='white')
            ax1.tick_params(axis='y', colors='white')
            ax1.grid(True, color="#333", linestyle="--")
            
            # Val Acc on secondary axis
            if not val.empty:
                ax2 = ax1.twinx()
                ax2.plot(val["epoch"], val["acc"], color="#ff00ff", marker="s", linestyle="--", label="Val Accuracy")
                ax2.set_ylabel("Accuracy", color="#ff00ff")
                ax2.tick_params(axis='y', colors='white')
                ax2.set_ylim(0, 1.0)
            
            st.pyplot(fig)
        else:
            st.info("Waiting for first aggregation...")

    with c2:
        st.subheader("📝 Live Event Feed")
        # Filter important events
        events = df.sort_values("ts", ascending=False).head(15)
        html_logs = ""
        for _, row in events.iterrows():
            evt = row['event']
            comp = row.get('component', 'sys')
            ts = row['ts'].strftime('%H:%M:%S')
            
            color_class = "log-sys"
            if comp == "worker": color_class = "log-wrk"
            if "error" in evt or "fail" in evt or "nack" in evt: color_class = "log-err"
            
            icon = "🔹"
            if "gradient" in evt: icon = "📦"
            if "epoch" in evt: icon = "🔄"
            if "error" in evt: icon = "⚠️"
            
            # Simplified message
            msg = f"{evt}"
            if "worker" in row and not pd.isna(row["worker"]):
                msg += f" (Worker {int(row['worker'])})"
                
            html_logs += f"<div class='log-entry {color_class}'>{ts} {icon} {msg}</div>"
            
        st.markdown(f"<div style='height: 300px; overflow-y: auto; background-color: #111; padding:10px; border-radius:5px;'>{html_logs}</div>", unsafe_allow_html=True)

    # --- Footer Usage Hint ---
    st.divider()
    st.info("💡 **Tip**: To use the trained model, run `python use_model.py` in your terminal.")
