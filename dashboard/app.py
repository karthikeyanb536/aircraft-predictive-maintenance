import streamlit as st
import requests
import plotly.graph_objects as go

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Aircraft Engine Monitor",
    page_icon="✈️",
    layout="wide"
)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("✈️ Aircraft Engine Predictive Maintenance")
st.markdown("Real-time Remaining Useful Life (RUL) Prediction using NASA C-MAPSS data")
st.divider()

# ─────────────────────────────────────────
# API URL
# ─────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ─────────────────────────────────────────
# SIDEBAR — SENSOR INPUTS
# ─────────────────────────────────────────
st.sidebar.header("⚙️ Engine Sensor Inputs")
st.sidebar.markdown("Adjust sensor readings:")

cycle        = st.sidebar.slider("Cycle",          1,    362,   50)
op_setting_1 = st.sidebar.slider("Op Setting 1",   -0.01, 0.01, 0.0023, step=0.0001, format="%.4f")
op_setting_2 = st.sidebar.slider("Op Setting 2",   -0.01, 0.01, 0.0003, step=0.0001, format="%.4f")
op_setting_3 = st.sidebar.slider("Op Setting 3",   60.0, 100.0, 100.0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sensor Readings:**")

sensor_2  = st.sidebar.slider("Sensor 2  (Fan Inlet Temp)",      641.0, 645.0, 642.0)
sensor_3  = st.sidebar.slider("Sensor 3  (LPC Outlet Temp)",    1570.0, 1620.0, 1590.0)
sensor_4  = st.sidebar.slider("Sensor 4  (HPC Outlet Temp)",    1380.0, 1450.0, 1400.0)
sensor_7  = st.sidebar.slider("Sensor 7  (Fan Inlet Pressure)",  549.0, 557.0,  554.0)
sensor_9  = st.sidebar.slider("Sensor 9  (Physical Fan Speed)", 9000.0, 9200.0, 9065.0)
sensor_11 = st.sidebar.slider("Sensor 11 (Bypass Ratio)",        47.0,  48.5,   47.5)
sensor_12 = st.sidebar.slider("Sensor 12 (Burner Fuel-Air)",    519.0, 523.0,  521.0)
sensor_14 = st.sidebar.slider("Sensor 14 (HPT Coolant Bleed)",  8100.0, 8200.0, 8140.0)
sensor_17 = st.sidebar.slider("Sensor 17 (Turbine Inlet Temp)", 388.0, 400.0,  392.0)
sensor_20 = st.sidebar.slider("Sensor 20 (Bypass Ratio 2)",      38.0,  39.5,   38.8)
sensor_21 = st.sidebar.slider("Sensor 21 (Fan Speed Ratio)",     23.0,  23.7,   23.2)

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
predict_btn = st.sidebar.button("🔍 Predict RUL", use_container_width=True)

# ─────────────────────────────────────────
# GAUGE CHART FUNCTION
# ─────────────────────────────────────────
def create_gauge(rul):
    if rul <= 30:
        color = "red"
    elif rul <= 60:
        color = "orange"
    elif rul <= 100:
        color = "yellow"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rul,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Remaining Useful Life (Cycles)", 'font': {'size': 20}},
        delta={'reference': 125, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 125], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30],   'color': '#ffcccc'},
                {'range': [30, 60],  'color': '#ffe5cc'},
                {'range': [60, 100], 'color': '#ffffcc'},
                {'range': [100, 125],'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    fig.update_layout(height=350)
    return fig

# ─────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────
col1, col2 = st.columns([1.5, 1])

with col1:
    if predict_btn:
        payload = {
            "cycle": cycle,
            "op_setting_1": op_setting_1,
            "op_setting_2": op_setting_2,
            "op_setting_3": op_setting_3,
            "sensor_2": sensor_2,
            "sensor_3": sensor_3,
            "sensor_4": sensor_4,
            "sensor_7": sensor_7,
            "sensor_9": sensor_9,
            "sensor_11": sensor_11,
            "sensor_12": sensor_12,
            "sensor_14": sensor_14,
            "sensor_17": sensor_17,
            "sensor_20": sensor_20,
            "sensor_21": sensor_21
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            rul        = result['predicted_rul']
            status     = result['status']
            confidence = result['confidence']

            # Gauge
            st.plotly_chart(create_gauge(rul), use_container_width=True)

            # Status cards
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted RUL", f"{rul} cycles")
            c2.metric("Engine Status", status)
            c3.metric("Current Cycle", cycle)

            # Alert
            if rul <= 30:
                st.error("🚨 CRITICAL: Schedule immediate maintenance!")
            elif rul <= 60:
                st.warning("⚠️ WARNING: Plan maintenance soon!")
            elif rul <= 100:
                st.info("ℹ️ MODERATE: Monitor closely")
            else:
                st.success("✅ Engine is healthy")

        except Exception as e:
            st.error(f"API Error: {e}. Make sure FastAPI is running!")

    else:
        st.info("👈 Adjust sensor values in the sidebar and click **Predict RUL**")
        st.plotly_chart(create_gauge(125), use_container_width=True)

with col2:
    st.subheader("📊 Model Information")
    try:
        info = requests.get(f"{API_URL}/model-info").json()
        st.markdown(f"**Model:** {info['model_name']}")
        st.markdown(f"**Version:** {info['version']}")
        st.markdown(f"**Val RMSE:** {info['val_rmse']} cycles")
        st.markdown(f"**Val MAE:** {info['val_mae']} cycles")
        st.markdown(f"**Dataset:** {info['dataset']}")
        st.markdown(f"**Trained on:** {info['trained_on']}")
        st.divider()
        st.subheader("🔴 RUL Zones")
        st.error("0–30 cycles → CRITICAL")
        st.warning("30–60 cycles → WARNING")
        st.info("60–100 cycles → MODERATE")
        st.success("100–125 cycles → HEALTHY")
    except:
        st.warning("Start FastAPI server to see model info")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown("Built with NASA C-MAPSS Dataset | Random Forest v1.0 | FastAPI + Streamlit")