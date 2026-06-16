import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json

# Set Page Configuration
st.set_page_config(
    page_title="AI Out-of-Stock Agent Control Center",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
from sensing_agent import scan_for_oos
from causal_model import analyze_causal_factors
from rca_agent import generate_rca_explanation
from recommendation_agent import generate_recommendation

# Load .env configuration
def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val.strip()

load_env()

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Header Gradient Style */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Metric Cards Styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #2a5298;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #212529;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Agent Workflow Banner */
    .workflow-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
    }
    
    .step-icon {
        background: #2ebd59;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: bold;
    }
    
    .step-text {
        font-weight: 600;
        font-size: 0.95rem;
        color: #343a40;
    }
    
    .step-desc {
        font-size: 0.8rem;
        color: #6c757d;
    }

    /* Table styles */
    .custom-table th {
        background-color: #2a5298 !important;
        color: white !important;
    }
    
    /* Email Draft Box */
    .email-box {
        background-color: #f1f3f5;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        color: #212529;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & CACHING -----------------
# Cache agent analysis in session state to prevent repetitive LLM API calls on streamlit re-runs
if "agent_cache" not in st.session_state:
    st.session_state["agent_cache"] = {}

def get_oos_analysis(sku, week, oos_df, supplier_df, historical_df, supplier_name):
    cache_key = f"{sku}_{week}"
    if cache_key in st.session_state["agent_cache"]:
        return st.session_state["agent_cache"][cache_key]
    
    # Otherwise, run agents and store in cache
    with st.spinner(f"Agent Pipeline: Analyzing SKU {sku} root causes..."):
        # 1. Causal Model calculations
        causal_res = analyze_causal_factors(sku, week, oos_df, supplier_df, historical_df)
        
        # 2. RCA Agent LLM generation
        rca_desc = generate_rca_explanation(causal_res)
        
        # 3. Recommendation Agent LLM generation
        rec_details = generate_recommendation(causal_res, rca_desc, supplier_name)
        
        analysis_payload = {
            "causal": causal_res,
            "rca": rca_desc,
            "recommendation": rec_details
        }
        st.session_state["agent_cache"][cache_key] = analysis_payload
        return analysis_payload

# ----------------- MAIN UI CODE -----------------

# Header Section
st.markdown("""
<div class="header-container">
    <div class="header-title">📦 AI Out-of-Stock Agent Control Center</div>
    <div class="header-subtitle">Enterprise-level Autonomous Sensing, Causal Diagnosis, and Mitigation Platform</div>
</div>
""", unsafe_allow_html=True)

# File loading logic
st.sidebar.header("📁 Data Inputs")
uploaded_oos = st.sidebar.file_uploader("Upload OOS Table (CSV)", type="csv")
uploaded_supplier = st.sidebar.file_uploader("Upload Supplier Info (CSV)", type="csv")
uploaded_history = st.sidebar.file_uploader("Upload Inbound Plan vs Actual (CSV)", type="csv")

# Set Default paths if not uploaded
oos_path = "data/oos_table.csv"
supplier_path = "data/supplier_info.csv"
history_path = "data/inbound_plan_actual.csv"

# Load DataFrames
try:
    if uploaded_oos:
        oos_df = pd.read_csv(uploaded_oos)
    else:
        oos_df = pd.read_csv(oos_path)
        
    if uploaded_supplier:
        supplier_df = pd.read_csv(uploaded_supplier)
    else:
        supplier_df = pd.read_csv(supplier_path)
        
    if uploaded_history:
        historical_df = pd.read_csv(uploaded_history)
    else:
        historical_df = pd.read_csv(history_path)
        
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}. Please upload valid CSV files.")
    data_loaded = False

if data_loaded:
    # ----------------- 1. SENSING AGENT RUN -----------------
    alerts = scan_for_oos(oos_df, supplier_df)
    
    # Calculate summary metrics
    total_skus = len(oos_df["SKU"].unique())
    total_alerts = len(alerts)
    
    # Filter for first stockout week per SKU to present in core UI list
    sku_first_alert = {}
    for alert in alerts:
        sku = alert["SKU"]
        if sku not in sku_first_alert:
            sku_first_alert[sku] = alert
        else:
            # Keep the earliest week index
            if alert["WeekIndex"] < sku_first_alert[sku]["WeekIndex"]:
                sku_first_alert[sku] = alert
                
    active_alerts_count = len(sku_first_alert)
    actionable_count = sum(1 for a in sku_first_alert.values() if a["WeekIndex"] >= int(supplier_df[supplier_df["SKU"] == a["SKU"]]["Lead Time Weeks"].values[0]))
    non_actionable_count = active_alerts_count - actionable_count

    # Display KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2a5298;">
            <div class="metric-title">Total SKUs Scanned</div>
            <div class="metric-value">{total_skus}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #f39c12;">
            <div class="metric-title">SKUs at Risk of OOS</div>
            <div class="metric-value">{active_alerts_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #2ecc71;">
            <div class="metric-title">Actionable Risks</div>
            <div class="metric-value">{actionable_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #e74c3c;">
            <div class="metric-title">Lead Time Constrained</div>
            <div class="metric-value">{non_actionable_count}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Display Agent Pipeline Status Visualizer
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-step">
            <div class="step-icon">1</div>
            <div>
                <div class="step-text">Sensing Agent Active</div>
                <div class="step-desc">Scanning OOS tables chronologically</div>
            </div>
        </div>
        <div style="font-size: 1.5rem; color: #ced4da;">→</div>
        <div class="workflow-step">
            <div class="step-icon" style="background-color: #9b59b6;">2</div>
            <div>
                <div class="step-text">RCA Causal Agent Active</div>
                <div class="step-desc">Attributing demand, supply & lead time drivers</div>
            </div>
        </div>
        <div style="font-size: 1.5rem; color: #ced4da;">→</div>
        <div class="workflow-step">
            <div class="step-icon" style="background-color: #3498db;">3</div>
            <div>
                <div class="step-text">Recommendation Agent Active</div>
                <div class="step-desc">Generating mitigations & email drafts</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Grid Layout: Left (Alert List) & Right (Drilldown Detail)
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.subheader("⚠️ Sensed Stockout Risks")
        st.write("Click on any SKU row below to trigger the RCA and Recommendation agents.")
        
        # Build Table data
        alert_table_data = []
        for sku, alert in sku_first_alert.items():
            sku_supplier = supplier_df[supplier_df["SKU"] == sku].iloc[0]
            lead_time = int(sku_supplier["Lead Time Weeks"])
            weeks_to_stockout = alert["WeekIndex"]
            
            actionable_badge = "🟢 Actionable" if weeks_to_stockout >= lead_time else "🔴 Constrained"
            
            alert_table_data.append({
                "SKU": sku,
                "First OOS Week": alert["Week"],
                "Weeks to OOS": weeks_to_stockout,
                "Lead Time": f"{lead_time} wks",
                "Status": actionable_badge,
                "Deficit Units": int(alert["Deficit"])
            })
            
        if alert_table_data:
            alert_df = pd.DataFrame(alert_table_data)
            
            # Select row
            selected_sku = st.selectbox(
                "Select SKU to analyze:",
                options=alert_df["SKU"].tolist()
            )
            
            # Display summary table for reference
            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No OOS alerts detected in the provided table!")
            selected_sku = None

    # Right Column: Multi-Agent Analysis Drilldown
    with right_col:
        if selected_sku:
            # Find the alert details
            alert_detail = sku_first_alert[selected_sku]
            sku_supplier_row = supplier_df[supplier_df["SKU"] == selected_sku].iloc[0]
            supplier_name = sku_supplier_row["Supplier Name"]
            safety_stock_threshold = float(sku_supplier_row["Safety Stock Threshold"])
            
            # Trigger Agent execution
            analysis = get_oos_analysis(
                selected_sku, 
                alert_detail["Week"], 
                oos_df, 
                supplier_df, 
                historical_df, 
                supplier_name
            )
            
            causal_res = analysis["causal"]
            rca_explanation = analysis["rca"]
            rec = analysis["recommendation"]
            
            st.markdown(f"## 🔍 Deep-Dive: {selected_sku} in {alert_detail['Week']}")
            st.write(f"**Supplier**: {supplier_name} | **Safety Stock Threshold**: {safety_stock_threshold} units")
            
            # Sub-tabs for: 1. Supply Chain View, 2. Root Cause, 3. Recommendations & Email, 4. Simulation Playground
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Inventory Projections", 
                "🧠 Causal RCA", 
                "⚡ Agent Actions & Email", 
                "🎮 Simulation Playground"
            ])
            
            with tab1:
                st.subheader("Inventory Metrics Horizon (CW to W25)")
                
                # Fetch row metrics for selected SKU
                sku_rows = oos_df[oos_df["SKU"] == selected_sku]
                week_cols = ["CW"] + [c for c in oos_df.columns if c.startswith("W") and c[1:].isdigit()]
                
                # Build Plotly Line Chart
                plot_data = []
                for _, row in sku_rows.iterrows():
                    metric = row["Metric"]
                    if metric in ["Opening Inventory", "Forecast", "Inbound Supply", "Closing Stock"]:
                        vals = [float(row[w]) for w in week_cols]
                        plot_data.append({"Metric": metric, "Values": vals})
                        
                fig = go.Figure()
                
                # Colors map
                color_map = {
                    "Opening Inventory": "#2980b9",
                    "Forecast": "#e74c3c",
                    "Inbound Supply": "#27ae60",
                    "Closing Stock": "#8e44ad"
                }
                
                for line in plot_data:
                    metric = line["Metric"]
                    fig.add_trace(go.Scatter(
                        x=week_cols,
                        y=line["Values"],
                        name=metric,
                        line=dict(color=color_map.get(metric, "#000"), width=2.5 if metric == "Closing Stock" else 1.5,
                                  dash="dash" if metric in ["Forecast", "Inbound Supply"] else "solid")
                    ))
                    
                # Add Safety Stock line
                fig.add_trace(go.Scatter(
                    x=week_cols,
                    y=[safety_stock_threshold] * len(week_cols),
                    name="Safety Stock Target",
                    line=dict(color="#d35400", width=2, dash="dot")
                ))
                
                fig.update_layout(
                    title=f"Inventory Projection curves for {selected_sku}",
                    xaxis_title="Week Horizon",
                    yaxis_title="Units",
                    template="plotly_white",
                    hovermode="x unified",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                st.subheader("Causal Root Cause Analysis (RCA)")
                
                c_col1, c_col2 = st.columns([1, 1])
                
                with c_col1:
                    # Pie/Donut Chart for causal breakdown
                    causal_labels = ["Demand Spike", "Historical Supplier Failure", "Future Supply Deficit", "Initial Inventory Shortfall"]
                    causal_values = [
                        causal_res["DemandSpikePct"],
                        causal_res["HistoricalSupplierFailurePct"],
                        causal_res["FutureSupplyDeficitPct"],
                        causal_res["InitialInventoryShortfallPct"]
                    ]
                    
                    # Filter out 0% drivers
                    filtered_labels = []
                    filtered_values = []
                    for lbl, val in zip(causal_labels, causal_values):
                        if val > 0:
                            filtered_labels.append(lbl)
                            filtered_values.append(val)
                            
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=filtered_labels,
                        values=filtered_values,
                        hole=.4,
                        marker=dict(colors=["#ff6b6b", "#e74c3c", "#f39c12", "#5dade2"])
                    )])
                    fig_donut.update_layout(
                        title="Quantitative Causal Attribution",
                        height=280,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
                with c_col2:
                    st.markdown("### 🧬 AI Causal Diagnosis")
                    st.info(rca_explanation)
                    
                    st.markdown("#### Mathematical Drivers:")
                    st.write(f"- **Demand Surge above Baseline**: {causal_res['DemandSpikeAmount']} units")
                    st.write(f"- **Historical Delivery Shortfall**: {causal_res['HistoricalSupplierDeficit']} units")
                    st.write(f"- **Future Replenishment Deficit**: {causal_res['FutureSupplyDeficit']} units")
                    
                    lead_time_status_text = (
                        f"🟢 **Lead Time is Open**: Standard ordering lead time is {causal_res['LeadTimeWeeks']} weeks, "
                        f"and we have {causal_res['WeeksToStockout']} weeks remaining to react."
                    ) if causal_res["Actionable"] else (
                        f"🔴 **Lead Time is Closed (Constrained)**: Standard ordering lead time is {causal_res['LeadTimeWeeks']} weeks, "
                        f"but we only have {causal_res['WeeksToStockout']} weeks remaining before stockout. Alternate options required."
                    )
                    st.markdown(lead_time_status_text)
                    
            with tab3:
                st.subheader("⚡ Agent Action Recommendation")
                
                # Check for alternative model parsing
                rec_title = rec.get("action_title", "Emergency Review")
                rec_cost = rec.get("cost_effort", "Medium")
                rec_steps = rec.get("steps", [])
                rec_alt = rec.get("alternative_action", "No alternative provided.")
                rec_email = rec.get("email_draft", "")
                
                # Display action details
                st.markdown(f"### **Action: {rec_title}**")
                
                col_badge1, col_badge2 = st.columns([1, 3])
                with col_badge1:
                    effort_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(rec_cost, "blue")
                    st.markdown(f"**Cost/Effort**: :{effort_color}[{rec_cost}]")
                    
                st.markdown("#### **Execution Steps:**")
                for s in rec_steps:
                    st.markdown(f"- {s}")
                    
                st.markdown(f"#### **Alternative Action**: *{rec_alt}*")
                
                # Process email
                email_text = ""
                if isinstance(rec_email, dict):
                    subj = rec_email.get("subject", "Urgent PO Request")
                    body = rec_email.get("body", "")
                    email_text = f"Subject: {subj}\n\n{body}"
                else:
                    email_text = str(rec_email)
                    
                st.markdown("---")
                st.markdown("### 📧 Generated Communication Draft")
                st.write("Copy the email below to send to the supplier or logistics manager:")
                st.markdown(f'<div class="email-box">{email_text}</div>', unsafe_allow_html=True)
                
            with tab4:
                st.subheader("🎮 Live Stock Simulation Playground")
                st.write("Simulate changes in planned inbound supply or forecasts to test if we can resolve the stockout in real-time.")
                
                # We let the user choose a week and add units to it
                simulation_week = st.selectbox(
                    "Select week to adjust planned inbound supply:",
                    options=week_cols
                )
                
                adjust_amount = st.slider(
                    f"Adjust Inbound Supply in {simulation_week} (units):",
                    min_value=-500,
                    max_value=1000,
                    value=0,
                    step=50
                )
                
                # Make local copy of rows for calculations
                sim_rows = sku_rows.copy()
                
                # Get the Inbound Row and update it
                inbound_row_idx = sim_rows[sim_rows["Metric"] == "Inbound Supply"].index[0]
                sim_rows.at[inbound_row_idx, simulation_week] = float(sim_rows.at[inbound_row_idx, simulation_week]) + adjust_amount
                
                # Recalculate closing and opening stock for all weeks
                forecast_row = sim_rows[sim_rows["Metric"] == "Forecast"].iloc[0]
                inbound_row = sim_rows[sim_rows["Metric"] == "Inbound Supply"].iloc[0]
                
                sim_opening = []
                sim_closing = []
                temp_inv = float(sim_rows[sim_rows["Metric"] == "Opening Inventory"].iloc[0]["CW"])
                
                for w in week_cols:
                    sim_opening.append(temp_inv)
                    f_val = float(forecast_row[w])
                    i_val = float(inbound_row[w])
                    cl_val = temp_inv + i_val - f_val
                    sim_closing.append(cl_val)
                    temp_inv = max(0.0, cl_val)
                    
                # Update DataFrame values for visualization
                opening_row_idx = sim_rows[sim_rows["Metric"] == "Opening Inventory"].index[0]
                closing_row_idx = sim_rows[sim_rows["Metric"] == "Closing Stock"].index[0]
                
                for idx, w in enumerate(week_cols):
                    sim_rows.at[opening_row_idx, w] = sim_opening[idx]
                    sim_rows.at[closing_row_idx, w] = sim_closing[idx]
                    
                # Evaluate if stockout is resolved
                resolved = True
                failing_week = None
                failing_closing = 0
                
                for idx, w in enumerate(week_cols):
                    if sim_closing[idx] < safety_stock_threshold:
                        resolved = False
                        failing_week = w
                        failing_closing = sim_closing[idx]
                        break
                        
                if resolved:
                    st.success(f"🎉 **Stockout Resolved!** Adjusting inbound supply by {adjust_amount} units keeps closing stock above the safety threshold of {safety_stock_threshold} units across all weeks.")
                else:
                    st.warning(f"⚠️ **OOS Risk Remains**: Closing stock in **{failing_week}** is projected at **{failing_closing} units** (below the safety threshold of {safety_stock_threshold} units). Additional supply or demand shaping required.")
                
                # Chart comparison
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(
                    x=week_cols,
                    y=[float(sku_rows[sku_rows["Metric"] == "Closing Stock"].iloc[0][w]) for w in week_cols],
                    name="Original Closing Stock",
                    line=dict(color="#bdc3c7", width=2, dash="dash")
                ))
                fig_sim.add_trace(go.Scatter(
                    x=week_cols,
                    y=sim_closing,
                    name="Simulated Closing Stock",
                    line=dict(color="#8e44ad", width=3)
                ))
                fig_sim.add_trace(go.Scatter(
                    x=week_cols,
                    y=[safety_stock_threshold] * len(week_cols),
                    name="Safety Stock Target",
                    line=dict(color="#d35400", width=2, dash="dot")
                ))
                fig_sim.update_layout(
                    title="Simulation Comparison: Original vs. Simulated Closing Stock",
                    xaxis_title="Week Horizon",
                    yaxis_title="Units",
                    template="plotly_white",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_sim, use_container_width=True)

else:
    st.info("Please ensure the CSV files are present in the `data/` folder or upload them using the sidebar.")
