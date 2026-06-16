import pandas as pd
import numpy as np

def get_week_columns(df):
    """
    Returns the list of week columns in chronological order: CW, W1, W2, ...
    """
    cols = list(df.columns)
    week_cols = ["CW"] + [c for c in cols if c.startswith("W") and c[1:].isdigit()]
    return week_cols

def analyze_causal_factors(sku, stockout_week, oos_df, supplier_df, historical_df):
    """
    Computes the causal drivers for a stockout of a SKU in a given week.
    
    Returns a dictionary with:
    - SKU, Week, WeekIndex
    - LeadTimeWeeks, WeeksToStockout, Actionable (Boolean)
    - Raw values of:
      - DemandSpikeAmount
      - StartingStockDeficit (split into HistoricalSupplierDeficit and InitialShortfall)
      - FutureSupplyDeficit
    - Percentage contributions of:
      - DemandSpikePct
      - HistoricalSupplierFailurePct
      - InitialInventoryShortfallPct
      - FutureSupplyDeficitPct
    """
    # 1. Basic properties
    week_cols = get_week_columns(oos_df)
    try:
        stockout_idx = week_cols.index(stockout_week)
    except ValueError:
        raise ValueError(f"Week {stockout_week} not found in week columns.")
    
    period_weeks = week_cols[:stockout_idx + 1]
    
    sku_supplier = supplier_df[supplier_df["SKU"] == sku].iloc[0]
    lead_time = int(sku_supplier["Lead Time Weeks"])
    safety_stock = float(sku_supplier["Safety Stock Threshold"])
    historical_attainment = float(sku_supplier["Historical Attainment Rate"])
    
    # Check if stockout is actionable via normal ordering
    weeks_to_stockout = stockout_idx
    actionable = weeks_to_stockout >= lead_time
    
    # 2. Extract metrics
    sku_data = oos_df[oos_df["SKU"] == sku]
    
    opening_row = sku_data[sku_data["Metric"] == "Opening Inventory"].iloc[0]
    forecast_row = sku_data[sku_data["Metric"] == "Forecast"].iloc[0]
    inbound_row = sku_data[sku_data["Metric"] == "Inbound Supply"].iloc[0]
    closing_row = sku_data[sku_data["Metric"] == "Closing Stock"].iloc[0]
    
    opening_cw = float(opening_row["CW"])
    
    # Calculate sum of forecast and inbound in the period
    forecasts = [float(forecast_row[w]) for w in period_weeks]
    inbounds = [float(inbound_row[w]) for w in period_weeks]
    
    sum_forecast = sum(forecasts)
    sum_inbound = sum(inbounds)
    
    # 3. Calculate Causal Drivers
    # A. Demand Spike
    # Baseline forecast is the forecast in current week (CW)
    baseline_forecast = float(forecast_row["CW"])
    steady_forecast_total = baseline_forecast * len(period_weeks)
    demand_spike_amount = max(0.0, sum_forecast - steady_forecast_total)
    
    # B. Starting Stock Deficit
    starting_stock_deficit = max(0.0, safety_stock - opening_cw)
    
    # Check historical performance to attribute starting stock deficit
    sku_hist = historical_df[historical_df["SKU"] == sku]
    if not sku_hist.empty:
        # Sum deviation: Planned Inbound - Actual Inbound
        hist_planned = sku_hist["Planned Inbound"].sum()
        hist_actual = sku_hist["Actual Inbound"].sum()
        historical_inbound_deficit = max(0.0, hist_planned - hist_actual)
    else:
        historical_inbound_deficit = 0.0
        
    historical_supplier_deficit = min(starting_stock_deficit, historical_inbound_deficit)
    initial_shortfall = starting_stock_deficit - historical_supplier_deficit
    
    # C. Future Supply Deficit
    # Expected replenishment gap based on baseline demand
    future_supply_deficit = max(0.0, steady_forecast_total - sum_inbound)
    
    # Calculate total driver sum to normalize
    driver_sum = demand_spike_amount + historical_supplier_deficit + initial_shortfall + future_supply_deficit
    
    if driver_sum > 0:
        demand_spike_pct = (demand_spike_amount / driver_sum) * 100
        hist_supplier_pct = (historical_supplier_deficit / driver_sum) * 100
        initial_shortfall_pct = (initial_shortfall / driver_sum) * 100
        future_supply_pct = (future_supply_deficit / driver_sum) * 100
    else:
        # If there are no positive deficits, check closing stock vs safety stock
        demand_spike_pct = 0.0
        hist_supplier_pct = 0.0
        initial_shortfall_pct = 0.0
        future_supply_pct = 100.0  # default to supply configuration
    
    # 4. Formulate the Output
    analysis = {
        "SKU": sku,
        "StockoutWeek": stockout_week,
        "WeekIndex": stockout_idx,
        "LeadTimeWeeks": lead_time,
        "WeeksToStockout": weeks_to_stockout,
        "Actionable": actionable,
        "SafetyStock": safety_stock,
        "OpeningStockCW": opening_cw,
        "ClosingStockAtStockout": float(closing_row[stockout_week]),
        # Raw Causal Values
        "DemandSpikeAmount": demand_spike_amount,
        "HistoricalInboundDeficit": historical_inbound_deficit,
        "HistoricalSupplierDeficit": historical_supplier_deficit,
        "InitialShortfall": initial_shortfall,
        "FutureSupplyDeficit": future_supply_deficit,
        "TotalDriverSum": driver_sum,
        # Percentage Drivers
        "DemandSpikePct": demand_spike_pct,
        "HistoricalSupplierFailurePct": hist_supplier_pct,
        "InitialInventoryShortfallPct": initial_shortfall_pct,
        "FutureSupplyDeficitPct": future_supply_pct
    }
    
    return analysis

if __name__ == "__main__":
    # Test script run
    try:
        from sensing_agent import load_data
        oos, supplier = load_data()
        hist = pd.read_csv("data/inbound_plan_actual.csv")
        
        test_cases = [
            ("SKU001", "W6"),
            ("SKU002", "W3"),
            ("SKU003", "CW")
        ]
        
        print("Causal Analysis Test Results:")
        for sku, week in test_cases:
            res = analyze_causal_factors(sku, week, oos, supplier, hist)
            print(f"\n--- {sku} Stockout in {week} ---")
            print(f"Lead Time: {res['LeadTimeWeeks']} wks | Weeks to Stockout: {res['WeeksToStockout']} wks | Actionable: {res['Actionable']}")
            print(f"Drivers: ")
            print(f"  - Demand Spike: {res['DemandSpikePct']:.1f}% ({res['DemandSpikeAmount']} units)")
            print(f"  - Past Supplier Failure: {res['HistoricalSupplierFailurePct']:.1f}% ({res['HistoricalSupplierDeficit']} units)")
            print(f"  - Future Supply Gap: {res['FutureSupplyDeficitPct']:.1f}% ({res['FutureSupplyDeficit']} units)")
            print(f"  - Initial Shortfall: {res['InitialInventoryShortfallPct']:.1f}% ({res['InitialShortfall']} units)")
    except Exception as e:
        print(f"Error running causal model: {e}")
