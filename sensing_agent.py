import pandas as pd
import os

def load_data(oos_path="data/oos_table.csv", supplier_path="data/supplier_info.csv"):
    """
    Loads OOS table and Supplier information.
    """
    if not os.path.exists(oos_path) or not os.path.exists(supplier_path):
        raise FileNotFoundError(f"Required data files not found: {oos_path} or {supplier_path}")
    
    oos_df = pd.read_csv(oos_path)
    supplier_df = pd.read_csv(supplier_path)
    return oos_df, supplier_df

def scan_for_oos(oos_df, supplier_df):
    """
    Scans the OOS table and flags any stockouts (closing stock < safety stock).
    Returns a list of alert dictionaries.
    """
    alerts = []
    
    # Map SKU to safety stock threshold
    safety_stock_map = dict(zip(supplier_df["SKU"], supplier_df["Safety Stock Threshold"]))
    
    # Identify week columns
    # Week columns are 'CW' and 'W1' through 'W25'
    all_columns = list(oos_df.columns)
    week_cols = ["CW"] + [col for col in all_columns if col.startswith("W") and col[1:].isdigit()]
    
    # Filter for Closing Stock rows
    closing_stock_df = oos_df[oos_df["Metric"] == "Closing Stock"]
    
    for _, row in closing_stock_df.iterrows():
        sku = row["SKU"]
        safety_stock = safety_stock_map.get(sku, 0)
        
        # Check each week chronologically
        for idx, week_col in enumerate(week_cols):
            closing_val = float(row[week_col])
            
            if closing_val < safety_stock:
                deficit = safety_stock - closing_val
                alerts.append({
                    "SKU": sku,
                    "Week": week_col,
                    "WeekIndex": idx,
                    "ClosingStock": closing_val,
                    "SafetyStock": safety_stock,
                    "Deficit": deficit,
                    "LeadTimeToStockout": idx  # index is the number of weeks from CW (CW=0, W1=1, etc.)
                })
                
    return alerts

if __name__ == "__main__":
    # Test script run
    try:
        oos, supplier = load_data()
        alerts = scan_for_oos(oos, supplier)
        print(f"Sensed {len(alerts)} Out of Stock alerts:")
        for alert in alerts:
            print(f"- {alert['SKU']} stockout in {alert['Week']} (Closing: {alert['ClosingStock']}, Target: {alert['SafetyStock']}, Deficit: {alert['Deficit']})")
    except Exception as e:
        print(f"Error running sensing agent: {e}")
