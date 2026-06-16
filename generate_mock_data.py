import os
import pandas as pd
import numpy as np

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Define columns
weeks = [f"W{i}" for i in range(1, 26)]
columns = ["SKU", "Metric", "Current Inventory", "CW"] + weeks

# Create data list
data = []

# --- SKU001: Demand Spike (Forecast Jumps in W4, stockout in W6) ---
# Opening Stock CW = 500. Lead time = 2 weeks.
# Weekly forecast starts at 100, jumps to 300 at W4.
sku_1 = "SKU001"
forecast_1 = [100] * 26
for i in range(4, 26):
    forecast_1[i] = 300
inbound_1 = [100] * 26  # steady supply of 100

# Calculate opening and closing stock
curr_inv_1 = 500
opening_1 = []
closing_1 = []
temp_inv = curr_inv_1

for w in range(26):
    opening_1.append(temp_inv)
    closing = temp_inv + inbound_1[w] - forecast_1[w]
    closing_1.append(closing)
    temp_inv = max(0, closing)  # roll over to next week (opening must be >= 0)

data.append([sku_1, "Opening Inventory", curr_inv_1, opening_1[0]] + opening_1[1:])
data.append([sku_1, "Forecast", 0, forecast_1[0]] + forecast_1[1:])
data.append([sku_1, "Inbound Supply", 0, inbound_1[0]] + inbound_1[1:])
data.append([sku_1, "Closing Stock", 0, closing_1[0]] + closing_1[1:])


# --- SKU002: Supplier Shortage / Delay (Inbound Supply drops to 0 in W2, stockout in W3) ---
# Opening Stock CW = 300. Lead time = 4 weeks.
# Forecast is steady at 120. Inbound planned is 120, but goes to 0 at W2 and W3.
sku_2 = "SKU002"
forecast_2 = [120] * 26
inbound_2 = [120] * 26
inbound_2[2] = 0  # Inbound planned goes to 0 in W2 (index 2 corresponds to W2)
inbound_2[3] = 0  # and W3 (index 3 corresponds to W3)

curr_inv_2 = 200
opening_2 = []
closing_2 = []
temp_inv = curr_inv_2

for w in range(26):
    opening_2.append(temp_inv)
    closing = temp_inv + inbound_2[w] - forecast_2[w]
    closing_2.append(closing)
    temp_inv = max(0, closing)

data.append([sku_2, "Opening Inventory", curr_inv_2, opening_2[0]] + opening_2[1:])
data.append([sku_2, "Forecast", 0, forecast_2[0]] + forecast_2[1:])
data.append([sku_2, "Inbound Supply", 0, inbound_2[0]] + inbound_2[1:])
data.append([sku_2, "Closing Stock", 0, closing_2[0]] + closing_2[1:])


# --- SKU003: Low Starting Inventory / Supplier Performance (Stockout in CW) ---
# Opening Stock CW = 10 (very low). Lead time = 3 weeks.
# Forecast = 100, Inbound planned = 50. Immediate stockout in CW.
sku_3 = "SKU003"
forecast_3 = [100] * 26
inbound_3 = [50] * 26

curr_inv_3 = 10
opening_3 = []
closing_3 = []
temp_inv = curr_inv_3

for w in range(26):
    opening_3.append(temp_inv)
    closing = temp_inv + inbound_3[w] - forecast_3[w]
    closing_3.append(closing)
    temp_inv = max(0, closing)

data.append([sku_3, "Opening Inventory", curr_inv_3, opening_3[0]] + opening_3[1:])
data.append([sku_3, "Forecast", 0, forecast_3[0]] + forecast_3[1:])
data.append([sku_3, "Inbound Supply", 0, inbound_3[0]] + inbound_3[1:])
data.append([sku_3, "Closing Stock", 0, closing_3[0]] + closing_3[1:])


# --- SKU004: Healthy SKU (No Stockout) ---
# Opening Stock CW = 1000. Lead time = 2 weeks.
# Forecast = 100. Inbound = 100.
sku_4 = "SKU004"
forecast_4 = [100] * 26
inbound_4 = [100] * 26

curr_inv_4 = 1000
opening_4 = []
closing_4 = []
temp_inv = curr_inv_4

for w in range(26):
    opening_4.append(temp_inv)
    closing = temp_inv + inbound_4[w] - forecast_4[w]
    closing_4.append(closing)
    temp_inv = max(0, closing)

data.append([sku_4, "Opening Inventory", curr_inv_4, opening_4[0]] + opening_4[1:])
data.append([sku_4, "Forecast", 0, forecast_4[0]] + forecast_4[1:])
data.append([sku_4, "Inbound Supply", 0, inbound_4[0]] + inbound_4[1:])
data.append([sku_4, "Closing Stock", 0, closing_4[0]] + closing_4[1:])

# Convert to DataFrame
df_oos = pd.DataFrame(data, columns=columns)
df_oos.to_csv("data/oos_table.csv", index=False)
print("Saved data/oos_table.csv")

# --- Supplier Info CSV ---
supplier_data = [
    {"SKU": "SKU001", "Supplier Name": "Acuity Electronics", "Lead Time Weeks": 2, "Historical Attainment Rate": 0.95, "Safety Stock Threshold": 50},
    {"SKU": "SKU002", "Supplier Name": "Global Logistics Corp", "Lead Time Weeks": 4, "Historical Attainment Rate": 0.60, "Safety Stock Threshold": 40},
    {"SKU": "SKU003", "Supplier Name": "Nexus Manufacturing", "Lead Time Weeks": 3, "Historical Attainment Rate": 0.45, "Safety Stock Threshold": 80},
    {"SKU": "SKU004", "Supplier Name": "Standard Parts Ltd", "Lead Time Weeks": 2, "Historical Attainment Rate": 0.98, "Safety Stock Threshold": 100}
]
df_supplier = pd.DataFrame(supplier_data)
df_supplier.to_csv("data/supplier_info.csv", index=False)
print("Saved data/supplier_info.csv")

# --- Inbound Plan vs Actual (Historical data for causal model) ---
historical_data = []

# SKU001: high attainment historically
for w in ["H1", "H2", "H3", "H4"]:
    historical_data.append({"SKU": "SKU001", "Week": w, "Planned Inbound": 100, "Actual Inbound": 100})

# SKU002: medium-low attainment
historical_data.append({"SKU": "SKU002", "Week": "H1", "Planned Inbound": 120, "Actual Inbound": 100})
historical_data.append({"SKU": "SKU002", "Week": "H2", "Planned Inbound": 120, "Actual Inbound": 80})
historical_data.append({"SKU": "SKU002", "Week": "H3", "Planned Inbound": 120, "Actual Inbound": 60})
historical_data.append({"SKU": "SKU002", "Week": "H4", "Planned Inbound": 120, "Actual Inbound": 0})  # major failure

# SKU003: very low attainment (explaining why starting stock is 10 instead of much higher)
historical_data.append({"SKU": "SKU003", "Week": "H1", "Planned Inbound": 100, "Actual Inbound": 50})
historical_data.append({"SKU": "SKU003", "Week": "H2", "Planned Inbound": 100, "Actual Inbound": 40})
historical_data.append({"SKU": "SKU003", "Week": "H3", "Planned Inbound": 100, "Actual Inbound": 20})
historical_data.append({"SKU": "SKU003", "Week": "H4", "Planned Inbound": 100, "Actual Inbound": 10})

# SKU004: perfect attainment
for w in ["H1", "H2", "H3", "H4"]:
    historical_data.append({"SKU": "SKU004", "Week": w, "Planned Inbound": 100, "Actual Inbound": 100})

df_historical = pd.DataFrame(historical_data)
df_historical["Attainment Deviation"] = df_historical["Actual Inbound"] - df_historical["Planned Inbound"]
df_historical.to_csv("data/inbound_plan_actual.csv", index=False)
print("Saved data/inbound_plan_actual.csv")
print("Mock datasets generated successfully.")
