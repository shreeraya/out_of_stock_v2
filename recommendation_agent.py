import os
import json
from openai import OpenAI

# Helper to load .env variables
def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val.strip()

load_env()

# Initialize OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def generate_recommendation(analysis, rca_explanation, supplier_name):
    """
    Calls OpenAI API to generate structured recommendation JSON.
    """
    if not api_key:
        return {
            "action_title": "Place Replenishment Order" if analysis["Actionable"] else "Urgent Stock Transfer Required",
            "cost_effort": "Medium",
            "steps": [
                f"Verify current closing stock project of {analysis['ClosingStockAtStockout']} units in {analysis['StockoutWeek']}.",
                f"Contact {supplier_name} to check lead time capabilities."
            ],
            "alternative_action": "Pause marketing/promotional campaigns to shape demand.",
            "email_draft": f"Subject: Urgent OOS Risk SKU {analysis['SKU']}\n\nDear {supplier_name} Team,\n\nWe project an out of stock situation for SKU {analysis['SKU']} in {analysis['StockoutWeek']}."
        }
        
    client = OpenAI(api_key=api_key)
    
    actionable_str = "Actionable (we have enough time to place a standard order)" if analysis["Actionable"] else "NOT Actionable (replenishment lead time exceeds the time remaining)"
    deficit = float(analysis["SafetyStock"]) - float(analysis["ClosingStockAtStockout"])
    
    prompt = f"""You are an expert Supply Chain Recommendation Agent. Your job is to read a quantitative causal analysis and a Root Cause Analysis (RCA) explanation of an Out of Stock (OOS) event, and formulate a structured recommendation plan.

Input Data:
- SKU: {analysis['SKU']}
- Supplier: {supplier_name}
- Stockout Week: {analysis['StockoutWeek']}
- Deficit: {deficit} units
- Lead Time: {analysis['LeadTimeWeeks']} weeks
- Weeks to Stockout: {analysis['WeeksToStockout']} weeks
- Actionability Status: {actionable_str}
- RCA Explanation: {rca_explanation}

Causal Driver Contributions:
- Forecast Demand Spike: {analysis['DemandSpikePct']:.1f}%
- Historical Supplier Delivery Failure: {analysis['HistoricalSupplierFailurePct']:.1f}%
- Future Planned Supply Deficit: {analysis['FutureSupplyDeficitPct']:.1f}%
- Pre-existing Inventory Shortfall: {analysis['InitialInventoryShortfallPct']:.1f}%

Guidelines for your Recommendation:
1. If Actionable is True:
   - Primary recommendation should be placing a standard replenishment order immediately with {supplier_name} for the deficit + safety stock.
   - Cost/Effort should be 'Low' or 'Medium'.
2. If Actionable is False but Future Supply Deficit is high (meaning we have planned shipments in the system but they are insufficient or delayed):
   - Primary recommendation should be to expedite existing POs or request supplier to ship outstanding backlogs.
   - Cost/Effort should be 'Medium'.
   - Draft an email to the supplier {supplier_name} to expedite PO deliveries.
3. If Actionable is False and Stockout is immediate (CW or W1) or starting stock is low due to supplier failure:
   - Primary recommendation should be an Inter-Warehouse Stock Transfer from a nearby regional warehouse to prevent immediate stockout, as standard ordering is too slow.
   - Cost/Effort should be 'Medium' or 'High' (due to shipping and logistical costs).
   - Draft an email/message to the Warehouse Operations Manager requesting inventory transfer.
4. Always suggest a practical alternative action (e.g., Demand Shaping - pausing marketing promotions, SKU substitution, or air-freight shipping).

You MUST return your response as a valid JSON object ONLY. Do not wrap it in markdown code blocks like ```json ... ```. The JSON must have these exact keys:
- "action_title": A concise action title (e.g., "Expedite PO #456 Delivery")
- "cost_effort": "Low", "Medium", or "High"
- "steps": An array of strings detailing the specific steps to execute the recommendation.
- "alternative_action": A short alternative recommendation if the primary recommendation cannot be executed.
- "email_draft": A drafted email to the supplier or operations manager, formatted with Subject and Body.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional supply chain coordinator. You output valid, clean JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        rec_json_str = response.choices[0].message.content.strip()
        return json.loads(rec_json_str)
    except Exception as e:
        # Fallback in case of parsing errors
        return {
            "action_title": "Emergency Review of SKU " + analysis["SKU"],
            "cost_effort": "High",
            "steps": ["Investigate supply chain discrepancy manually.", f"Error details: {e}"],
            "alternative_action": "None",
            "email_draft": "Error generating recommendation."
        }

if __name__ == "__main__":
    # Test script run
    try:
        import pandas as pd
        from sensing_agent import load_data
        from causal_model import analyze_causal_factors
        from rca_agent import generate_rca_explanation
        
        oos, supplier = load_data()
        hist = pd.read_csv("data/inbound_plan_actual.csv")
        
        # Test one case
        sku, week = "SKU001", "W6"
        sku_supplier = supplier[supplier["SKU"] == sku].iloc[0]
        supplier_name = sku_supplier["Supplier Name"]
        
        analysis = analyze_causal_factors(sku, week, oos, supplier, hist)
        rca = generate_rca_explanation(analysis)
        rec = generate_recommendation(analysis, rca, supplier_name)
        
        print(f"\n--- Recommendation Test for {sku} in {week} ---")
        print(json.dumps(rec, indent=2))
        
    except Exception as e:
        print(f"Error testing Recommendation Agent: {e}")
