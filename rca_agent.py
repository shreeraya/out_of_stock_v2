import os
from openai import OpenAI

# Helper to load .env variables without external dependency if needed, though dotenv is good
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

if not api_key:
    # Fallback/warning if key is missing, so dashboard won't crash immediately but alert user
    print("Warning: OPENAI_API_KEY not found in environment.")

def generate_rca_explanation(analysis):
    """
    Calls OpenAI API to generate a professional business explanation
    based on the quantitative causal analysis dictionary.
    """
    if not api_key:
        return (
            f"Unable to generate AI explanation: OPENAI_API_KEY is missing. "
            f"Primary mathematical drivers: Demand Spike ({analysis['DemandSpikePct']:.1f}%), "
            f"Supplier Failure ({analysis['HistoricalSupplierFailurePct']:.1f}%), "
            f"Future Supply Gap ({analysis['FutureSupplyDeficitPct']:.1f}%)."
        )
        
    client = OpenAI(api_key=api_key)
    
    actionable_str = "Actionable (we have enough time to place a standard order)" if analysis["Actionable"] else "NOT Actionable (replenishment lead time exceeds the time remaining)"
    
    prompt = f"""You are an expert Supply Chain Root Cause Analysis (RCA) Agent. Your task is to interpret a quantitative causal analysis of an Out of Stock (OOS) event and write a concise, professional executive explanation (2-3 sentences) for a business operations team.

Input Data:
- SKU: {analysis['SKU']}
- Projected Stockout Week: {analysis['StockoutWeek']}
- Safety Stock Target: {analysis['SafetyStock']} units
- Projected Closing Stock: {analysis['ClosingStockAtStockout']} units
- Lead Time to Replenish: {analysis['LeadTimeWeeks']} weeks
- Weeks Remaining to React: {analysis['WeeksToStockout']} weeks
- Actionability Status: {actionable_str}

Causal Driver Contributions:
- Forecast Demand Spike: {analysis['DemandSpikePct']:.1f}% ({analysis['DemandSpikeAmount']} units above baseline)
- Historical Supplier Delivery Failure: {analysis['HistoricalSupplierFailurePct']:.1f}% ({analysis['HistoricalSupplierDeficit']} units shortfall)
- Future Planned Supply Deficit: {analysis['FutureSupplyDeficitPct']:.1f}% ({analysis['FutureSupplyDeficit']} units gap)
- Pre-existing Inventory Shortfall: {analysis['InitialInventoryShortfallPct']:.1f}% ({analysis['InitialShortfall']} units short)

Guidelines:
1. Be direct and clear. State the primary driver(s) and how they caused the projected stockout.
2. Mention the lead time constraint explicitly, stating if the window to react is open or closed.
3. Use professional supply chain terms.
4. Keep the explanation to exactly 2-3 sentences. Do not add markdown formatting to the text itself.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional supply chain analyst agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return f"Error generating AI explanation: {e}"

if __name__ == "__main__":
    # Test script run
    try:
        import pandas as pd
        from sensing_agent import load_data
        from causal_model import analyze_causal_factors
        
        oos, supplier = load_data()
        hist = pd.read_csv("data/inbound_plan_actual.csv")
        
        print("Testing RCA Agent LLM generation:")
        for sku, week in [("SKU001", "W6"), ("SKU002", "W3"), ("SKU003", "CW")]:
            analysis = analyze_causal_factors(sku, week, oos, supplier, hist)
            explanation = generate_rca_explanation(analysis)
            print(f"\n--- {sku} in {week} ---")
            print(explanation)
            
    except Exception as e:
        print(f"Error testing RCA Agent: {e}")
