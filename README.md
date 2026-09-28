# AI-Powered Fraud Detection for Financial Transactions

## Problem statement

Financial institutions need a quick way to monitor large transaction streams and identify suspicious activity before it becomes a financial loss. Traditional manual review is slow and inconsistent, and a simple rule-based system can highlight risky patterns without requiring a complex machine-learning pipeline.

## Solution

This project creates a lightweight demo dashboard that ingests a stream of synthetic financial transactions, scores each payment for suspicious behavior, and explains the result in plain language. It is designed as a hackathon-ready MVP that can run locally with Streamlit.

## Features

- Risk score from 0 to 100 per transaction
- Explainable anomaly detection logic
- Recommended action: APPROVE, HOLD, or BLOCK
- Behavioral baseline analysis by account
- Timeline view per account
- Sidebar controls for thresholds and filters
- CSV upload support with friendly validation warnings
- Fraud scenario injection demo
- Clear “No anomalies detected” state
- Synthetic dataset built for demo and presentation

## Fraud detection methodology

The system uses explainable, rule-based logic instead of opaque ML models.

- Amount anomaly: compares a transaction against the account's average and typical history.
- Impossible travel: uses a Haversine distance check between consecutive transactions in different cities and a short time window.
- Rapid transactions: flags bursts of payments within a configurable time window.
- New location: highlights cities or countries that are not part of the account’s historical pattern.

## Risk scoring

The scoring uses weighted contributions:

- Unusually large transaction: +30
- Impossible travel: +45
- New/unusual location: +15
- Rapid transaction activity: +20

The total is capped at 100.

Risk bands:

- 0–29: LOW
- 30–59: MEDIUM
- 60–79: HIGH
- 80–100: CRITICAL

## Technology stack

- Python
- Streamlit
- Pandas
- NumPy
- OpenAI Python SDK (optional for AI explanation mode)

## How to install

1. Open a terminal in the project folder.
2. Create and activate a virtual environment if desired.
3. Run:

   pip install -r requirements.txt

## Optional LLM setup

This app supports an optional OpenAI-powered explanation layer that converts the already-calculated fraud evidence into a short, human-readable summary.

The fraud scoring itself remains entirely in the deterministic rule engine. The LLM never changes the risk score or business action.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:OPENAI_MODEL="gpt-5.6-luna"
streamlit run app.py
```

If the API key is not set, or the request fails, the app will automatically fall back to the built-in deterministic explanation without crashing.

### LLM behavior

- AUTO: uses the LLM if an API key exists; otherwise falls back
- LLM: attempts to use the LLM and falls back safely on errors
- Fallback: never calls the API and always uses the deterministic explanation

## How to run

From the project directory:

streamlit run app.py

## Demo scenario

The default demo data contains suspicious patterns including:

- A101: an optional impossible-travel scenario, added with the Inject Fraud Scenario button
- A104: rapid burst of transactions in Bengaluru
- Large-value outliers for low-normal accounts
- Transactions from a new country for an account with a fixed historical region

## Limitations

- This is a demo system, not a production banking control system.
- The scoring model is intentionally rule-based and simple.
- It uses synthetic data only.
- It does not connect to a real banking backend or payment network.

## Future improvements

- Add a more robust anomaly model
- Include user-defined watchlists and manual review queues
- Add CSV export for flagged transactions
- Add better time-series charts and account clustering
- Add optional LLM narration behind a deterministic template layer
