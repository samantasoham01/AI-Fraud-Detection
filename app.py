"""Streamlit dashboard for explainable fraud detection on synthetic transaction data."""

from __future__ import annotations

import os
import hashlib
from io import StringIO

import pandas as pd
import streamlit as st

from fraud_engine import analyze_transactions, generate_llm_explanation, get_risk_level

DEFAULT_PATH = "data.csv"
REQUIRED_COLUMNS = ["transaction_id", "account_id", "timestamp", "amount", "city", "country"]
DEMO_TRANSACTION_ID = "TX_FRAUD_DEMO"
DEMO_TRANSACTION = {
    "transaction_id": DEMO_TRANSACTION_ID,
    "account_id": "A101",
    "timestamp": "2026-09-28 14:03:00",
    "amount": 2500,
    "city": "Tokyo",
    "country": "Japan",
}

st.set_page_config(page_title="AI Fraud Detection Center", page_icon="🛡️", layout="wide")


@st.cache_data
def load_default_dataset() -> pd.DataFrame:
    """Load the bundled demo dataset."""
    return pd.read_csv(DEFAULT_PATH)


def render_badge(label: str) -> str:
    """Return a plain-language badge for a risk label."""
    mapping = {
        "LOW": "🟢 LOW",
        "MEDIUM": "🟡 MEDIUM",
        "HIGH": "🟠 HIGH",
        "CRITICAL": "🔴 CRITICAL",
    }
    return mapping.get(str(label).upper(), str(label).upper())


def risk_label_text(label: str) -> str:
    """Return a user-friendly risk label for table display."""
    return render_badge(label)


def get_ai_mode() -> str:
    """Return the current AI mode setting from session state."""
    return str(st.session_state.get("ai_mode", "AUTO")).upper()


def get_llm_status_text() -> str:
    """Return the status text for the current AI mode."""
    mode = get_ai_mode()
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if mode == "FALLBACK":
        return "FALLBACK MODE"
    if mode == "LLM":
        return "CONNECTED" if has_key else "OFFLINE"
    return "CONNECTED" if has_key else "FALLBACK MODE"


def get_account_baseline(raw_df: pd.DataFrame, account_id: str) -> dict:
    """Compute a compact baseline summary for the selected account."""
    if raw_df is None or raw_df.empty:
        return {"normal_amount_range": "N/A", "normal_location": "Unknown"}

    account_df = raw_df[raw_df["account_id"].astype(str) == str(account_id)].copy().sort_values("timestamp")
    if account_df.empty:
        return {"normal_amount_range": "N/A", "normal_location": "Unknown"}

    if len(account_df) > 1:
        normal_history = account_df.iloc[:-1].copy()
    else:
        normal_history = account_df.copy()

    if normal_history.empty:
        normal_history = account_df.copy()

    normal_location = "Unknown"
    if not normal_history["city"].mode().empty:
        normal_location = str(normal_history["city"].mode().iloc[0])

    if not normal_history.empty:
        low_amount = float(normal_history["amount"].min())
        high_amount = float(normal_history["amount"].max())
        normal_amount_range = f"${low_amount:,.2f} - ${high_amount:,.2f}"
        baseline_amounts = normal_history["amount"].tolist()
    else:
        low_amount = float(account_df["amount"].iloc[0])
        high_amount = float(account_df["amount"].iloc[0])
        normal_amount_range = f"${low_amount:,.2f} - ${high_amount:,.2f}"
        baseline_amounts = [float(account_df["amount"].iloc[0])]

    previous_location = "Unknown"
    if len(account_df) > 1:
        previous_location = str(account_df.iloc[-2]["city"]) if "city" in account_df.columns else "Unknown"

    distance_km = 0.0
    time_minutes = 0.0
    if len(account_df) > 1:
        previous_row = account_df.iloc[-2]
        current_row = account_df.iloc[-1]
        if str(previous_row.get("city", "")).strip() and str(current_row.get("city", "")).strip():
            prev_city = str(previous_row.get("city", "")).strip()
            curr_city = str(current_row.get("city", "")).strip()
            if prev_city != curr_city:
                from fraud_engine import get_city_coordinates, haversine_km

                prev_coords = get_city_coordinates(prev_city)
                curr_coords = get_city_coordinates(curr_city)
                if prev_coords and curr_coords:
                    distance_km = float(haversine_km(prev_coords[0], prev_coords[1], curr_coords[0], curr_coords[1]))
                    time_minutes = float(abs((pd.to_datetime(current_row["timestamp"]) - pd.to_datetime(previous_row["timestamp"])).total_seconds() / 60.0))

    return {
        "normal_amount_range": normal_amount_range,
        "normal_location": normal_location,
        "previous_location": previous_location,
        "distance_km": distance_km,
        "time_minutes": time_minutes,
        "baseline_amounts": baseline_amounts,
    }


def safe_load_csv(uploaded_file) -> pd.DataFrame:
    """Load CSV content from an upload or fall back to the bundled dataset."""
    if uploaded_file is None:
        return load_default_dataset()
    try:
        content = uploaded_file.getvalue()
        if not content:
            st.warning("Uploaded file is empty. Falling back to the demo dataset.")
            return load_default_dataset()
        return pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as exc:
        st.warning(f"Could not read the uploaded CSV: {exc}")
        return load_default_dataset()


def ensure_valid_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataset shape and return a usable DataFrame."""
    if df is None or df.empty:
        st.warning("The current dataset is empty. Showing the demo data instead.")
        return load_default_dataset()

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        st.error(f"Invalid dataset structure. Missing required columns: {', '.join(missing)}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return df


def inject_fraud_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exactly one demo transaction, adding it only when absent."""
    if DEMO_TRANSACTION_ID in df["transaction_id"].astype(str).values:
        return df.copy()
    return pd.concat([df.copy(), pd.DataFrame([DEMO_TRANSACTION])], ignore_index=True)


def render_summary_metrics(df: pd.DataFrame, risk_threshold: int, min_risk_score: int) -> None:
    """Display the KPI cards for total, suspicious, high-risk, and average risk."""
    suspicious = int((df["risk_score"] >= risk_threshold).sum()) if "risk_score" in df.columns else 0
    high_risk = int((df["risk_score"] >= 60).sum()) if "risk_score" in df.columns else 0
    avg_risk = float(df["risk_score"].mean()) if "risk_score" in df.columns and not df.empty else 0.0

    cols = st.columns(4)
    cols[0].metric("Total Transactions", len(df))
    cols[1].metric("Suspicious Transactions", suspicious)
    cols[2].metric("High Risk Transactions", high_risk)
    cols[3].metric("Average Risk Score", f"{avg_risk:.1f}")

    st.caption(f"Displayed risk threshold: {risk_threshold} | Minimum risk filter: {min_risk_score}")


def render_invalid_rows_section(invalid_rows: list) -> None:
    """Display rows that could not be evaluated because of invalid/missing required values."""
    if not invalid_rows:
        return
    invalid_df = pd.DataFrame(invalid_rows)
    if invalid_df.empty:
        return

    st.markdown("### ⚠️ Could not evaluate")
    st.dataframe(invalid_df[["transaction_id", "account_id", "timestamp", "amount", "city", "country"]], use_container_width=True, hide_index=True)


def render_flagged_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Render the flagged transactions table sorted by highest risk."""
    if df.empty:
        st.info("✅ No anomalies detected")
        return pd.DataFrame()

    flagged = df.sort_values("risk_score", ascending=False).reset_index(drop=True).copy()
    display_cols = [
        "transaction_id",
        "account_id",
        "timestamp",
        "amount",
        "city",
        "risk_score",
        "risk_level",
        "reason",
        "recommended_action",
    ]
    flagged = flagged[display_cols].rename(
        columns={
            "transaction_id": "Transaction ID",
            "account_id": "Account",
            "timestamp": "Time",
            "amount": "Amount",
            "city": "Location",
            "risk_score": "Risk Score",
            "risk_level": "Risk Level",
            "reason": "Reason",
            "recommended_action": "Recommended Action",
        }
    )

    styled = flagged.copy()
    styled["Risk Level"] = styled["Risk Level"].map(lambda value: risk_label_text(value))
    styled["Amount"] = styled["Amount"].map(lambda x: f"${x:,.2f}")
    st.markdown("### 🚨 Flagged Transactions")
    st.dataframe(styled, use_container_width=True, hide_index=True)
    return flagged


def render_explanation_panel(df: pd.DataFrame, selected_tx: str | None, raw_df: pd.DataFrame | None = None) -> None:
    """Show detailed explainability, AI explanation, and actions for a selected suspicious transaction."""
    if df.empty or selected_tx is None:
        st.info("Select a flagged transaction to inspect the explainability details.")
        return

    tx = df[df["transaction_id"] == selected_tx]
    if tx.empty:
        return
    row = tx.iloc[0]

    st.markdown("### 🚨 TRANSACTION RISK ANALYSIS")
    st.markdown(f"**Transaction:** {row['transaction_id']}")
    st.markdown(f"**Account:** {row['account_id']}")
    st.markdown(f"**Amount:** ${row['amount']:,.2f}")
    st.markdown(f"**Location:** {row['city']}, {row['country']}")
    st.markdown(f"**Risk:** {int(row['risk_score'])} / 100")
    st.markdown(f"**Level:** {render_badge(row['risk_level'])}")
    st.markdown("**Detected Factors:**")
    for factor in row.get("detected_factors", []):
        st.markdown(f"🔴 {factor}")

    st.markdown("---")
    st.markdown("### 🤖 AI-POWERED EXPLANATION")
    mode = get_ai_mode()
    llm_status = get_llm_status_text()
    if llm_status == "CONNECTED":
        st.caption("🟢 LLM Connected")
    else:
        st.caption("🟡 LLM Offline — Using Explainable Fallback")

    if "ai_explanations" not in st.session_state:
        st.session_state.ai_explanations = {}

    explanation_cache = st.session_state.ai_explanations
    explanation_key = str(row["transaction_id"])
    explanation_text = explanation_cache.get(explanation_key)
    if mode == "FALLBACK":
        explanation_text = row.get("explanation", "No explanation available.")
    elif mode == "LLM" and not explanation_text:
        explanation_text = row.get("explanation", "No explanation available.")

    generate_clicked = st.button("🤖 Generate AI Explanation", key=f"generate_ai_{explanation_key}")
    if generate_clicked:
        if mode == "FALLBACK":
            explanation_text = row.get("explanation", "No explanation available.")
            st.warning("⚠️ LLM unavailable. Using explainable fallback.")
        else:
            account_baseline = get_account_baseline(raw_df, row["account_id"]) if raw_df is not None else {}
            try:
                explanation_text = generate_llm_explanation(
                    row.to_dict(),
                    row.get("detected_factors", []),
                    int(row["risk_score"]),
                    row["risk_level"],
                    account_baseline,
                )
                if explanation_text:
                    explanation_cache[explanation_key] = explanation_text
                    st.success("AI explanation generated.")
            except Exception:
                explanation_text = row.get("explanation", "No explanation available.")
                st.warning("⚠️ LLM unavailable. Using explainable fallback.")

    if not explanation_text:
        explanation_text = row.get("explanation", "No explanation available.")
    if explanation_key not in explanation_cache and mode != "FALLBACK" and mode != "LLM":
        explanation_cache[explanation_key] = explanation_text

    if mode == "AUTO":
        if llm_status == "CONNECTED" and explanation_key not in explanation_cache:
            explanation_text = row.get("explanation", explanation_text)

    st.markdown(f'"{explanation_text}"')

    st.markdown(f"**Recommended Action:** {row['recommended_action']}")
    recommended_action = str(row['recommended_action']).upper()
    if recommended_action == "BLOCK":
        st.markdown("🔴 BLOCK")
    elif recommended_action == "HOLD":
        st.markdown("🟠 HOLD FOR REVIEW")
    else:
        st.markdown("🟢 APPROVE")

    if "decision_state" not in st.session_state:
        st.session_state.decision_state = {}
    decision = st.session_state.decision_state.get(row["transaction_id"], "")

    col1, col2, col3 = st.columns(3)
    if col1.button("[ APPROVE ]"):
        st.session_state.decision_state[row["transaction_id"]] = "APPROVE"
        st.success("Transaction marked as APPROVE.")
    if col2.button("[ HOLD ]"):
        st.session_state.decision_state[row["transaction_id"]] = "HOLD"
        st.warning("Transaction marked as HOLD.")
    if col3.button("[ BLOCK ]"):
        st.session_state.decision_state[row["transaction_id"]] = "BLOCK"
        st.error("Transaction marked as BLOCK.")

    if decision:
        st.caption(f"Current demo status: {decision}")


def render_account_timeline(df: pd.DataFrame, evaluated_df: pd.DataFrame | None = None) -> None:
    """Show transactional history for a selected account with suspicious markers."""
    if df.empty:
        st.info("No account data available yet.")
        return

    timeline_df = df.copy()
    if evaluated_df is not None and not evaluated_df.empty:
        timeline_df = timeline_df.merge(
            evaluated_df[["transaction_id", "risk_score", "risk_level", "detected_factors"]],
            on="transaction_id",
            how="left",
        )

    accounts = sorted(timeline_df["account_id"].astype(str).unique())
    selected_account = st.selectbox("Select account", accounts, key="account_timeline")
    account_df = timeline_df[timeline_df["account_id"].astype(str) == selected_account].copy()
    account_df = account_df.sort_values("timestamp").reset_index(drop=True)

    st.markdown(f"### 📍 Account Behavioral Timeline — {selected_account}")
    for _, row in account_df.iterrows():
        risk_score = float(row.get("risk_score", 0) or 0)
        icon = "🚨" if risk_score >= 30 else ""
        st.markdown(f"- {row['timestamp']} | {row['city']} | ${row['amount']:,.2f} {icon}")

    st.caption("Suspicious transactions are highlighted with the warning icon.")


def render_account_behavior(raw_df: pd.DataFrame, evaluated_df: pd.DataFrame | None = None) -> None:
    """Show normal baseline metrics separately from the current suspicious transaction."""
    if raw_df.empty:
        st.info("No behavioral data available.")
        return

    account = st.selectbox("Account behavior summary", sorted(raw_df["account_id"].astype(str).unique()), key="account_behavior")
    account_df = raw_df[raw_df["account_id"].astype(str) == account].copy().sort_values("timestamp").reset_index(drop=True)
    if account_df.empty:
        return

    suspicious_rows = pd.DataFrame()
    if evaluated_df is not None and not evaluated_df.empty:
        suspicious_rows = evaluated_df[evaluated_df["account_id"].astype(str) == account].copy()
        suspicious_rows = suspicious_rows.sort_values("risk_score", ascending=False).reset_index(drop=True)

    if not suspicious_rows.empty:
        current_tx = suspicious_rows.iloc[0].copy()
        suspicious_ids = set(suspicious_rows["transaction_id"].astype(str).tolist())
        baseline_df = account_df[~account_df["transaction_id"].astype(str).isin(suspicious_ids)].copy()
        if baseline_df.empty:
            baseline_df = account_df.iloc[:-1].copy() if len(account_df) > 1 else account_df.copy()
    else:
        current_tx = account_df.iloc[-1].copy()
        baseline_df = account_df.iloc[:-1].copy() if len(account_df) > 1 else account_df.copy()

    typical_amount = baseline_df["amount"].median() if not baseline_df.empty else current_tx["amount"]
    avg_amount = baseline_df["amount"].mean() if not baseline_df.empty else current_tx["amount"]
    max_normal = baseline_df["amount"].max() if not baseline_df.empty else current_tx["amount"]
    typical_location = baseline_df["city"].mode().iloc[0] if not baseline_df.empty and not baseline_df["city"].mode().empty else "Unknown"
    suspicious_location = current_tx.get("city", "Unknown")
    current_risk_level = get_risk_level(int(current_tx.get("risk_score", 0))) if "risk_score" in current_tx else "LOW"

    st.markdown(f"### 📊 Account Behavior — {account}")
    st.markdown("#### Normal baseline")
    cols = st.columns(4)
    cols[0].metric("Typical Amount", f"${typical_amount:,.2f}")
    cols[1].metric("Average Amount", f"${avg_amount:,.2f}")
    cols[2].metric("Maximum Normal Amount", f"${max_normal:,.2f}")
    cols[3].metric("Typical Location", typical_location)

    st.markdown("#### Current suspicious transaction")
    curr_cols = st.columns(4)
    curr_cols[0].metric("Transaction ID", str(current_tx.get("transaction_id", "N/A")))
    curr_cols[1].metric("Current Amount", f"${float(current_tx.get('amount', 0) or 0):,.2f}")
    curr_cols[2].metric("Suspicious Location", suspicious_location)
    curr_cols[3].metric("Risk", f"{int(current_tx.get('risk_score', 0) or 0)}/100")
    st.markdown(f"**Current Risk Level:** {render_badge(current_risk_level)}")

    st.markdown("#### Detected anomalies")
    factors = current_tx.get("detected_factors", [])
    if isinstance(factors, list) and factors:
        for factor in factors:
            st.markdown(f"- {factor}")
    else:
        st.caption("No anomalies currently detected for this account.")

    st.markdown(f"**Transaction Frequency:** {len(account_df)}")
    st.markdown(f"**Suspicious Transactions Count:** {len(suspicious_rows) if not suspicious_rows.empty else 0}")


def main() -> None:
    """Application entry point."""
    st.title("🛡️ AI Fraud Detection Center")
    st.caption("Real-time transaction risk monitoring and explainable anomaly detection")
    st.markdown(f"**🤖 AI Explanation: {get_llm_status_text()}**")

    if "base_df" not in st.session_state:
        st.session_state.base_df = ensure_valid_dataset(load_default_dataset())
    if "fraud_injected" not in st.session_state:
        st.session_state.fraud_injected = st.query_params.get("fraud_injected", "").lower() == "true"
    if "ai_mode" not in st.session_state:
        st.session_state.ai_mode = "AUTO"
    if "ai_explanations" not in st.session_state:
        st.session_state.ai_explanations = {}

    base_df = st.session_state.base_df.copy()

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        st.selectbox("AI Explanation Mode", ["AUTO", "LLM", "Fallback"], index=0, key="ai_mode")
        large_tx_multiplier = st.slider("Large transaction multiplier", min_value=1.5, max_value=6.0, value=3.0, step=0.1)
        rapid_window_minutes = st.slider("Rapid transaction window (minutes)", min_value=1, max_value=15, value=5)
        high_risk_threshold = st.slider("High risk threshold", min_value=20, max_value=90, value=60)
        critical_threshold = st.slider("Critical threshold", min_value=50, max_value=100, value=80)
        min_risk_score = st.slider("Minimum risk score filter", min_value=0, max_value=100, value=20)

        dataset_preview = base_df
        account_options = ["All"]
        city_options = ["All"]
        if dataset_preview is not None and not dataset_preview.empty:
            account_options.extend(sorted(dataset_preview["account_id"].astype(str).unique().tolist(), key=str))
            city_options.extend(sorted(dataset_preview["city"].astype(str).unique().tolist(), key=str))
        account_filter = st.selectbox("Account filter", account_options)
        city_filter = st.selectbox("City / country filter", city_options)

        st.markdown("---")
        if st.button("🚨 Inject Fraud Scenario"):
            st.session_state.fraud_injected = True
            st.query_params["fraud_injected"] = "true"
            st.rerun()
        if st.button("🔄 Reset Dataset"):
            st.session_state.base_df = ensure_valid_dataset(load_default_dataset())
            st.session_state.fraud_injected = False
            if "fraud_injected" in st.query_params:
                del st.query_params["fraud_injected"]
            st.session_state.ai_explanations = {}
            st.session_state.pop("decision_state", None)
            st.session_state.pop("dataset", None)
            st.rerun()

    if uploaded_file is not None:
        upload_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if upload_signature != st.session_state.get("uploaded_file_signature"):
            st.session_state.base_df = ensure_valid_dataset(safe_load_csv(uploaded_file))
            st.session_state.fraud_injected = False
            if "fraud_injected" in st.query_params:
                del st.query_params["fraud_injected"]
            st.session_state.ai_explanations = {}
            st.session_state.pop("decision_state", None)
            st.session_state.uploaded_file_signature = upload_signature
    else:
        st.session_state.uploaded_file_signature = None
    base_df = st.session_state.base_df.copy()

    working_df = base_df.copy()
    if st.session_state.fraud_injected:
        working_df = inject_fraud_scenario(working_df)

    demo_mask = working_df["transaction_id"].astype(str) == DEMO_TRANSACTION_ID
    demo_count = int(demo_mask.sum())
    if demo_count > 1:
        first_demo_index = working_df.index[demo_mask][0]
        working_df = working_df.loc[~demo_mask | (working_df.index == first_demo_index)].copy()
    working_df = working_df.drop_duplicates(subset=["transaction_id"], keep="first")

    df = working_df
    if df.empty:
        st.warning("No valid data available for analysis.")
        st.stop()

    try:
        analysis = analyze_transactions(
            df,
            large_tx_multiplier=large_tx_multiplier,
            rapid_window_minutes=rapid_window_minutes,
            high_risk_threshold=high_risk_threshold,
            critical_threshold=critical_threshold,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    evaluated = analysis["evaluated_df"].copy()
    invalid_rows = analysis["invalid_rows"]

    if invalid_rows:
        st.warning(f"⚠️ {len(invalid_rows)} transactions could not be evaluated because required fields were missing or invalid.")
    render_invalid_rows_section(invalid_rows)

    filtered_eval = evaluated.copy()
    if not filtered_eval.empty:
        filtered_eval = filtered_eval[filtered_eval["risk_score"] >= min_risk_score].copy()
        if account_filter != "All":
            filtered_eval = filtered_eval[filtered_eval["account_id"].astype(str) == account_filter].copy()
        if city_filter != "All":
            filtered_eval = filtered_eval[filtered_eval["city"].astype(str) == city_filter].copy()

    render_summary_metrics(filtered_eval, high_risk_threshold, min_risk_score)
    st.markdown("---")

    if filtered_eval.empty:
        st.info("✅ No anomalies detected")
    else:
        rendered_flagged = render_flagged_transactions(filtered_eval)
        if not rendered_flagged.empty:
            tx_options = rendered_flagged["Transaction ID"].tolist()
            selected_tx = st.selectbox("Select a transaction", tx_options, index=0)
            render_explanation_panel(filtered_eval, selected_tx, df)

    st.markdown("---")
    render_account_timeline(df, evaluated)
    st.markdown("---")
    render_account_behavior(df, evaluated)

    if not evaluated.empty:
        st.markdown("### Risk Distribution")
        st.bar_chart(evaluated["risk_score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
