"""Explainable rule-based fraud detection functions for demo financial transaction analysis."""

from __future__ import annotations

import json
import math
import os
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

CITY_COORDINATES = {
    "Kolkata": (22.5726, 88.3639),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Tokyo": (35.6762, 139.6503),
    "Singapore": (1.3521, 103.8198),
    "London": (51.5072, -0.1276),
    "Paris": (48.8566, 2.3522),
    "New York": (40.7128, -74.0060),
    "Dubai": (25.2048, 55.2708),
}

REQUIRED_COLUMNS = ["transaction_id", "account_id", "timestamp", "amount", "city", "country"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the approximate distance in kilometers between two coordinates."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def get_city_coordinates(city: str) -> Tuple[float, float] | None:
    """Return coordinates for a known city, otherwise None."""
    if city is None:
        return None
    normalized = str(city).strip().title()
    return CITY_COORDINATES.get(normalized)


def calculate_amount_anomaly(account_history: Iterable[float], current_amount: float, multiplier: float = 3.0) -> Dict[str, Any]:
    """Detect unusually large transactions compared with the account's historical spending."""
    history_list = [float(value) for value in account_history if pd.notna(value)]
    if current_amount is None or not pd.notna(current_amount):
        return {"flag": False, "score": 0, "reason": "", "threshold": 0.0}
    if not history_list:
        return {"flag": False, "score": 0, "reason": "", "threshold": 0.0}

    average_amount = sum(history_list) / len(history_list)
    median_amount = pd.Series(history_list).median()
    typical_amount = max(median_amount, average_amount)
    threshold = max(average_amount * multiplier, median_amount * 2.0, 100.0)

    flagged = current_amount >= threshold and current_amount > (typical_amount * 1.5)
    if flagged:
        contribution = 30
        reason = (
            "Unusually large transaction compared with this account's normal spending pattern. "
            f"Current amount ${current_amount:,.2f} exceeds the typical pattern threshold of ${threshold:,.2f}."
        )
    else:
        contribution = 0
        reason = ""

    return {
        "flag": flagged,
        "score": contribution,
        "reason": reason,
        "threshold": threshold,
        "average_amount": average_amount,
        "typical_amount": typical_amount,
    }


def detect_impossible_travel(previous_tx: Dict[str, Any] | None, current_tx: Dict[str, Any]) -> Dict[str, Any]:
    """Detect impossible travel between two consecutive transactions for the same account."""
    if previous_tx is None:
        return {"flag": False, "score": 0, "reason": "", "distance_km": 0.0, "time_minutes": 0.0}

    prev_city = str(previous_tx.get("city", "")).strip()
    curr_city = str(current_tx.get("city", "")).strip()
    prev_country = str(previous_tx.get("country", "")).strip()
    curr_country = str(current_tx.get("country", "")).strip()
    prev_ts = previous_tx.get("timestamp")
    curr_ts = current_tx.get("timestamp")

    if not prev_city or not curr_city or pd.isna(prev_ts) or pd.isna(curr_ts):
        return {"flag": False, "score": 0, "reason": "", "distance_km": 0.0, "time_minutes": 0.0}

    delta_minutes = abs((curr_ts - prev_ts).total_seconds() / 60.0)
    prev_coords = get_city_coordinates(prev_city)
    curr_coords = get_city_coordinates(curr_city)
    if prev_coords is None or curr_coords is None:
        return {"flag": False, "score": 0, "reason": "", "distance_km": 0.0, "time_minutes": delta_minutes}

    distance_km = haversine_km(prev_coords[0], prev_coords[1], curr_coords[0], curr_coords[1])
    if distance_km > 2500 and delta_minutes <= 180:
        return {
            "flag": True,
            "score": 45,
            "reason": (
                "Impossible travel — this account made transactions in locations thousands of kilometers apart only "
                f"{delta_minutes:,.0f} minutes apart."
            ),
            "distance_km": distance_km,
            "time_minutes": delta_minutes,
        }

    return {"flag": False, "score": 0, "reason": "", "distance_km": distance_km, "time_minutes": delta_minutes}


def detect_rapid_transactions(account_history: List[Dict[str, Any]], current_tx: Dict[str, Any], window_minutes: int = 5) -> Dict[str, Any]:
    """Check if multiple transactions happened within a short time window."""
    if current_tx is None:
        return {"flag": False, "score": 0, "reason": "", "count": 0}

    current_ts = current_tx.get("timestamp")
    if pd.isna(current_ts):
        return {"flag": False, "score": 0, "reason": "", "count": 0}

    recent = []
    for item in account_history:
        ts = item.get("timestamp")
        if pd.isna(ts):
            continue
        if current_ts >= ts and (current_ts - ts) <= timedelta(minutes=window_minutes):
            recent.append(item)

    if len(recent) >= 3:
        return {
            "flag": True,
            "score": 20,
            "reason": (
                "Rapid transaction activity — multiple transactions occurred within a short period. "
                f"The account had {len(recent)} transactions in the last {window_minutes} minutes."
            ),
            "count": len(recent),
        }

    return {"flag": False, "score": 0, "reason": "", "count": len(recent)}


def detect_new_location(account_history: List[Dict[str, Any]], current_tx: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a transaction from a location not previously associated with this account."""
    city = str(current_tx.get("city", "")).strip()
    country = str(current_tx.get("country", "")).strip()
    if not city:
        return {"flag": False, "score": 0, "reason": "", "location": city}

    existing_locations = {str(item.get("city", "")).strip().lower() for item in account_history if str(item.get("city", "")).strip()}
    if existing_locations and city.lower() not in existing_locations:
        return {
            "flag": True,
            "score": 15,
            "reason": (
                "Transaction originated from a location not previously associated with this account. "
                f"This account commonly transacts in {', '.join(sorted(existing_locations)[:3]) or 'known locations'} but this payment came from {city}, {country}."
            ),
            "location": city,
        }

    return {"flag": False, "score": 0, "reason": "", "location": city}


def calculate_risk_score(factors: Iterable[Dict[str, Any]]) -> int:
    """Sum weighted risk contributions with a strict cap at 100."""
    total = 0
    for factor in factors:
        if isinstance(factor, dict):
            total += int(factor.get("score", 0))
    return min(100, total)


def generate_reason(transaction: Dict[str, Any], detected_factors: List[str], distance_km: float = 0.0, time_minutes: float = 0.0) -> str:
    """Return a plain-language reason for the suspicious transaction."""
    if not detected_factors:
        return "No suspicious pattern detected."

    pieces = []
    if "amount" in detected_factors:
        pieces.append("Unusually large transaction compared with this account's normal spending pattern.")
    if "travel" in detected_factors:
        pieces.append(
            f"Impossible travel: this account made a transaction in a distant city only {time_minutes:,.0f} minutes earlier. "
            f"The distance is approximately {distance_km:,.0f} km."
        )
    if "rapid" in detected_factors:
        pieces.append("Rapid transaction activity suggests a burst of transactions in a short time.")
    if "new_location" in detected_factors:
        pieces.append("The transaction originated from a location not previously associated with this account.")

    return " ".join(pieces)


def generate_fallback_explanation(
    transaction: Dict[str, Any],
    detected_factors: List[str],
    risk_score: int,
    risk_level: str,
    account_baseline: Dict[str, Any] | None = None,
) -> str:
    """Return a deterministic explanation that never recalculates risk or business decisions."""
    account_baseline = account_baseline or {}
    if not transaction:
        return "No transaction data to evaluate."

    factors = detected_factors or []
    amount = float(transaction.get("amount", 0) or 0)
    city = transaction.get("city", "Unknown")
    country = transaction.get("country", "Unknown")
    account_id = transaction.get("account_id", "Unknown")
    normal_location = account_baseline.get("normal_location", "Unknown")
    normal_amount_range = account_baseline.get("normal_amount_range", "N/A")
    distance_km = float(account_baseline.get("distance_km", 0.0) or 0.0)
    time_minutes = float(account_baseline.get("time_minutes", 0.0) or 0.0)
    previous_location = account_baseline.get("previous_location", "Unknown")

    if "travel" in factors and "amount" in factors and "new_location" in factors:
        return (
            f"This transaction is highly unusual because account {account_id} normally makes relatively small payments in {normal_location}, "
            f"but this payment is ${amount:,.2f} from {city}, {country}. The account made a {previous_location} transaction only {time_minutes:,.0f} minutes earlier, "
            f"and the locations are approximately {distance_km:,.0f} km apart, making the travel implausible. The existing risk engine therefore rates this transaction as {risk_level} with a score of {risk_score}/100 and recommends holding it for review."
        )

    if "amount" in factors and "new_location" in factors:
        return (
            f"This transaction is suspicious because account {account_id} usually spends around {normal_amount_range} in {normal_location}, but this payment is ${amount:,.2f} from {city}, {country}. "
            f"The location is outside the account's usual pattern and the amount is far above the typical baseline. The risk engine therefore flags this transaction as {risk_level} and recommends review."
        )

    if "travel" in factors:
        return (
            f"This transaction is suspicious because the account moved from {previous_location} to {city}, {country} in only {time_minutes:,.0f} minutes while the distance is roughly {distance_km:,.0f} km. "
            f"That timing is not consistent with the account's usual pattern and has triggered the existing risk engine to rate it as {risk_level}."
        )

    return (
        f"This transaction for account {account_id} is unusual because it is outside the account's typical behavior in {normal_location} and does not match the usual money movement pattern. "
        f"The supplied signals ({', '.join(factors) if factors else 'detected fraud pattern'}) caused the risk engine to classify it as {risk_level} with a score of {risk_score}/100."
    )


def generate_explanation(transaction: Dict[str, Any], detected_factors: List[str], distance_km: float = 0.0, time_minutes: float = 0.0) -> str:
    """Optional LLM-ready wrapper for deterministic explanations."""
    if not transaction:
        return "No transaction data to evaluate."

    factors = detected_factors or []
    amount = float(transaction.get("amount", 0) or 0)
    city = transaction.get("city", "Unknown")
    country = transaction.get("country", "Unknown")
    account_id = transaction.get("account_id", "Unknown")

    if "travel" in factors and "amount" in factors and "new_location" in factors:
        return (
            f"This transaction is highly suspicious because account {account_id} made a transaction in Kolkata only a few minutes earlier and then made a ${amount:,.2f} transaction in {city}, {country}. "
            f"The distance is approximately {distance_km:,.0f} km and the time gap is too short for realistic travel. The amount is far above the account's normal spending behavior."
        )

    reason = generate_reason(transaction, factors, distance_km, time_minutes)
    return f"This transaction for account {account_id} in {city}, {country} is suspicious because {reason.lower()} The amount of ${amount:,.2f} is notable for this account profile."


def generate_llm_explanation(
    transaction,
    detected_factors,
    risk_score,
    risk_level,
    account_baseline,
):
    """Return a concise explanation using the OpenAI SDK when possible, otherwise a deterministic fallback."""
    if not transaction:
        return generate_fallback_explanation({}, detected_factors or [], int(risk_score or 0), str(risk_level or "LOW"), account_baseline)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return generate_fallback_explanation(
            transaction,
            detected_factors or [],
            int(risk_score or 0),
            str(risk_level or "LOW"),
            account_baseline,
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=20.0)
        model = (os.getenv("OPENAI_MODEL") or "gpt-5.6-luna").strip() or "gpt-5.6-luna"

        payload = {
            "transaction_id": transaction.get("transaction_id", "Unknown"),
            "account_id": transaction.get("account_id", "Unknown"),
            "amount": float(transaction.get("amount", 0) or 0),
            "timestamp": str(transaction.get("timestamp", "Unknown")),
            "city": transaction.get("city", "Unknown"),
            "country": transaction.get("country", "Unknown"),
            "risk_score": int(risk_score or 0),
            "risk_level": str(risk_level or "LOW"),
            "detected_factors": detected_factors or [],
            "normal_account_amount": account_baseline.get("normal_amount_range", "N/A") if isinstance(account_baseline, dict) else "N/A",
            "normal_account_location": account_baseline.get("normal_location", "Unknown") if isinstance(account_baseline, dict) else "Unknown",
            "distance_km": float((account_baseline or {}).get("distance_km", 0.0) or 0.0) if isinstance(account_baseline, dict) else 0.0,
            "time_difference_minutes": float((account_baseline or {}).get("time_minutes", 0.0) or 0.0) if isinstance(account_baseline, dict) else 0.0,
            "previous_location": account_baseline.get("previous_location", "Unknown") if isinstance(account_baseline, dict) else "Unknown",
        }

        response = client.responses.create(
            model=model,
            instructions=(
                "You are a financial fraud explanation assistant for a synthetic hackathon demonstration. "
                "Do not independently decide whether a transaction is fraudulent. "
                "Do not invent evidence. Do not change the supplied risk score. "
                "Explain only the detected factors supplied by the fraud engine. "
                "Write a concise explanation understandable by a bank operations reviewer. "
                "Mention what happened, why it is unusual, which supplied signals caused concern, and what the existing risk engine recommends. "
                "Keep the explanation to approximately 2-4 sentences."
            ),
            input=[{"role": "user", "content": str(payload)}],
        )

        text = getattr(response, "output_text", "")
        if not text and hasattr(response, "output"):
            for item in response.output:
                if getattr(item, "type", "") == "message":
                    content = getattr(item, "content", [])
                    for block in content:
                        text_value = getattr(block, "text", "")
                        if text_value:
                            text = text_value
                            break
                    if text:
                        break
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        pass

    return generate_fallback_explanation(
        transaction,
        detected_factors or [],
        int(risk_score or 0),
        str(risk_level or "LOW"),
        account_baseline,
    )


def recommend_action(score: int, factors: List[str]) -> str:
    """Translate the aggregate risk score into a recommended action."""
    if score >= 80:
        return "BLOCK" if "travel" in factors or "amount" in factors else "HOLD"
    if score >= 60:
        return "HOLD"
    if score >= 30:
        return "HOLD" if factors else "APPROVE"
    return "APPROVE"


def get_risk_level(score: int) -> str:
    """Map a numeric score to a human-readable risk band."""
    if score <= 29:
        return "LOW"
    if score <= 59:
        return "MEDIUM"
    if score <= 79:
        return "HIGH"
    return "CRITICAL"


def get_openai_config() -> Dict[str, str]:
    """Return the OpenAI configuration from environment variables without exposing secrets."""
    return {
        "api_key": (os.getenv("OPENAI_API_KEY") or "").strip(),
        "model": (os.getenv("OPENAI_MODEL") or "gpt-5.6-luna").strip() or "gpt-5.6-luna",
    }


def generate_llm_explanation(
    transaction: Dict[str, Any],
    detected_factors: List[str],
    risk_score: int,
    risk_level: str,
    account_baseline: Dict[str, Any] | None = None,
) -> str:
    """Create a concise natural-language explanation using the optional LLM layer."""
    account_baseline = account_baseline or {}
    if not transaction or not detected_factors:
        return generate_explanation(transaction or {}, detected_factors or [], account_baseline.get("distance_km", 0.0), account_baseline.get("time_minutes", 0.0))

    config = get_openai_config()
    if not config["api_key"]:
        return generate_explanation(
            transaction,
            detected_factors,
            distance_km=account_baseline.get("distance_km", 0.0),
            time_minutes=account_baseline.get("time_minutes", 0.0),
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=config["api_key"], timeout=20.0)
        payload = {
            "transaction_id": transaction.get("transaction_id", "Unknown"),
            "account_id": transaction.get("account_id", "Unknown"),
            "amount": float(transaction.get("amount", 0) or 0),
            "timestamp": str(transaction.get("timestamp", "Unknown")),
            "city": transaction.get("city", "Unknown"),
            "country": transaction.get("country", "Unknown"),
            "risk_score": int(risk_score),
            "risk_level": str(risk_level),
            "detected_factors": detected_factors,
            "normal_account_amount": account_baseline.get("normal_amount_range", "N/A"),
            "normal_account_location": account_baseline.get("normal_location", "Unknown"),
            "distance_km": float(account_baseline.get("distance_km", 0.0) or 0.0),
            "time_difference_minutes": float(account_baseline.get("time_minutes", 0.0) or 0.0),
            "previous_location": account_baseline.get("previous_location", "Unknown"),
        }

        response = client.responses.create(
            model=config["model"],
            instructions=(
                "You are a financial fraud explanation assistant for a synthetic hackathon demonstration. "
                "Do not independently decide whether a transaction is fraudulent. Do not invent evidence. "
                "Do not change the supplied risk score. Do not create facts that are not provided. "
                "Explain only the detected factors supplied by the fraud engine. Write a concise explanation understandable by a bank operations reviewer. "
                "Mention: 1. What happened. 2. Why the transaction is unusual. 3. Which supplied signals caused concern. 4. What the existing risk engine recommends. "
                "Keep the explanation to approximately 2-4 sentences."
            ),
            input=[
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            ],
        )

        explanation_text = getattr(response, "output_text", "")
        if not explanation_text and hasattr(response, "output"):
            for item in response.output:
                if getattr(item, "type", "") == "message":
                    content = getattr(item, "content", [])
                    for block in content:
                        text_value = getattr(block, "text", "")
                        if text_value:
                            explanation_text = text_value
                            break
                    if explanation_text:
                        break

        return explanation_text.strip() if explanation_text else generate_explanation(
            transaction,
            detected_factors,
            distance_km=account_baseline.get("distance_km", 0.0),
            time_minutes=account_baseline.get("time_minutes", 0.0),
        )
    except Exception:
        return generate_explanation(
            transaction,
            detected_factors,
            distance_km=account_baseline.get("distance_km", 0.0),
            time_minutes=account_baseline.get("time_minutes", 0.0),
        )


def analyze_transactions(
    dataframe: pd.DataFrame,
    large_tx_multiplier: float = 3.0,
    rapid_window_minutes: int = 5,
    high_risk_threshold: int = 60,
    critical_threshold: int = 80,
) -> Dict[str, Any]:
    """Evaluate a transaction stream and return evaluated rows plus invalid transactions."""
    result = {"evaluated_df": pd.DataFrame(), "invalid_rows": [], "message": ""}

    if dataframe is None or dataframe.empty:
        result["message"] = "No data available to evaluate."
        return result

    required = set(REQUIRED_COLUMNS)
    missing_columns = sorted(required.difference(set(dataframe.columns)))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["account_id"] = df["account_id"].fillna("").astype(str).str.strip()
    df["city"] = df["city"].fillna("").astype(str).str.strip()
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    df["transaction_id"] = df["transaction_id"].fillna("").astype(str).str.strip()

    invalid_rows = []
    valid_rows = []
    for row in df.to_dict(orient="records"):
        amount = row.get("amount")
        ts = row.get("timestamp")
        account_id = row.get("account_id")
        city = row.get("city")
        country = row.get("country")
        if pd.isna(amount) or pd.isna(ts) or not account_id or not city or not country:
            invalid_rows.append(row)
            continue
        valid_rows.append(row)

    valid_rows = sorted(valid_rows, key=lambda row: row["timestamp"])
    account_history: Dict[str, List[Dict[str, Any]]] = {}
    evaluated = []

    for row in valid_rows:
        account_id = str(row["account_id"])
        history = account_history.get(account_id, [])

        amount_anomaly = calculate_amount_anomaly([item["amount"] for item in history], float(row["amount"]), large_tx_multiplier)
        new_location = detect_new_location(history, row)
        travel = detect_impossible_travel(history[-1] if history else None, row)
        rapid = detect_rapid_transactions(history, row, rapid_window_minutes)

        factors = []
        risk_points = 0
        if amount_anomaly["flag"]:
            factors.append("amount")
            risk_points += amount_anomaly["score"]
        if travel["flag"]:
            factors.append("travel")
            risk_points += travel["score"]
        if new_location["flag"]:
            factors.append("new_location")
            risk_points += new_location["score"]
        if rapid["flag"]:
            factors.append("rapid")
            risk_points += rapid["score"]

        risk_score = min(100, risk_points)
        reason = generate_reason(row, factors, distance_km=travel.get("distance_km", 0.0), time_minutes=travel.get("time_minutes", 0.0))
        recommendation = recommend_action(risk_score, factors)

        record = {
            "transaction_id": row["transaction_id"],
            "account_id": account_id,
            "timestamp": row["timestamp"],
            "amount": float(row["amount"]),
            "city": row["city"],
            "country": row["country"],
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score),
            "reason": reason,
            "recommended_action": recommendation,
            "detected_factors": factors,
            "is_suspicious": risk_score >= high_risk_threshold,
            "distance_km": travel.get("distance_km", 0.0),
            "time_minutes": travel.get("time_minutes", 0.0),
            "explanation": generate_explanation(row, factors, distance_km=travel.get("distance_km", 0.0), time_minutes=travel.get("time_minutes", 0.0)),
        }
        evaluated.append(record)

        history.append({
            "transaction_id": row["transaction_id"],
            "account_id": account_id,
            "amount": float(row["amount"]),
            "city": row["city"],
            "country": row["country"],
            "timestamp": row["timestamp"],
        })
        account_history[account_id] = history

    evaluated_df = pd.DataFrame(evaluated)
    if not evaluated_df.empty:
        evaluated_df = evaluated_df.sort_values(["risk_score", "amount"], ascending=[False, False]).reset_index(drop=True)

    result["evaluated_df"] = evaluated_df
    result["invalid_rows"] = invalid_rows
    result["message"] = (
        f"⚠️ {len(invalid_rows)} transactions could not be evaluated because required fields were missing or invalid."
        if invalid_rows
        else ""
    )
    return result
