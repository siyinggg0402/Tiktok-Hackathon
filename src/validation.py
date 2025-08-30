import os, json, argparse, re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import LLMClient
import LLM_structuring

# COLUMN NAMES for the ground truth labels for validation
GT_RELEVANCE_COL = "Relevance Score"
GT_QUALITY_COL   = "Quality Score"

# ---- helpers ----
def extract_json(s: str) -> Dict[str, Any]:
    if s is None:
        return {}
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL)
    return json.loads(s)


def make_messages(review_text: str, location: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build the messages for one inference using the global one-shot `prompt`
    defined in LLM_structuring.py, then append the current review as a new user turn.
    """
    base = getattr(LLM_structuring, "prompt", None)
    if not isinstance(base, list):
        raise ValueError("LLM_structuring.prompt must be a list of messages.")

    # shallow-copy the list (and each dict) so we do not mutate the original prompt and cause any
    messages = [m.copy() for m in base]
    
    # append the message prompt of user for the LLM model
    messages.append({
        "role": "user",
        "content": f"""
            Review:
            \"{review_text}\"

            Metadata:
            Name: {location.get('name')}
            Category: {location.get('category')}
            Address: {location.get('address')}
            Opening Hours: {location.get('hours')}
            Timestamp: {location.get('time')}

            Return ONLY the JSON object as specified.
            """.strip()
                })

    return messages

def row_to_location(row: pd.Series) -> Dict[str, Any]:
    return {
        "name": row.get("name"),
        "category": row.get("category"),
        "address": row.get("address"),
        "hours": row.get("hours"),
        "time": row.get("time"),
    }

def normalize_label(x: Any) -> str:
    # just in case the LLM model outputs not as accurate labels
    if x is None:
        return ""
    s = str(x).strip().lower()
    # collapse variants like "avg"/"medium "
    if s in {"med", "avg", "medium"}:
        return "Average"
    return s

# ---- main ----
def main():

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel("../cleaned_data/cleaned_reviews.xlsx")
    needed = {"text", GT_RELEVANCE_COL, GT_QUALITY_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {missing}")

    val_df = df.iloc[: min(10, len(df))].copy().reset_index(drop=True)

    client = LLMClient.LLMClient(model="gpt-4o")

    preds_rel, preds_qual = [], []
    rows_out = []

    for i, row in val_df.iterrows():
        review_text = str(row["text"])
        location = row_to_location(row)
        messages = make_messages(review_text, location)
        raw = client.call_LLM(messages)

        try:
            parsed = extract_json(raw)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": raw}

        pr = normalize_label(parsed.get("Relevance Score"))
        pq = normalize_label(parsed.get("Quality Score"))
        preds_rel.append(pr)
        preds_qual.append(pq)

        rows_out.append({**row.to_dict(), **parsed})

    # Save per-row predictions
    pred_csv = out_dir / "validation_predictions.csv"
    pd.DataFrame(rows_out).to_csv(pred_csv, index=False, encoding="utf-8")
    print(f"[ok] wrote {pred_csv}")

    # Metrics
    y_rel = [normalize_label(x) for x in val_df[GT_RELEVANCE_COL].tolist()]
    y_qual = [normalize_label(x) for x in val_df[GT_QUALITY_COL].tolist()]

    labels = ["low", "average", "high"]  # fixed label order for reports

    def report(y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        print(f"\n=== {name.upper()} ===")
        print(f"Accuracy: {acc:.3f}   Macro-F1: {f1m:.3f}")
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        return {"accuracy": acc, "macro_f1": f1m,
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}

    metrics = {
        "Relevance Score": report(y_rel, preds_rel, "Relevance Score"),
        "Quality Score":   report(y_qual, preds_qual, "Quality Score"),
    }

    with (out_dir / "validation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[ok] wrote {out_dir / 'validation_metrics.json'}")

if __name__ == "__main__":
    main()