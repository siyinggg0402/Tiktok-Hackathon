import os, json, argparse, re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import LLMClient
import LLM_structuring

# ---------- helpers ----------
def extract_json(s: str) -> Dict[str, Any]:
    """Tolerant JSON extractor: strips code fences and parses JSON."""
    if s is None:
        return {}
    s = s.strip()
    # remove ```json ... ``` fences if present
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL)
    return json.loads(s)
    # to clean into a cleaner json formatting


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

            Location:
            Name: {location.get('name')}
            Category: {location.get('category')}
            Address: {location.get('address')}
            Opening Hours: {location.get('hours')}
            Timestamp: {location.get('time')}

            Return ONLY the JSON object as specified.
            """.strip()
                })

    return messages

def pick_training_rows(df: pd.DataFrame, start_index: int, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= start_index:
        raise ValueError(f"Data has only {len(df)} rows; cannot start at {start_index}.")
    return df.iloc[start_index:].sample(n=min(n, len(df) - start_index), random_state=seed)

def row_to_location(row: pd.Series) -> Dict[str, Any]:
    # pull if present, else None
    return {
        "name": row.get("name"),
        "category": row.get("category"),
        "address": row.get("address"),
        "hours": row.get("hours"),
        "time": row.get("time"),
    }

# ---------- main ----------
def main():

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel("../cleaned_data/cleaned_reviews.csv")
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column with the review text.")

    sample_df = pick_training_rows(df, start_index=100, n=2)
    sample_df = sample_df.reset_index(drop=True)

    # Init LLM
    client = LLMClient.LLMClient(model= "gpt-4o")

    results: List[Dict[str, Any]] = []
    for i, row in sample_df.iterrows():
        review_text = str(row["text"])
        location = row_to_location(row)
        messages = make_messages(review_text, location)
        raw = client.call_LLM(messages)

        try:
            parsed = extract_json(raw)
        except Exception as e:
            parsed = {"_parse_error": str(e), "_raw": raw}

        result_row = {**row.to_dict(), **parsed}
        results.append(result_row)

    out_csv = out_dir / "training_llm_labels.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[ok] wrote {out_csv} with {len(results)} rows")

    # also keep raw JSON for inspection
    out_jsonl = out_dir / "training_llm_labels.json"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {out_jsonl}")

if __name__ == "__main__":
    main()

