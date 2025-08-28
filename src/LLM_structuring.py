from LLMClient import LLMClient
import pandas as pd
from textwrap import dedent
import json

#load dataset
df = pd.read_csv("reviews.csv")
results =  new_res_file

prompt = [
    {
        "role": "system",
        "content": dedent(
            """
            You are a professional review moderator, specialising in evaluating Google Reviews. Your job is to evaluate whether a user review for a business location is relevant, high-quality, and policy-compliant.

            Policies to enforce:


            For each review, analyse its content along with the location metadata, and return your decision in the following strict JSON format:

            {
            "relevance": "high" | "medium" | "low",
            "quality": "high" | "medium" | "low",
            "policy_flags": ["advertisement", "irrelevant", "false_review"],  // may be empty
            "justification": [string],    
            "relevance_score": 0-100,
            "quality_score": 0-100,
            "visited_likelihood": 0-100
            }

            Think in a step-by-step manner, then output only valid JSON.
            """
        )
    },
    {"role": "assistant", "content": """Here is a clear example...    """},
    {"role": "user", "content": """


        """
    },
    {"role": "assistant", "content": "Thank you for the example. I am ready to...Please provide the ... now."}
]

def generate_review_prompt(review_text: str, location: dict) -> list[dict]:
    prompt = prompt.copy()

    # Add new user input for this review
    prompt.append({
        "role": "user",
        "content": dedent(f"""
            Review:
            "{review_text}"

            Location:
            Name: {location.get('name')}
            Category: {location.get('category')}
            Address: {location.get('address')}
            Menu/Services: {location.get('menu')}
            Opening Hours: {location.get('open_hours')}
            Timestamp: {location.get('timestamp')}

            Please return the moderation result in the specified JSON format.
        """)
    })

    return prompt

######################### GET LLM Labelled Results ##########################

client = LLMClient()

for i, row in df.iterrows():
    review_text = row["review_text"]
    location = {
        "name": row.get("location_name", ""),
        "category": row.get("location_category", ""),
        "address": row.get("location_address", ""),
        "menu": row.get("menu", ""),
        "open_hours": row.get("opening_hours", ""),
        "timestamp": row.get("review_time", "")
    }

    full_prompt = generate_review_prompt(review_text, location)
    response = client.call_LLM(full_prompt)

    try:
        parsed = json.loads(response)
    except Exception as e:
        parsed = {"error": str(e), "raw_response": response}

    results.append(parsed)
