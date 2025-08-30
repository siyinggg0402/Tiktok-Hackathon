from LLMClient import LLMClient
import pandas as pd
from textwrap import dedent
import json
from copy import deepcopy

### PROCESS:
#Loads input from Excel.

#Defines a full system prompt with policy/score rules and an example.

#Appends a review to the prompt with metadata per row.

#Calls LLM and parses JSON output.

#Stores each result with row index.

#Merges it back into the original dataframe.

#Saves final output with result columns added.

#load dataset
df = pd.read_excel("../cleaned_data/cleaned_reviews.xlsx") 
results = []

prompt = [
    {
        "role": "system",
        "content":
            """
            You are a professional review moderator, specialising in evaluating Google Reviews. Your job is to evaluate whether a user review for a business location is relevant, high-quality, and policy-compliant. The end goal is to successfully extract 6 entities: `Advertisement`, `Irrelevant Review`, `False Review`, `Vulgar Language`, `Relevance Score`, `Quality Score`.
            
            Follow the instructions step-by-step below to extract the 6 entities which includes 4 policy flagging entities, 1 relevance scoring, and 1 quality scoring:

            1. Follow the list of **IMPORTANT POLICIES TO FLAG** below to extract and flag out `Advertisement`, `Irrelevant Review`, `False Review`, `Vulgar Language`. If any of the 4 above policies are violated, you are to flag them out acccording to the policy that was violated, stating them clearly.
                **IMPORTANT POLICIES TO FLAG**
                    1. **Advertisement**: 
                        - Reviews containing evidence of promotional language, calls to action, or any URLs/links, would be flagged out as advertising. (e.g. "Interested in your next investment success? Click on "wwww.investsuccess.com" to find out more!", "Join my telegram channel to learn more about fast cash: t.me/FASTCASHTODAYBYEDWARD")
                        - Reviews containing evidence of promotions related to the business itself **should not be flagged out as an advertisement** (e.g. "The restaurant is having a 20% discount right now for all lunch mains! Just quote the promotion "SG60" to get it!").
                    2. **Irrelevant Review**: 
                        - Reviews containing evidences of the discussion of topics completely unrelated to the {name} and {category}, should be flagged out as irrelevant. (e.g. if the name of the business is 'Dunkin Donuts' but the review is about 'Krispy Creme', the review is considered as an Irrelevant Review.)
                        - Reviews containing information about personal experiences, or general commentary that has no connection to the business, should also be flagged out as irrelevant. (e.g. Review containing "I want to get a new phone!" at a restaurant business is considered an Irrelevant Review.)
                    3. **False Review**: 
                        - Reviews that come from users who have not visited the location, or that contain fabricated information, are considered violations. (e.g., "I've never been here but I heard this sucks" is considered a False Review.)
                        
                    4. **Vulgar Language**: 
                        - Reviews that contain vulgar, offensive, or inappropriate language are considered violations. (e.g. "F**k this place", "Sh***y food" should be flagged out as vulgar language)
            
                    **Format for extracting policies**:    
                        - Extract "Yes" if policy is violated.
                        - Extract "No" if policy is not violated.
            
            2. If the reviews violate any of the above policies, extract "-" for both `Relevance Score` and `Quality Score`. If the reviews do not violate any of the above policies, you will move on to the next steps to evaluate the review to assign a suitable **Relevance Score** and a **Quality Score**. 
            
            3. Determine the `Relevance Score` based on the **Relevance Criteria** below:
                - **Relevance Criteria**
                    1. Place identification: Mention of place name, branch, or landmark (e.g., "Kopi Heaven at Tiong Bahru")
                    2. Service/product mention: Mention of specific services or products provided by the business.
                    3. Visit Evidence: Descriptions of actions like ordering, paying, wait time, timestamps, or mentioning a visit window.
                    4. Location metadata match: Mention of info that matches metadata (e.g., opening hours, staff, location category)
                - **Relevance Score (Low, Average, High)**:
                    - Low:
                        - 0-1 points mentioned out of the 4 points in the **Relevance Criteria**.
                        - There is completely no mention of the business and the services or products it provides.
                        - The review is usually extremely vague with no elaboration about the context.
                        - Reviews that are extremely short but non-malicious (e.g. “Great”, “Nice place”, “Yummy”) may be given a “Low” quality score if they imply some visit evidence and have no spam/policy violations.
                    - Average: 
                        - 2 points mentioned out of the 4 points in the **Relevance Criteria**.
                        - There is some mention of the business and the services or products it provides, as well as mention of staff name or opening hours, but not in great detail. 
                    - High: 
                        - 3-4 points mentioned out of the 4 points in the **Relevance Criteria**.
                        - Detailed mention of the business and the location, its services/products, staff, and specific experiences.

            4. Determine the `Quality Score` based on the **Quality Criteria** Below
                - **Quality Criteria**
                    1. Detailed Experience: Review mentions specifics such as time, price, item names, service, waiting time.
                    2. Structured Writing: Review has sentences, not just phrases; avoids all-caps/shouting
                    3. Tone & Language: Review uses a respectful, legible, non-hostile tone.
                    4. Non-Spam: no links, ads, handles, referral codes, etc.
                - **Quality Score (Low, Average, High)**:
                    - Low: 
                        - 1 point mentioned out of the 4 points in the **Quality Criteria**.
                        - Fragmented, sloppy, or minimal reviews.
                        - Reviews containing improper languages. 
                        - Reviews which only contain 1-2 words are also considered low quality.
                    - Average: 
                        - 2 points mentioned out of the 4 points in the **Quality Criteria**.
                        - Reviews with some detail about the experience, but lacking depth or specifics.
                    - High (3-4 points): 
                        - 3-4 points mentioned out of the 4 points in the **Quality Criteria**.
                        - Detailed reviews that provide specific information about the experience, including aspects like quality, service, ambiance, and value for money. Reviews also contain proper language use and are structured well.
            
            5. Lastly come up with a `Extraction Justification` that clearly explains how each entity was extracted while ensuring this remains traceable with the review evidence.
            
            """
        
    },
    {"role": "assistant", "content": "Here is a clear example demonstrating the accurate extraction of the 6 entities related to review evaluation."},
    {"role": "user", "content": """
    
            Review:
            "The Plumbing Bros provided excellent service. Gabriel was professional and did not make me wait for long, fixing my toilet in a short span of time. The pricing was very reasonable and cheaper than most found in the market. They also provide high quality plumbing products, such as their pipe wrenches, for us to choose from. Highly recommend this service and their products for anyone in need of plumbing work."

            Location: -
            Name: "Plumbing Bros"
            Category: [Plumbing Services, Home Services, Plumbing Products]
            Address: 123 Main St, Springfield, USA
            Opening Hours: [['Thursday', '8AM-5PM'], ['Friday', '8AM-5PM'], ['Saturday', 'Closed'], ['Sunday', 'Closed'], ['Monday', '8AM-5PM'], ['Tuesday', '8AM-5PM'], ['Wednesday', '8AM-5PM']]
            Timestamp: 2021-06-10 09:50:58 EDT

            Please return the moderation result in the specified JSON format. Below is an example of the expected output:

            Output:
            {
                "Advertisement": "No",
                "Irrelevant Review": "No",
                "False Review": "No",
                "Vulgar Language": "No",
                "Relevance Score": "High",
                "Quality Score": "High",
                "Extraction Justification": "Mentions staff name, service, pricing, and positive experience with visit evidence."
            }
            """
    },
    {"role": "assistant", "content": "Thank you for the example. I am ready to perform review evaluation. Please provide the reviews now."},
]

def generate_review_prompt(review_text: str, location: dict) -> list[dict]:
    prompt_copy = deepcopy(prompt)

    # Add new user input for this review
    prompt_copy.append({
        "role": "user",
        "content": dedent(f"""
            Review:
            "{review_text}"

            Metadata:
            Name: {location.get('name')}
            Category: {location.get('category')}
            Address: {location.get('address')}
            Opening Hours: {location.get('open_hours')}
            Timestamp: {location.get('time')}

            Output the result in strict JSON format with the following keys:
            {{
              "Advertisement": "Yes" | "No",
              "Irrelevant Review": "Yes" | "No",
              "False Review": "Yes" | "No",
              "Vulgar Language": "Yes" | "No",
              "Relevance Score": "High" | "Average" | "Low" | "-",
              "Quality Score": "High" | "Average" | "Low" | "-",
              "Extraction Justification": "<short explanation describing why the decisions above were made>"
            }}
            Output only valid JSON. Do not explain anything.
        """)
    })

    return prompt_copy

def extract(raw_review: str, location: dict, client=None) -> dict:
    if client is None:
        client = LLMClient()

    full_prompt = generate_review_prompt(raw_review, location)
    response = client.call_LLM(full_prompt)

    if not response:
        return {"error": "No response", "raw_response": None}
    
    try:
        parsed = json.loads(response)
    except Exception as e:
        parsed = {"error": str(e), "raw_response": response}

    return parsed

######################### GET LLM Labelled Results ##########################

client = LLMClient()

for i, row in df.iterrows():
    review_text = row["text"]
    location = {
        "name": row["name"],
        "category": row["category"],
        "address": row["address"],
        "open_hours": row["hours"],
        "timestamp": row["time"]
    }

    full_prompt = generate_review_prompt(review_text, location)
    response = client.call_LLM(full_prompt)

    if not response:
        parsed = {"error": "No response", "raw_response": None}
    else:
        try:
            parsed = json.loads(response)
        except Exception as e:
            parsed = {"error": str(e), "raw_response": response}

    parsed["row_index"] = i
    results.append(parsed)

# Save results
# Create results dataframe and align it with input
results_df = pd.DataFrame(results)

# Rename result keys with 'res: ' prefix
results_df = results_df.rename(columns={
    "Advertisement": "res: Advertisement",
    "Irrelevant Review": "res: Irrelevant Review",
    "False Review": "res: False Review",
    "Vulgar Language": "res: Vulgar Language",
    "Relevance Score": "res: Relevance Score",
    "Quality Score": "res: Quality Score",
    "Extraction Justification": "res: Extraction Justification",
    "error": "res: error"
})

# Join results with original df using index
final_df = df.copy()
results_df.set_index("row_index", inplace=True)
final_df = final_df.join(results_df, how="left")

# Save
final_df.to_csv("moderated_reviews_with_results.csv", index=False)

