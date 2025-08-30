from LLMClient import LLMClient
import pandas as pd
from textwrap import dedent
import json
import csv

#load dataset
df = pd.read_excel("../cleaned_data/cleaned_reviews.csv") 
results = []

prompt = [
    {
        "role": "system",
        "content": dedent(
            """
            You are a professional review moderator, specialising in evaluating Google Reviews. Your job is to evaluate whether a user review for a business location is relevant, high-quality, and policy-compliant.

            You have to follow these policies strictly and flag out any one of the violations clearly:

            1. Advertisements: Reviews that contain promotional language, calls to action, or any URLs/links are considered violations. 
            2. Irrelevant content: Reviews that discuss topics unrelated to the business, unrelated personal experiences, or general commentary, are considered violations.
            3. False reviews: Reviews that come from users who have not visited the location, or that contain fabricated information, are considered violations.
            4. Vulgarities: Reviews that contain vulgar, offensive, or inappropriate language are considered violations.

            If any of the 4 above policies are violated, you are to flag them out acccording to the policy that was violated, stating them clearly.

            If the reviews do not violate any of the above policies, you will move on to evaluate the review based on the following criteria. You will assign a relevance score, quality score, and visited likelihood score based on the criteria below:

            1. Relevance (High, Medium, Low):
                - Low (0 points): No mention of the business or its services/products.
                - Average (1-2 points): There is some mention of the business and the services or products it provides, as well as mention of staff name or opening hours, but not in great detail. 
                - High (3 points): Detailed mention of the business and the location, its services/products, staff, and specific experiences.

                According to the above criteria, you will assign a relevance score from 0-100, where 0 is completely irrelevant and 100 is highly relevant.

            2. Quality (High, Medium, Low):
                - Low (0 point): No text or gibberish text, or reviews containing improper languages. Reviews which only contain 1-2 words are also considered low quality.
                - Low (1 point): Very brief reviews with minimal detail, such as 3-5 words. 
                - Average (2 points): Reviews with some detail about the experience, but lacking depth or specifics.
                - High (3 points): Detailed reviews that provide specific information about the experience, including aspects like quality, service, ambiance, and value for money. Reviews also contain proper language use and are structured well.

                According to the above criteria, you will assign a quality score from 0-100, where 0 is very poor quality and 100 is excellent quality.

            3. Visited Likelihood (0-100):
                Based on the above two scores, you will estimate the likelihood that the reviewer has actually visited the location. A score of 0 indicates no likelihood of visit, while a score of 100 indicates definite visit.

            Finally, for each review, you will analyse its content along with the location metadata, and return your decision in the following strict JSON format:

            {
            "relevance": "high" | "medium" | "low",
            "quality": "high" | "medium" | "low",
            "policy_flags": ["advertisement", "irrelevant", "false_review"],  // may be empty if no violations are found
            "justification": [string],    
            "relevance_score": 0-100,
            "quality_score": 0-100,
            "visited_likelihood": 0-100
            }

            Think in a step-by-step manner, then output only valid JSON.

            Follow these steps to extract details of valid reviews:
            1. First, identify all mentioned products or services
            2. Then, find any dates, times, or locations
            3. Look for names, contact information, or IDs
            4. Extract any quantities, prices, or measurements
            5. Note any issues, complaints, or requests
            6. Format the results as specified below
            """
        )
    },
    {"role": "assistant", "content": """
    
        Here is a clear example of a review evaluation:

            Review:
            "The Plumbing Bros provided excellent service. Gabriel was professional and did not make me wait for long, fixing my toilet in a short span of time. The pricing was very reasonable and cheaper than most found in the market. They also provide high quality plumbing products, such as their pipe wrenches, for us to choose from. Highly recommend this service and their products for anyone in need of plumbing work."

            Location:
            Name: Plumbing Bros
            Category: [Plumbing Services, Home Services, Plumbing Products]
            Address: 123 Main St, Springfield, USA
            Opening Hours: [['Thursday', '8AM-5PM'], ['Friday', '8AM-5PM'], ['Saturday', 'Closed'], ['Sunday', 'Closed'], ['Monday', '8AM-5PM'], ['Tuesday', '8AM-5PM'], ['Wednesday', '8AM-5PM']]
            Timestamp: 2021-06-10 09:50:58 EDT

            Please return the moderation result in the specified JSON format. Below is an example of the expected output:

            {
            "relevance": "high",
            "quality": "high",
            "policy_flags": [],
            "justification": "Mentions staff name, services provided and pricing with clear detail.",
            "relevance_score": 95,
            "quality_score": 90,
            "visited_likelihood": 95
            }
    
        """},
    {"role": "user", "content": """
        Help me evaluate the following review based on the policies and criterias. 

        Review:
        "The Hawaiian pizza was absolutely delicious and came out piping hot, and was priced affordably. The staff, especially Maria, was very friendly and recommended their house special. The ambiance was cozy, with amazing ocean views nearby, though the wait time was a little long. Overall, great experience!"

        Location:
        Name: Pizza Palace
        Category: [Restaurant, Food, Italian]
        Address: 456 Market St, Springfield, USA
        Opening Hours: [['Monday','10AM–10PM'],['Tuesday','10AM–10PM'],['Wednesday','10AM–10PM'],['Thursday','10AM–10PM'],['Friday','10AM–11PM'],['Saturday','11AM–11PM'],['Sunday','11AM–9PM']]
        Timestamp: 2021-07-15 18:20:00 EDT

        Please return the moderation result in the specified JSON format.

        """
    },
    {"role": "assistant", "content": """Thank you for the example. I am ready to carry out the review evaluation task now. Here is the evaluation result:
    
    {
        "relevance": "high",
        "quality": "high",
        "policy_flags": [],
        "justification": "Mentions specific details about the pizza, staff, ambiance, and overall experience.",
        "relevance_score": 90,
        "quality_score": 85,
        "visited_likelihood": 90
    }
    
    """}
]

def generate_review_prompt(review_text: str, location: dict) -> list[dict]:
    prompt = []

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
            Opening Hours: {location.get('hours')}
            Timestamp: {location.get('time')}

            Please return the moderation result in the specified JSON format.
        """)
    })

    return prompt

######################### GET LLM Labelled Results ##########################
"""
client = LLMClient()

for i, row in df.iterrows():
    review_text = row["text"]
    location = {
        "name": row.get("name", ""),
        "category": row.get("category", ""),
        "address": row.get("address", ""),
        #"menu": row.get("menu", ""),
        "open_hours": row.get("hours", ""),
        "timestamp": row.get("time", "")
    }

    full_prompt = generate_review_prompt(review_text, location)
    response = client.call_LLM(full_prompt)

    try:
        parsed = json.loads(response)
    except Exception as e:
        parsed = {"error": str(e), "raw_response": response}

    results.append(parsed)
"""