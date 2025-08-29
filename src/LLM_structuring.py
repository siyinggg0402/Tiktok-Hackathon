from LLMClient import LLMClient
import pandas as pd
from textwrap import dedent
import json

#load dataset
df = pd.read_csv("../cleaned_data/cleaned_reviews.csv")
results = []

prompt = [
    {
        "role": "system",
        "content": dedent(
            """
            You are a professional review moderator, specialising in evaluating Google Reviews. Your job is to evaluate whether a user review for a business location is relevant, high-quality, and policy-compliant.

            Policies to enforce:

            1. Advertisements: There should be strictly no advertisements or promotional content. Reviews that contain promotional language, calls to action, or any URLs/links are considered violations. 
            For example, reviews that containt statements such as "Check our our website here! at www.example.com" or "Use the code SAVE20 for a special discount!" are violations.

            2. Irrelevant content: Reviews must be relevant to the business. Reviews that discuss topics unrelated to the business, such as politics, unrelated personal experiences, or general commentary, are considered violations.
            For example, a review for a restaurant discusses about politics or a giving experiences of a different restaurant are violations.

            3. False reviews: There should be no false reviews or rants that do not reflect customer experiences. Reviews that come from users who have not visited the location, or that contain fabricated information, are considered violations.
            For example, a review that states "I never visited this place but heard it's bad" or "This place is terrible, I had a horrible experience" without any specific details are violations.

            If any of the 3 above policies are violated, they should be flagged out.
            After flagging out the violations, you will move on to the review evaluation scorings.

            Review Evaluation Criteria:

            1. Relevance (High, Medium, Low):
                - Low (0 points): No mention of the business or its services/products. For example, "This place is in a great location. Highly recommended".
                - Average (1-2 points): There is some mention of the business and the services or products it provides, as well as mention of staff name or opening hours, but not in great detail. For example, "The food is great and Anne was a friendly server".
                - High (3 points): Detailed mention of the business and the location, its services/products, staff, and specific experiences. For example, "The food was delicious, especially the best-selling Aglio Olio pasta. Anne was a friendly server who made great recommendations, and was very patient in answering our queries. The ambiance was cozy and perfect for a date night, especially with the nearby ocean views and ferris wheel."

                According to the above criteria, assign a relevance score from 0-100, where 0 is completely irrelevant and 100 is highly relevant.

            2. Quality (High, Medium, Low):
                - Low (0 point): No text or gibberish text, or reviews containing improper languages. Reviews which only contain 1-2 words are also considered low quality. For example, "Good", "Bad", "Ok", "Meh", "asdnjkwe".
                - Low (1 point): Very brief reviews with minimal detail, such as 3-5 words. For example, "Great products and service".
                - Average (2 points): Reviews with some detail about the experience, but lacking depth or specifics. For example, "The staff were friendly, but the place was a bit noisy".
                - High (3 points): Detailed reviews that provide specific information about the experience, including aspects like quality, service, ambiance, and value for money. Reviews also contain proper language use and are structured well. For example, "The plumbing services provided was excellent. The plumber, Alex, was punctual, professional, and fixed my leaking sink quickly. The pricing was transparent and fair. Highly recommend this service for anyone in need of plumbing work."

                According to the above criteria, assign a quality score from 0-100, where 0 is very poor quality and 100 is excellent quality.

            3. Visited Likelihood (0-100):
                Based on the above two scores, estimate the likelihood that the reviewer has actually visited the location. A score of 0 indicates no likelihood of visit, while a score of 100 indicates definite visit.

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
    {"role": "assistant", "content": """
    
        Here is a clear example of a review evaluation:
            {
            "relevance": "high",
            "quality": "high",
            "policy_flags": [],
            "justification": "Mentions staff, product, ambiance, and pricing with clear detail.",
            "relevance_score": 95,
            "quality_score": 90,
            "visited_likelihood": 95
            }
    
        """},
    {"role": "user", "content": """
        Review:
            "The Plumbing Bros provided excellent service. Gabriel was professional and did not make me wait for long, fixing my toilet in a short span of time. The pricing was very reasonable and cheaper than most found in the market. They also provide high quality plumbing products, such as their pipe wrenches, for us to choose from. Highly recommend this service and their products for anyone in need of plumbing work."

            Location:
            Name: Plumbing Bros
            Category: [Plumbing Services, Home Services, Plumbing Products]
            Address: 123 Main St, Springfield, USA
            Opening Hours: [['Thursday', '8AM‚Äì5PM'], ['Friday', '8AM‚Äì5PM'], ['Saturday', 'Closed'], ['Sunday', 'Closed'], ['Monday', '8AM‚Äì5PM'], ['Tuesday', '8AM‚Äì5PM'], ['Wednesday', '8AM‚Äì5PM']]
            Timestamp: 2021-06-10 09:50:58 EDT

            Please return the moderation result in the specified JSON format.

        """
    },
    {"role": "assistant", "content": "Thank you for the example. I am ready to carry out the review evaluation task. Please provide the dataset now."}
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
