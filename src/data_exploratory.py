import pandas as pd
import ast  # to safely parse stringified lists
import helper

dat = pd.read_csv("../cleaned_data/cleaned_reviews.csv")

def reviews_per_category(df):
    """
    Groups reviews based on category and counts unique reviews per category.
    
    Args:
        df (pd.DataFrame): Input dataframe with at least ['category', 'text'] columns.
        
    Returns:
        pd.DataFrame: category and count of unique reviews.
    """
    # Ensure category column is a list (convert from string repr if needed)
    df['category'] = df['category'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Explode categories into separate rows
    exploded = df.explode('category')
    
    # Group by category and count unique reviews
    counts = (
        exploded.groupby('category')['text']
        .nunique()   # unique review texts per category
        .reset_index(name='unique_review_count')
        .sort_values(by='unique_review_count', ascending=False)
    )
    return counts

# Example: assuming your dataset is in df
category_counts = reviews_per_category(dat)
print(category_counts)


# to check for the existence of the PII elements in the review text
def checks_for_pii(df):
    """
    Adds columns to the dataframe indicating presence of PII elements in review text.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'text' column.
        
    Returns:
        pd.DataFrame: Original dataframe with added boolean columns for PII detection.
    """
    df = df.copy()
    df['has_email'] = df['text'].apply(lambda x: bool(helper.find_emails(x)))
    df['has_phone'] = df['text'].apply(lambda x: bool(helper.find_phone_numbers(x)))
    
    social_handles = df['text'].apply(helper.find_social_handles)
    df['has_telegram'] = social_handles.apply(lambda x: bool(x.get('telegram')))
    df['has_whatsapp'] = social_handles.apply(lambda x: bool(x.get('whatsapp')))
    df['has_dm_request'] = social_handles.apply(lambda x: bool(x.get('dm')))
    
    return df

counts_for_flagging = checks_for_pii(dat)
pii_cols = ["has_email", "has_phone", "has_telegram", "has_whatsapp", "has_dm_request"]

# filter rows where at least one of them is True
pii_rows = counts_for_flagging[counts_for_flagging[pii_cols].any(axis=1)]
pii_rows.to_csv("output.csv", index=False, encoding="utf-8")

