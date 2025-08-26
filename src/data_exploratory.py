import pandas as pd
import ast  # to safely parse stringified lists

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


