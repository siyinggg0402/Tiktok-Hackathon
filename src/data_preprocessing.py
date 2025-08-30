# clean and remove null values
# match the metadata and review to the corresponding 
import pandas as pd
import numpy as np
import os
import csv
import helper

def match_metadata_reviews(metadata, reviews):

    merged_df = pd.merge(metadata, reviews, on='gmap_id', how='inner')

    merged_df["text"] = (
    merged_df["text"]
      .astype(str)
      .str.replace("\n", " ", regex=False)
      .str.replace(r"\u2013", "-", regex=False)
      .str.replace(r"\u2019", "`", regex=False)
      .str.replace(r"\s*/u\w+\s*", " ", regex=True)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)
    print(merged_df.head())
    cleaned_df = merged_df.drop(columns=["num_of_reviews", "avg_rating","price","MISC","state","user_id","relative_results","url","description","user_id","name_y","pics", "resp"])
    cleaned_df = cleaned_df.rename(columns={"name_x": "name"})
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.drop_duplicates(subset=["text"], keep="first")
    cleaned_df = cleaned_df[cleaned_df["text"].str.lower() != "none"]
    # remove rows with null values in critical columns
    #cleaned_df = merged_df.dropna(subset=['name', 'category', 'latitude', 'longitude','text','hours'])
    for idx, row in cleaned_df.iterrows():
        cleaned_df.at[idx, "time"] = helper.unix_to_vermont_time(int(row["time"])//1000)
        cleaned_df.at[idx, "text"] = helper.normalize_whitespace(row["text"])
        cleaned_df.at[idx, "text"] = helper.standardize_quotes_dashes(row["text"])
        cleaned_df.at[idx, "text"] = helper.clean_emojis(row["text"])

    return cleaned_df


def main():
    meta = pd.read_json("../data/meta-Vermont.json", lines=True)
    reviews = pd.read_json("../data/review-Vermont.json",lines= True)
    cleaned = match_metadata_reviews(meta,reviews)
    output_dir = "../cleaned_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cleaned_reviews.csv")
    cleaned.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    main()
