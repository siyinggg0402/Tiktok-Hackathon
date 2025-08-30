# ML Filtering of Reviews Based on Location
Online reviews are one of the most widely used sources of information when customers make decisions about restaurants, hotels, and local services. However, the **trustworthiness of reviews has been increasingly challenged** by the presence of spam, irrelevant comments, location-inconsistent reviews (e.g., reviews posted in the wrong region), or even AI-generated content.

This project focuses on **filtering and classifying Local reviews by leveraging both review text and location metadata**.  
The key idea is that a review is only useful if it is consistent with the business’s location and context.  
For example, a review mentioning a restaurant chain in New York but attached to a location in Vermont may be suspicious or unhelpful.

To address this, we design a pipeline that combines **data preprocessing, LLM-based structuring, and validation**:

### 🔹 Data Preprocessing
- Clean and normalize the raw review and metadata files.  
- Handle missing values and standardize location information.  
- Merge text with associated metadata.  
- Store the processed output in a structured format (`cleaned_reviews.xlsx`).  

### 🔹 LLM-Based Structuring
- Use a **large language model (LLM)** with prompt engineering and fine-tuning strategies.  
- Classify reviews into categories such as:
  - *relevant*  
  - *irrelevant*  
  - *spam/advertisement*  
  - *rant without visit*  
  - *genuine feedback*  
- The LLM captures nuanced text (e.g., sarcasm, off-topic rants) that traditional ML classifiers may miss.  
- Metadata (e.g., location and business category) is integrated into prompts to make classification **context-aware**, ensuring reviews align with their claimed location.  

### 🔹 Validation
- Compare the model’s classifications against a **human-labeled ground truth dataset**.  
- Evaluate performance using metrics such as:
  - **Accuracy**  
  - **F1-score**  
  - **Confusion matrices**  
- Assess how well the pipeline distinguishes **trustworthy vs. untrustworthy reviews**.  


---

## Project Structure

```
├── data/ # Raw data storage (reviews + metadata)
│ ├── reviews/ # Downloaded reviews dataset
│ └── metadata/ # Downloaded metadata dataset
│
├── cleaned_data/ # Output folder for cleaned review data
│ └── cleaned_reviews.xlsx
│
├── src/ # Source code
│ ├── data_preprocessing.py # Script to clean and structure raw review data
│ ├── LLM_structuring.py # Script to run training/structuring cases
│ └── validation.py # Script to run validation against human-labeled cases
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```
---

## Dataset

We use the **Google Local Reviews dataset** curated by McAuley Lab (UCSD):  
[Download dataset here](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)

You’ll need to download both:
- `reviews` dataset  
- `metadata` dataset  
You may choose to download the datasets that were from Vermont.
Store them under the `data/` folder as shown above.

---

## ⚙️ Setup Instructions

1. **Clone this repository**  
   ```bash
   git clone https://github.com/siyinggg0402/Tiktok-Hackathon.git
   cd Tiktok-Hackathon

Our members, Amelia, Si Ying, Su En and Sze Yui worked on different parts that make up to the project together.
