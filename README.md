# AI Resume Screening and Ranking System
The AI Resume Screening and Ranking System is an advanced application that leverages Natural Language Processing (NLP) and Machine Learning (ML) to evaluate and rank resumes based on their relevance to a job description. This tool is designed to automate and streamline the recruitment process by providing insights into candidate suitability, extracting key features, and generating actionable improvement suggestions.

# To Run this File Follow These Steps
Required Software:- 
1)visual studio code
2)python

# Features

Resume Text Extraction: Extracts text from uploaded PDF resumes.

Text Preprocessing: Cleans and tokenizes text for analysis.

Feature Extraction: Evaluates resumes based on keyword matching, sentiment analysis, and other metrics.

Skill Identification: Identifies relevant technical and soft skills.

Resume Scoring and Ranking: Scores and ranks resumes based on their relevance to the job description.

Improvement Suggestions: Provides tips to improve resumes.

Visualization: Displays comparative analysis through charts and tables.

# File Path Blue Print
## Project Root Directory ##
├── app.py                  # Main Streamlit application
├── deep_learning.py        # Module for deep learning tasks (e.g., BERT embeddings)
├── feature_extraction.py   # Module for text preprocessing and feature extraction
├── requirements.txt        # List of required Python packages
└── README.md               # Project documentation

# Step-by-Step Code Execution

## Step 1: Set Up the Environment ##
1) **Install Python:** Ensure you have Python 3.8 or later installed.
2) **Create a Virtual Environment:** python -m venv venv
3) **Activate the Virtual Environment:**
   For Windows:- .\venv\Scripts\activate
   For Macos/Linux:- source venv/bin/activate
## Step 2: Install Dependencies ##
Install the required libraries using the **requirements.txt** file:
pip install -r requirements.txt
## Step 3: Run the Application ##
Start the Streamlit app: streamlit run app.py

# Detailed Explanation

**1. Text Extraction and Preprocessing**

*Module:* feature_extraction.py

*Functionality:*

Extracts text from PDF files using PyPDF2.

Cleans text (removes special characters, converts to lowercase).

Tokenizes and lemmatizes text using NLTK.

**2. Feature Extraction**

*Module:* feature_extraction.py

**Key Features:**

*Keyword Match Ratio:* Measures overlap between resume and job description keywords.

*Sentiment Analysis:* Calculates polarity and subjectivity scores.

*Education and Experience Scores:* Assesses educational background and work experience.

**3. Deep Learning Integration**

*Module:* deep_learning.py

*Functionality:*

Leverages BERT for semantic similarity between resumes and job descriptions.

Computes embeddings and cosine similarity.

**4. Streamlit App**

*File:* app.py

*Features:*

Allows users to upload resumes and job descriptions.

Displays ranked resumes in a table with detailed analysis.

Provides visualizations of scores and identified skills.

# Example Workflow

**1)Upload Resumes and Job Description**

Upload multiple PDF resumes.

Enter or upload a job description.

**2)Process Resumes**

Click the "Process Resumes" button.

The system extracts, preprocesses, and analyzes resumes.

**3)View Results**

Navigate to the "Results & Rankings" tab to view ranked resumes.

Use the "Detailed Analysis" tab to analyze individual resumes.

**4)Generate Improvement Suggestions**

Receive actionable tips to enhance resume quality.

# Additional Notes

**1)Error Handling**

Ensure all necessary NLTK resources (e.g., punkt, stopwords) are downloaded.

Add missing libraries to requirements.txt and reinstall dependencies if needed.

**2)File Requirements**

Resumes: Must be in PDF format.

Job Description: Can be plain text or PDF.

# Requirements

Python 3.8+

**Libraries:**

  Streamlit

  PyPDF2

  NLTK

  TextBlob

  Transformers

  NumPy

  Pandas

# Future Enhancements

Add support for other file formats (e.g., DOCX).

Implement additional ML models for ranking.

Provide export options for ranked results.




