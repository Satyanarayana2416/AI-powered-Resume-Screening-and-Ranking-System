import re
import nltk
import numpy as np
from PyPDF2 import PdfReader
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Ensure NLTK resources are downloaded
download_nltk_resources()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file
    
    Args:
        pdf_file: The uploaded PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def preprocess_text(text):
    """
    Preprocess the extracted text
    
    Args:
        text: The extracted text from the PDF
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        # Join tokens back into text
        preprocessed_text = ' '.join(filtered_tokens)
        return preprocessed_text
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return ""

def extract_features(text, job_description):
    """
    Extract features from the resume text based on the job description
    
    Args:
        text: The preprocessed resume text
        job_description: The preprocessed job description text
        
    Returns:
        dict: Features extracted from the resume
    """
    features = {}
    try:
        # Calculate text similarity with job description
        resume_blob = TextBlob(text)
        job_blob = TextBlob(job_description)
        
        # Extract resume words and job description words
        resume_words = set(text.split())
        job_words = set(job_description.split())
        
        # Calculate keyword matching
        matching_keywords = resume_words.intersection(job_words)
        features['keyword_match_ratio'] = len(matching_keywords) / len(job_words) if job_words else 0
        
        # Calculate sentiment
        features['sentiment_polarity'] = resume_blob.sentiment.polarity
        features['sentiment_subjectivity'] = resume_blob.sentiment.subjectivity
        
        # Calculate text length features
        features['word_count'] = len(text.split())
        features['unique_word_count'] = len(set(text.split()))
        features['word_density'] = features['unique_word_count'] / features['word_count'] if features['word_count'] > 0 else 0
        
        # Education keywords
        education_keywords = ['degree', 'bachelor', 'master', 'phd', 'diploma', 'university', 'college', 'school', 'education']
        features['education_score'] = sum(1 for keyword in education_keywords if keyword in text.split()) / len(education_keywords)
        
        # Experience keywords
        experience_keywords = ['experience', 'year', 'work', 'project', 'develop', 'manage', 'lead', 'team', 'skill']
        features['experience_score'] = sum(1 for keyword in experience_keywords if keyword in text.split()) / len(experience_keywords)
    except Exception as e:
        print(f"Error extracting features: {e}")
    
    return features

def extract_skills(text):
    """
    Extract skills from the resume text
    
    Args:
        text: The preprocessed resume text
        
    Returns:
        list: List of potential skills found in the resume
    """
    # Common technical skills
    technical_skills = [
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go',
        'html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'express',
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql', 'firebase', 'aws', 'azure', 'gcp',
        'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
        'machine learning', 'deep learning', 'ai', 'data science', 'tensorflow', 'pytorch', 'keras',
        'nlp', 'computer vision', 'data analysis', 'data visualization', 'tableau', 'power bi',
        'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'ci/cd', 'test driven development'
    ]
    
    # Soft skills
    soft_skills = [
        'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
        'time management', 'adaptability', 'creativity', 'collaboration', 'organization',
        'decision making', 'conflict resolution', 'emotional intelligence', 'negotiation',
        'presentation', 'public speaking', 'writing', 'customer service', 'interpersonal'
    ]
    
    # Combine all skills
    all_skills = technical_skills + soft_skills
    
    # Find skills in the resume
    found_skills = [skill for skill in all_skills if skill in text.lower()]
    
    return found_skills

