import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

class ResumeRanker:
    def __init__(self):
        """Initialize the ResumeRanker with necessary models and components"""
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Load pre-trained BERT model for text embeddings
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            self.tokenizer = None
            self.bert_model = None
            
        # Feature importance weights (can be adjusted based on job requirements)
        self.feature_weights = {
            'keyword_match_ratio': 0.35,
            'education_score': 0.15,
            'experience_score': 0.25,
            'word_density': 0.05,
            'sentiment_polarity': 0.05,
            'sentiment_subjectivity': 0.05,
            'bert_similarity': 0.10
        }
    
    def get_bert_embedding(self, text):
        """
        Get BERT embeddings for a given text
        
        Args:
            text: The preprocessed text
            
        Returns:
            numpy.ndarray: BERT embedding vector
        """
        if self.tokenizer is None or self.bert_model is None:
            # Return zeros if BERT model is not available
            return np.zeros(768)
            
        try:
            # Tokenize and get BERT embeddings
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            # Use the [CLS] token embedding as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            return embeddings[0]
        except Exception as e:
            print(f"Error generating BERT embeddings: {e}")
            return np.zeros(768)
    
    def calculate_bert_similarity(self, resume_text, job_description):
        """
        Calculate similarity between resume and job description using BERT embeddings
        
        Args:
            resume_text: The preprocessed resume text
            job_description: The preprocessed job description text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if self.tokenizer is None or self.bert_model is None:
            return 0.5  # Default value if BERT is not available
            
        try:
            # Get embeddings
            resume_embedding = self.get_bert_embedding(resume_text)
            job_embedding = self.get_bert_embedding(job_description)
            
            # Calculate cosine similarity
            similarity = np.dot(resume_embedding, job_embedding) / (
                np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
            )
            
            # Normalize to 0-1 range
            similarity = (similarity + 1) / 2
            return similarity
        except Exception as e:
            print(f"Error calculating BERT similarity: {e}")
            return 0.5
    
    def score_resume(self, features, resume_text, job_description):
        """
        Score a resume based on extracted features and similarity to job description
        
        Args:
            features: Dictionary of features extracted from the resume
            resume_text: The preprocessed resume text
            job_description: The preprocessed job description text
            
        Returns:
            float: Score between 0 and 1
        """
        # Calculate BERT similarity if not already in features
        if 'bert_similarity' not in features:
            features['bert_similarity'] = self.calculate_bert_similarity(resume_text, job_description)
        
        # Calculate weighted score
        score = 0
        for feature, weight in self.feature_weights.items():
            if feature in features:
                score += features[feature] * weight
        
        return min(max(score, 0), 1)  # Ensure score is between 0 and 1
    
    def rank_resumes(self, resumes_data, job_description):
        """
        Rank multiple resumes based on their scores
        
        Args:
            resumes_data: List of dictionaries containing resume features and text
            job_description: The preprocessed job description text
            
        Returns:
            pandas.DataFrame: Ranked resumes with scores
        """
        results = []
        
        for resume in resumes_data:
            score = self.score_resume(resume['features'], resume['text'], job_description)
            
            result = {
                'filename': resume['filename'],
                'score': score,
                'features': resume['features'],
                'skills': resume.get('skills', [])
            }
            
            results.append(result)
        
        # Create DataFrame and sort by score
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        return df_results
    
    def get_improvement_suggestions(self, features, skills, job_description):
        """
        Generate suggestions for improving a resume based on its features and the job description
        
        Args:
            features: Dictionary of features extracted from the resume
            skills: List of skills found in the resume
            job_description: The preprocessed job description text
            
        Returns:
            list: List of improvement suggestions
        """
        suggestions = []
        
        # Check keyword match ratio
        if features['keyword_match_ratio'] < 0.2:
            suggestions.append("Consider adding more keywords from the job description to your resume.")
        
        # Check education score
        if features['education_score'] < 0.3:
            suggestions.append("Add more details about your educational background.")
        
        # Check experience score
        if features['experience_score'] < 0.3:
            suggestions.append("Elaborate more on your work experience and projects.")
        
        # Check word density
        if features['word_density'] < 0.4:
            suggestions.append("Try to use more diverse vocabulary in your resume.")
        elif features['word_density'] > 0.8:
            suggestions.append("Your resume might be too sparse. Consider adding more detailed descriptions.")
        
        # Check word count
        if features['word_count'] < 200:
            suggestions.append("Your resume seems too short. Consider adding more content.")
        elif features['word_count'] > 1000:
            suggestions.append("Your resume might be too long. Consider making it more concise.")
        
        # Extract important skills from job description that are missing in the resume
        job_words = set(job_description.lower().split())
        important_skills = ['python', 'java', 'javascript', 'machine learning', 'data science', 
                           'sql', 'aws', 'azure', 'docker', 'kubernetes', 'react', 'angular']
        
        missing_skills = [skill for skill in important_skills 
                         if skill in job_description.lower() and skill not in skills]
        
        if missing_skills:
            skill_suggestion = "Consider adding these skills if you have them: " + ", ".join(missing_skills)
            suggestions.append(skill_suggestion)
        
        return suggestions
