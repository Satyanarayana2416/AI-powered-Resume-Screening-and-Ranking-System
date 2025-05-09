import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import tempfile
from feature_extraction import (
    extract_text_from_pdf,
    preprocess_text,
    extract_features,
    extract_skills
)
from deep_learning import ResumeRanker

# Set page configuration
st.set_page_config(
    page_title="AI Resume Screening And Candidate Ranking System", 
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'ranked_resumes' not in st.session_state:
    st.session_state.ranked_resumes = None
if 'processed_resumes' not in st.session_state:
    st.session_state.processed_resumes = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 5px;
        border-radius: 5px;
    }
    .score-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .score-medium {
        color: #F57F17;
        font-weight: bold;
    }
    .score-low {
        color: #C62828;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>AI Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='card'>
    <p>This application uses AI to analyze and rank resumes based on their relevance to a job description. 
    Upload multiple resumes and a job description to see which candidates are the best match.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2>Settings & Information</h2>", unsafe_allow_html=True)
    
    st.markdown("### About")
    st.markdown("""
    This tool uses Natural Language Processing and Machine Learning to:
    - Extract text from PDF resumes
    - Analyze resume content
    - Match skills with job requirements
    - Rank candidates based on relevance
    - Provide improvement suggestions
    """)
    
    st.markdown("### How It Works")
    st.markdown("""
    1. Upload multiple resumes (PDF format)
    2. Enter or upload a job description
    3. Click 'Process Resumes' to analyze
    4. View ranked results and detailed analysis
    """)
    
    st.markdown("### Features")
    st.markdown("""
    - Resume text extraction
    - Keyword matching
    - Skills identification
    - Resume scoring and ranking
    - Improvement suggestions
    - Comparative analysis
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Upload & Process", "Results & Rankings", "Detailed Analysis"])

with tab1:
    st.markdown("<h2 class='sub-header'>Upload Resumes & Job Description</h2>", unsafe_allow_html=True)
    
    # File uploader for resumes
    uploaded_resumes = st.file_uploader("Upload Resumes (PDF format)", type="pdf", accept_multiple_files=True)
    
    # Job description input
    st.markdown("<h3>Job Description</h3>", unsafe_allow_html=True)
    job_desc_option = st.radio("Choose input method:", ["Enter Text", "Upload File"])
    
    if job_desc_option == "Enter Text":
        job_description = st.text_area("Enter the job description:", height=200)
    else:
        uploaded_job_desc = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])
        if uploaded_job_desc is not None:
            if uploaded_job_desc.type == "application/pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_job_desc.getvalue())
                    tmp_file_path = tmp_file.name
                
                job_description = extract_text_from_pdf(tmp_file_path)
                os.unlink(tmp_file_path)  # Delete the temporary file
            else:  # txt file
                job_description = uploaded_job_desc.getvalue().decode("utf-8")
            
            st.text_area("Extracted Job Description:", job_description, height=200)
        else:
            job_description = ""
    
    # Process button
    if st.button("Process Resumes"):
        if not uploaded_resumes:
            st.error("Please upload at least one resume.")
        elif not job_description:
            st.error("Please provide a job description.")
        else:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize resume ranker
            ranker = ResumeRanker()
            
            # Process job description
            status_text.text("Processing job description...")
            preprocessed_job_desc = preprocess_text(job_description)
            
            # Store job description in session state
            st.session_state.job_description = job_description
            
            # Process each resume
            processed_resumes = []
            for i, resume_file in enumerate(uploaded_resumes):
                progress = (i + 1) / (len(uploaded_resumes) + 1)
                progress_bar.progress(progress)
                status_text.text(f"Processing resume {i+1}/{len(uploaded_resumes)}: {resume_file.name}")
                
                # Create a temporary file to save the uploaded PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(resume_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text from PDF
                resume_text = extract_text_from_pdf(tmp_file_path)
                os.unlink(tmp_file_path)  # Delete the temporary file
                
                # Preprocess text
                preprocessed_text = preprocess_text(resume_text)
                
                # Extract features and skills
                features = extract_features(preprocessed_text, preprocessed_job_desc)
                skills = extract_skills(preprocessed_text)
                
                # Store processed resume data
                processed_resume = {
                    'filename': resume_file.name,
                    'text': preprocessed_text,
                    'raw_text': resume_text,
                    'features': features,
                    'skills': skills
                }
                
                processed_resumes.append(processed_resume)
            
            # Rank resumes
            status_text.text("Ranking resumes...")
            ranked_df = ranker.rank_resumes(processed_resumes, preprocessed_job_desc)
            
            # Store results in session state
            st.session_state.ranked_resumes = ranked_df
            st.session_state.processed_resumes = processed_resumes
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("Processing complete! Go to the Results tab to view rankings.")
            
            # Add success message
            st.success(f"Successfully processed {len(uploaded_resumes)} resumes!")

with tab2:
    st.markdown("<h2 class='sub-header'>Resume Rankings</h2>", unsafe_allow_html=True)
    
    if st.session_state.ranked_resumes is not None:
        # Display ranking table
        ranked_df = st.session_state.ranked_resumes
        
        # Create a more user-friendly display dataframe
        display_df = pd.DataFrame({
            'Rank': range(1, len(ranked_df) + 1),
            'Resume': ranked_df['filename'],
            'Match Score': [f"{score:.2f}" for score in ranked_df['score']]
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # Create bar chart of scores
        st.markdown("<h3>Score Comparison</h3>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(ranked_df['filename'], ranked_df['score'], color='#1E88E5')
        
        # Add score labels
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{ranked_df['score'].iloc[i]:.2f}",
                va='center'
            )
        
        ax.set_xlabel('Match Score')
        ax.set_title('Resume Ranking by Match Score')
        ax.set_xlim(0, 1)
        
        # Display the chart
        st.pyplot(fig)
        
        # Top skills section
        st.markdown("<h3>Top Skills in Selected Resumes</h3>", unsafe_allow_html=True)
        
        # Get all skills from top 3 resumes (or all if less than 3)
        top_n = min(3, len(st.session_state.processed_resumes))
        top_resumes_indices = ranked_df.index[:top_n].tolist()
        
        all_skills = []
        for idx in top_resumes_indices:
            resume = st.session_state.processed_resumes[idx]
            all_skills.extend(resume['skills'])
        
        # Count skill frequencies
        skill_counts = pd.Series(all_skills).value_counts()
        
        # Display top 10 skills
        if not skill_counts.empty:
            top_skills = skill_counts.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(top_skills.index, top_skills.values, color='#42A5F5')
            
            ax.set_xlabel('Frequency')
            ax.set_title('Top Skills in Highest-Ranked Resumes')
            
            # Display the chart
            st.pyplot(fig)
        else:
            st.info("No skills detected in the top resumes.")
    else:
        st.info("No resumes have been processed yet. Please go to the Upload & Process tab.")

with tab3:
    st.markdown("<h2 class='sub-header'>Detailed Resume Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.ranked_resumes is not None:
        # Select resume for detailed analysis
        resume_options = st.session_state.ranked_resumes['filename'].tolist()
        selected_resume = st.selectbox("Select a resume for detailed analysis:", resume_options)
        
        # Get the selected resume data
        selected_idx = st.session_state.ranked_resumes[st.session_state.ranked_resumes['filename'] == selected_resume].index[0]
        resume_data = st.session_state.processed_resumes[selected_idx]
        
        # Display resume score
        score = st.session_state.ranked_resumes.loc[selected_idx, 'score']
        score_class = "score-high" if score >= 0.7 else "score-medium" if score >= 0.4 else "score-low"
        
        st.markdown(f"""
        <div class='card'>
            <h3>Resume: {selected_resume}</h3>
            <p>Match Score: <span class='{score_class}'>{score:.2f}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature breakdown
        st.markdown("<h3>Feature Breakdown</h3>", unsafe_allow_html=True)
        
        features = resume_data['features']
        feature_df = pd.DataFrame({
            'Feature': list(features.keys()),
            'Value': list(features.values())
        })
        
        # Create a horizontal bar chart for features
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(feature_df['Feature'], feature_df['Value'], color='#5C6BC0')
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{feature_df['Value'].iloc[i]:.2f}",
                va='center'
            )
        
        ax.set_xlabel('Score')
        ax.set_title('Feature Scores')
        ax.set_xlim(0, 1)
        
        # Display the chart
        st.pyplot(fig)
        
        # Skills section
        st.markdown("<h3>Detected Skills</h3>", unsafe_allow_html=True)
        
        skills = resume_data['skills']
        if skills:
            # Display skills as a tag cloud
            skills_html = " ".join([f"<span class='highlight'>{skill}</span>" for skill in skills])
            st.markdown(f"<div style='line-height: 2.5'>{skills_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No specific skills detected in this resume.")
        
        # Improvement suggestions
        st.markdown("<h3>Improvement Suggestions</h3>", unsafe_allow_html=True)
        
        # Initialize resume ranker to get suggestions
        ranker = ResumeRanker()
        suggestions = ranker.get_improvement_suggestions(
            resume_data['features'],
            resume_data['skills'],
            preprocess_text(st.session_state.job_description)
        )
        
        if suggestions:
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        else:
            st.success("This resume is well-optimized for the job description!")
        
        # Raw text section (collapsible)
        with st.expander("View Raw Resume Text"):
            st.text(resume_data['raw_text'])
    else:
        st.info("No resumes have been processed yet. Please go to the Upload & Process tab.")

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 3rem; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>
    <p>AI Resume Screening System | Powered by Python, Streamlit, and Machine Learning</p>
</div>
""", unsafe_allow_html=True)
