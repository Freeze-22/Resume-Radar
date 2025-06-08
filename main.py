# To run this Streamlit application, ensure you have the following libraries installed.
# You can install them using pip:
# pip install streamlit PyPDF2 python-docx requests

import streamlit as st
import PyPDF2
from docx import Document # pip install python-docx
import re
import requests # Used for making the API call to the LLM
import json # Used for JSON parsing of LLM response

# --- 1. Define Preexisting Keywords ---
# This dictionary serves as the "database" for relevant keywords for different designations.
designation_keywords = {
    "Software Engineer": [
        "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue.js",
        "Node.js", "Django", "Flask", "Spring Boot", "REST API", "microservices",
        "SQL", "NoSQL", "Git", "Agile", "Scrum", "data structures", "algorithms",
        "cloud computing", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
        "test-driven development", "CI/CD", "object-oriented programming", "backend",
        "frontend", "full-stack", "debugging", "problem-solving", "critical thinking",
        "software development life cycle", "SDLC", "version control", "unit testing"
    ],
    "Data Scientist": [
        "Python", "R", "SQL", "machine learning", "deep learning", "statistical modeling",
        "data analysis", "data visualization", "Pandas", "NumPy", "Scikit-learn",
        "TensorFlow", "Keras", "PyTorch", "Jupyter Notebook", "predictive modeling",
        "natural language processing", "NLP", "computer vision", "A/B testing",
        "experimental design", "big data", "Hadoop", "Spark", "cloud platforms",
        "AWS Sagemaker", "Azure ML", "GCP AI Platform", "communication",
        "storytelling", "problem-solving", "mathematics", "statistics", "ETL",
        "data cleaning", "feature engineering", "model deployment", "model evaluation"
    ],
    "Marketing Manager": [
        "digital marketing", "content marketing", "SEO", "SEM", "social media marketing",
        "email marketing", "PPC", "Google Ads", "Facebook Ads", "campaign management",
        "brand strategy", "market research", "analytics", "Google Analytics",
        "CRM", "HubSpot", "Salesforce", "marketing automation", "lead generation",
        "conversion optimization", "public relations", "event marketing",
        "budget management", "team leadership", "communication", "creativity",
        "strategic planning", "data analysis", "ROI", "customer segmentation",
        "omnichannel marketing", "customer journey", "A/B testing", "market segmentation"
    ],
    "Product Manager": [
        "product lifecycle", "roadmap", "user stories", "agile development", "scrum",
        "market research", "competitor analysis", "product strategy", "MVP",
        "feature prioritization", "A/B testing", "user experience", "UX",
        "data analysis", "stakeholder management", "communication", "technical understanding",
        "business acumen", "design thinking", "customer feedback", "product launch",
        "go-to-market strategy", "cross-functional teams", "Jira", "Confluence"
    ]
}

# --- Helper Functions for Resume Parsing ---
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    Args:
        pdf_file: A file-like object (e.g., from st.file_uploader).
    Returns:
        The extracted text as a string.
    """
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}. Please ensure it's a valid PDF.")
        return ""

def extract_text_from_docx(docx_file):
    """
    Extracts text from a DOCX file.
    Args:
        docx_file: A file-like object (e.g., from st.file_uploader).
    Returns:
        The extracted text as a string.
    """
    try:
        document = Document(docx_file)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}. Please ensure it's a valid DOCX.")
        return ""

def clean_text(text):
    """
    Cleans the extracted text by converting to lowercase, removing non-alphanumeric characters,
    and normalizing whitespace.
    Args:
        text (str): The raw text extracted from the resume.
    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    # Remove characters that are not letters, numbers, or spaces
    # NOTE: This regex keeps alphanumeric and spaces. If keywords like "C++" or "C#"
    # are expected to match, they will be converted to "c" after this step.
    # For more precise matching of such terms, a more sophisticated NLP approach
    # like tokenization and lemma matching would be needed.
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Normalize whitespace: replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Score Calculation Function ---
def calculate_score(resume_text, selected_designation):
    """
    Calculates the ATS match score based on keyword matching.
    Args:
        resume_text (str): The cleaned text of the resume.
        selected_designation (str): The chosen job designation.
    Returns:
        tuple: (percentage_score, matched_keywords, missing_keywords)
    """
    target_keywords = designation_keywords.get(selected_designation, [])
    total_keywords = len(target_keywords)

    if total_keywords == 0:
        return 0, [], []

    score = 0
    matched = set()
    missing = set(target_keywords)

    for keyword_raw in target_keywords:
        # Clean the keyword itself to match the cleaning applied to the resume text
        keyword_cleaned = clean_text(keyword_raw)
        
        # Check for presence of the cleaned keyword in the cleaned resume text
        if keyword_cleaned in resume_text:
            score += 1
            matched.add(keyword_raw) # Store the original keyword for display
            if keyword_raw in missing:
                missing.remove(keyword_raw)

    percentage = (score / total_keywords) * 100
    return percentage, sorted(list(matched)), sorted(list(missing))

# --- LLM Integration Function ---
def fetch_llm_suggestions(resume_text, selected_designation, missing_keywords):
    """
    Fetches LLM-generated suggestions for integrating missing keywords.
    Args:
        resume_text (str): The cleaned resume text.
        selected_designation (str): The selected job designation.
        missing_keywords (list): A list of keywords not found in the resume.
    Returns:
        str: LLM's suggestions as a bulleted string, or an error message.
    """
    if not resume_text or not missing_keywords:
        return "No specific suggestions needed, or no missing keywords to analyze."

    # Construct the prompt for the LLM
    prompt = f"""Given these missing keywords for a "{selected_designation}" role: {', '.join(missing_keywords)}.
    And considering the user's resume text (snippet for context, truncated for brevity: {resume_text[:500]}).
    Provide concise, actionable advice for the user to integrate these keywords into their resume to improve ATS match.
    Focus on specific phrasing and where to best place them (e.g., summary, experience bullet points, skills section).
    Format each suggestion as a single, short sentence or phrase, one per line. Do not include any leading asterisks, hyphens, numbers, or other bullet characters."""

    # Gemini API details
    # The API key will be provided by the Canvas environment at runtime if left as an empty string.
    api_key = "AIzaSyDYkNiExLyVvs0vczS_3KPvn3Kgy9EbAOA"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Could not generate specific LLM suggestions. The response format was unexpected."
    except requests.exceptions.RequestException as e:
        return f"Failed to get LLM suggestions: A network or API error occurred: {e}"
    except Exception as e:
        return f"Failed to get LLM suggestions: An unexpected error occurred: {e}"

# --- Streamlit UI Layout ---
st.set_page_config(page_title="ATS Resume Tracker", layout="centered", initial_sidebar_state="auto")

st.title("📄 ATS Resume Tracker")
st.markdown("Optimize your resume to pass Applicant Tracking Systems with ease.")

st.markdown("---")

# --- 2. Create the Dropdown Menu for Designation Selection ---
st.subheader("1. Select Your Target Job Designation:")
selected_designation = st.selectbox(
    "Choose a designation from the list below:",
    options=[""] + list(designation_keywords.keys()), # Add an empty option
    format_func=lambda x: "Choose a designation..." if x == "" else x,
    key="designation_select"
)

st.markdown("---")

# --- 3. Parse the Resume Data ---
st.subheader("2. Upload Your Resume:")
uploaded_file = st.file_uploader(
    "Upload your resume here (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    key="resume_uploader"
)

resume_text_content = ""
if uploaded_file is not None:
    # Display loading spinner
    with st.spinner(f"Processing {uploaded_file.type} file..."):
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == "pdf":
            resume_text_content = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            resume_text_content = extract_text_from_docx(uploaded_file)
        elif file_extension == "txt":
            # For .txt files, read directly
            resume_text_content = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

    if resume_text_content:
        st.success("Resume text extracted successfully!")
        # st.text(f"Extracted Text (first 500 chars):\n{clean_text(resume_text_content)[:500]}...") # For debugging
    else:
        st.warning("Could not extract text from the uploaded file. Please try another file.")

cleaned_resume_text = clean_text(resume_text_content)

st.markdown("---")

# --- 4. & 6. Score Calculation & Display Recommendations ---
st.subheader("3. Your ATS Report")

if selected_designation and uploaded_file and cleaned_resume_text:
    # Calculate score
    percentage_score, matched_keywords, missing_keywords = calculate_score(
        cleaned_resume_text, selected_designation
    )

    # Display Match Score
    st.markdown(f"**Match Score for '{selected_designation}':**")
    st.progress(percentage_score / 100)
    st.write(f"Your resume has an **{percentage_score:.2f}%** match with the required keywords for this designation.")

    st.markdown("---")

    # Display Matched Keywords
    if matched_keywords:
        st.markdown(f"#### ✅ Matched Keywords ({len(matched_keywords)} found):")
        with st.expander("Click to view matched keywords"):
            # Display keywords in a multi-column layout for better readability
            cols_per_row = 3
            cols = st.columns(cols_per_row)
            for i, keyword in enumerate(matched_keywords):
                cols[i % cols_per_row].write(f"- {keyword}")

    # Display Missing Keywords
    if missing_keywords:
        st.markdown(f"#### ⚠️ Missing Keywords ({len(missing_keywords)}):")
        st.warning("Consider integrating these highly relevant keywords to improve your resume's visibility to ATS:")
        # Display missing keywords in a multi-column layout
        cols_per_row = 3
        cols = st.columns(cols_per_row)
        for i, keyword in enumerate(missing_keywords):
            cols[i % cols_per_row].write(f"- **{keyword}**")
    else:
        st.success("🎉 Excellent! Your resume covers all the essential keywords for this designation.")

    st.markdown("---")

    # --- 5. Provide Suggestions for Improvement (LLM Powered) ---
    st.markdown("#### ✨ Smart Suggestions for Improvement:")
    if missing_keywords and cleaned_resume_text:
        with st.spinner("Generating personalized suggestions (this might take a moment)..."):
            llm_suggestions_text = fetch_llm_suggestions(cleaned_resume_text, selected_designation, missing_keywords)
            # Split the LLM response into lines and render as a list
            suggestion_lines = [line.strip() for line in llm_suggestions_text.split('\n') if line.strip()]
            if suggestion_lines:
                # Use Markdown to render a list, ensuring no leading characters from LLM
                st.markdown(
                    "<ul style='list-style-type: disc; padding-left: 20px;'>" +
                    "".join([f"<li style='margin-bottom: 8px;'>{line.strip()}</li>" for line in suggestion_lines]) +
                    "</ul>",
                    unsafe_allow_html=True
                )
            else:
                st.info("No specific LLM suggestions generated at this time. Please try again.")
    else:
        st.info("No specific keyword suggestions needed from the LLM, or please upload your resume.")

    st.markdown("---")

    # --- General ATS Tips ---
    st.markdown("#### 📝 General ATS Best Practices:")
    st.markdown("""
    <ul style='list-style-type: disc; padding-left: 20px;'>
        <li style='margin-bottom: 8px;'><b>Tailor Each Application:</b> Always customize your resume for every specific job description.</li>
        <li style='margin-bottom: 8px;'><b>Use Standard Formats:</b> Stick to clean, simple layouts. Avoid complex graphics, tables, or text boxes that ATS might struggle with.</li>
        <li style='margin-bottom: 8px;'><b>Quantify Achievements:</b> Use numbers and metrics to demonstrate impact (e.g., "Increased sales by 15%").</li>
        <li style='margin-bottom: 8px;'><b>Proofread Thoroughly:</b> Typos and grammatical errors are easily caught by ATS and recruiters.</li>
        <li style='margin-bottom: 8px;'><b>Save as PDF:</b> PDF generally preserves formatting best when uploaded, although some ATS prefer DOCX.</li>
    </ul>
    """, unsafe_allow_html=True)

elif selected_designation and not uploaded_file:
    st.info("Please upload your resume to get your ATS report for the selected designation!")
elif not selected_designation and not uploaded_file:
    st.info("Select a job designation and upload your resume to get started!")
else:
    st.info("Please upload your resume to get started!") # Fallback for edge cases

st.markdown("---")
st.caption("Developed as an ATS Tracker using Streamlit and Python.")

