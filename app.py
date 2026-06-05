import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load AI Model
st.set_page_config(
    page_title="Ushasi's Resume Detector",
    page_icon="🤖"
)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read Resume
with open("resume.txt", "r", encoding="utf-8") as file:
    resume = file.read()

# Read Job Description
with open("job_description.txt", "r", encoding="utf-8") as file:
    job_desc = file.read()

# Convert text into embeddings
resume_embedding = model.encode([resume])
job_embedding = model.encode([job_desc])

# Calculate similarity
similarity = cosine_similarity(resume_embedding, job_embedding)

score = round(similarity[0][0] * 100, 2)

# UI
st.title("AI Resume Screening System")

st.subheader("Resume")
st.write(resume)

st.subheader("Job Description")
st.write(job_desc)

st.subheader("Matching Score")
st.success(f"Resume matches the job description by {score}%")

# Decision
if score > 75:
    st.write("Strong Match")
elif score > 50:
    st.write("Moderate Match")
else:
    st.write("Weak Match")