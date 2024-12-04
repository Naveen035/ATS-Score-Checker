import streamlit as st
import fitz  
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import difflib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

nltk.download('stopwords')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Preprocessing function to clean the extracted text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  #non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text).strip()  # extra spaces
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # stopwords
    return text

# Function for partial matching (with stemming and similarity)
def match_keywords(text, keywords):
    score = 0
    keyword_matches = 0
    stemmer = PorterStemmer()

    # Stem keywords for better match (e.g., "developing" -> "develop")
    stemmed_keywords = [stemmer.stem(keyword) for keyword in keywords]
    words_in_text = text.split()
    stemmed_words_in_text = [stemmer.stem(word) for word in words_in_text]
    for keyword in stemmed_keywords:
        for word in stemmed_words_in_text:
            if difflib.SequenceMatcher(None, keyword, word).ratio() > 0.7:  # Threshold for similarity
                keyword_matches += 1
                break 
    
    score = (keyword_matches / len(keywords)) * 100  
    return score

# Build the RNN model 
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=input_shape))
    model.add(SimpleRNN(128, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def prepare_data(pdf_file):
    pdf_text = extract_text_from_pdf(pdf_file)
    texts = [pdf_text]
    labels = [1]  # Dummy label for now

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    return padded_sequences, np.array(labels), tokenizer

# Training the RNN model
def train_rnn_model(pdf_file):
    texts, labels, tokenizer = prepare_data(pdf_file)
    model = build_rnn_model(texts.shape[1])
    model.fit(texts, labels, epochs=5, batch_size=2)
    return model, tokenizer

# Evaluate ATS score based on keyword matching and RNN output
def evaluate_resume(text, keywords, model, tokenizer):
    preprocessed_text = preprocess_text(text)
    keyword_score = match_keywords(preprocessed_text, keywords)

    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    rnn_score = model.predict(padded_sequence)[0][0]
    final_score = min(100, (keyword_score * 0.8) + (rnn_score * 50)) 
    return final_score

def ats_score_checker(pdf_file, keywords, model, tokenizer):
    text = extract_text_from_pdf(pdf_file)
    score = evaluate_resume(text, keywords, model, tokenizer)
    return score

def main():
    st.title("ATS Score Checker 📝")
    st.write("Please upload your Resume Pdf file for checking the ATS score of Your Resume")
    
    uploaded_file = st.file_uploader("Upload your Resume (PDF format)", type="pdf")
    
    if uploaded_file:
        pdf_file = "temp_resume.pdf"
        with open(pdf_file, "wb") as f:
            f.write(uploaded_file.read())
        model, tokenizer = train_rnn_model(pdf_file)

        # List of relevant keywords for Data Scientist and Data Analyst resumes
        keywords = [
            "data analysis", "machine learning", "SQL", "statistical analysis", "python programming", 
    "data visualization", "business intelligence", "predictive modeling", 
    "project management", "communication skills", "adaptability", "problem-solving", 
    "leadership", "time management", "collaboration", "critical thinking",
    "education", "b.tech", "team leadership", "project coordination", "design", "develop", "eda",
    "analyzed", "maintained", "data", "model data", "visualizations", "etl", "powerpivot", "sas",
    "business intelligence", "trends", "optimization", "algorithms", "classification", "prediction",
    "feature engineering", "K-means", "clustering", "transform", "graphs", "charts", "techniques",
    "preprocessing", "predictor", "metrics", "sentiment", "accuracy", "proficiency","JavaScript",
    "HTML", "CSS", "Apache Spark", "Hadoop", "Kafka", 
    "Hive", "Pig", "Flink", "Dask", "NoSQL", "MongoDB", "Cassandra", "HBase", "ETL pipelines", 
    "Data cleaning", "Data wrangling", "Data migration", "Neural networks", "Deep learning", 
    "Reinforcement learning", "Support vector machines", "Random forests", "Decision trees", 
    "Natural language processing", "Image recognition", "Computer vision", "Bayesian inference", 
    "Hypothesis testing", "Regression analysis", "Time series analysis", "A/B testing", 
    "Monte Carlo simulations", "Statistical modeling", "Descriptive statistics", "Inferential statistics", 
    "Tableau", "Power BI", "Google Data Studio", "Matplotlib", "Seaborn", 
    "Data dashboards", "Business reporting", "Interactive charts", "Azure", "Google Cloud Platform", 
    "Google BigQuery", "Databricks", "Reinforcement learning", "Feature selection", 
    "Hyperparameter tuning", "Dimensionality reduction", "PCA", "t-SNE", "Model deployment", "Flask", 
    "Model evaluation metrics","ROC-AUC", "F1 score","Cross-functional teams", "Data-driven decision-making", 
    "Agile methodologies", "Scrum", "Sprint planning", "Mentorship", "Training", "Documentation", 
    "Algorithm optimization", "Data-driven insights", "Model validation", "SQL queries", 
    "Database management", "Time series forecasting", "Anomaly detection", "Risk analysis", 
    "Data integrity", "Data security", "Data governance"
        ]

        keywords = [keyword.lower() for keyword in keywords]

        ats_score = ats_score_checker(pdf_file, keywords, model, tokenizer)
        st.subheader(f"ATS Score: {ats_score:.2f} 📈")
        st.subheader("Extracted Text Preview 👀:")
        st.text(extract_text_from_pdf(pdf_file)[:10000]) 
        st.text("Note : This ATS Resume Checker is only for the Data Analyst and Data Scientiest role")

if __name__ == "__main__":
    main()
