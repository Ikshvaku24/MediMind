import streamlit as st
import pickle
import torch
from transformers import BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForSequenceClassification
import faiss
import pandas as pd
import numpy as np
import contractions
import re
import os

# Load the trained disease prediction model
with open('./models/tfidf_trigrams_model.pkl', 'rb') as model_file:
    tfidf_trigrams_model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('./vectorizers/tfidf_vectorizer3.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load BioBERT tokenizer and model
biobert_tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
biobert_model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.2",
    num_labels=1
)
biobert_model.eval()  # Set model to evaluation mode

# Load BioBERT weights from checkpoint
try:
    checkpoint_path = "./checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)
    biobert_model.load_state_dict(checkpoint['model_state_dict'])
    print("BioBERT model weights loaded successfully.")
except Exception as e:
    print(f"Error loading BioBERT checkpoint: {e}")

# Load GPT-2 tokenizer and model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token  # Set pad token to eos_token

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")  # Load base GPT-2 model
gpt2_model.eval()  # Set model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
biobert_model = biobert_model.to(device)
gpt2_model = gpt2_model.to(device)

# Load GPT-2 weights from checkpoint
try:
    gpt2_checkpoint_path = "./finalmodel2.pth"
    gpt2_checkpoint = torch.load(gpt2_checkpoint_path)
    gpt2_model.load_state_dict(gpt2_checkpoint)
    print("GPT-2 model weights loaded successfully.")
except Exception as e:
    print(f"Error loading GPT2 checkpoint: {e}")

# Load FAISS index
answer_index = faiss.read_index('./answer_index.faiss')

# Load training data for GPT-2 preparation
train_gpt_data = pd.read_pickle('./train_gpt_input_data.pkl')
validation_gpt_data = pd.read_pickle('./validation_gpt_input_data.pkl')

# Function for expanding contractions
def decontractions(phrase):
    phrase = contractions.fix(phrase).lower()  # Expand contractions
    phrase = re.sub(r"[^\w\s]", "", phrase)  # Remove punctuation
    return phrase

# Function to prepare GPT input
def prepare_gpt_input(question, question_embedding, tokenizer, top_k=20, similarity_threshold=0.8):
    # Normalize and search FAISS index
    faiss.normalize_L2(np.array([question_embedding]).astype('float32'))
    distances, indices = answer_index.search(np.array([question_embedding]), top_k)
    
    # Filter based on similarity threshold
    similar_qas = [
        (train_gpt_data.iloc[idx]['question'], train_gpt_data.iloc[idx]['answer'])
        for idx, dist in zip(indices[0], distances[0])
        if dist >= similarity_threshold
    ]
    
    # Generate context string
    seen_questions = set()
    context_string = f"`QUESTION: {question} `ANSWER:"

    # Limit number of questions in context
    max_questions_in_context = 5  # Adjust this number based on how much context you want to provide

    for q, a in similar_qas:
        if q not in seen_questions:
            pair = f"`QUESTION: {q} `ANSWER: {a} "
            pair = pair.replace("\n","")
            if len(tokenizer.encode(pair + context_string)) < 1024:
                context_string = pair + context_string
                seen_questions.add(q)
            if len(seen_questions) >= max_questions_in_context:
                break  # Stop adding more questions after max_questions_in_context

    # Tokenize and truncate
    tokenized_input = tokenizer.encode(context_string, truncation=True, max_length=1024)
    return tokenized_input[-1024:]

# Function to extract embeddings from BioBERT
def extract_embeddings(text, model, tokenizer, max_length=512):
    # Tokenize the input text
    inputs = tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to('cuda')
    attention_mask = inputs['attention_mask'].to('cuda')
    
    # Get embeddings from hidden states of the last layer
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Use the last hidden state as sentence embeddings, averaging across tokens
        last_hidden_state = outputs.hidden_states[-1]  # Get last layer
        embeddings = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    return embeddings

# Function to extract and combine unique answers
def extract_and_combine_unique_answers(decoded_output):
    parts = decoded_output.split("`QUESTION: ")
    seen_answers = set()
    unique_answers = []
    for part in parts[1:]:
        if "`ANSWER:" in part:
            answer_start = part.split("`ANSWER: ")[1].strip()
            if answer_start not in seen_answers:
                seen_answers.add(answer_start)
                unique_answers.append(answer_start)
    return " ".join(unique_answers)

# Function to generate answers using GPT-2
def generate_answer(question, answer_length=50):
    preprocessed_question = decontractions(question)
    question_embedding = extract_embeddings(preprocessed_question, biobert_model, biobert_tokenizer)
    gpt_input = prepare_gpt_input(preprocessed_question, question_embedding, gpt2_tokenizer)
    
    if len(gpt_input) > (1024 - answer_length):
        gpt_input = gpt_input[-(1024 - answer_length):]
    
    input_ids = torch.tensor([gpt_input]).to('cuda')
    gpt_output = gpt2_model.generate(input_ids=input_ids, max_length=1024, temperature=0.9, top_k=50, top_p=0.9)
    decoded_output = gpt2_tokenizer.decode(gpt_output[0])
    return decoded_output

# Final pipeline for generating answers
def final_pipeline(question):
    generated_answer = generate_answer(question)
    return extract_and_combine_unique_answers(generated_answer)

# Streamlit interface
st.title("Disease Prediction System with Answer Generation")
st.write("Enter your symptoms to predict the disease and get further information.")

# Disease prediction block
st.subheader("1. Predict Disease")
symptoms = st.text_area("Describe your symptoms in detail (e.g., fever, cough, fatigue)")

if st.button("Predict Disease"):
    if symptoms.strip():  # Ensure non-empty input
        try:
            # Preprocess and transform user input
            text_transformed = tfidf_vectorizer.transform([symptoms])
            
            # Make disease prediction
            disease_prediction = tfidf_trigrams_model.predict(text_transformed)[0]
            st.success(f"The predicted disease is: **{disease_prediction}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter your symptoms before predicting.")

# Get Further Information block
st.subheader("2. Get Further Information")
information_request = st.text_area("Ask for additional information (e.g., Treatment for disease, Symptoms of disease)")

if st.button("Get Information"):
    if information_request.strip():  # Ensure non-empty input
        try:
            # Generate answer using final pipeline
            generated_answer = final_pipeline(information_request)
            st.write(f"Additional information: {generated_answer}")
        
        except Exception as e:
            st.error(f"An error occurred while generating information: {e}")
    else:
        st.warning("Please enter your query before requesting information.")