import streamlit as st
import nltk
nltk.data.path.append("C:/Users/Joydip/nltk_data")
nltk.download('punkt_tab', quiet=True)  # Ensure tokenizer data is downloaded

from tensorflow.keras.models import load_model
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def Load_Model():
    lstm_model = load_model('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/spam_classifier_Blstm.h5')
    with open('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    classical_ml = pickle.load(open('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/bernoulli_naive_bayes_model.sav', 'rb'))
    fcnn = load_model('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/fcnn_combined_model_etc.h5')
    with open('C:/Users/Joydip/OneDrive/Desktop/College Assignmets_Material/INSTRU/tfidf_vectorizer.pkl', 'rb') as handle:
        tfidf = pickle.load(handle)
    return lstm_model, classical_ml, fcnn, tokenizer, tfidf

def preprocess_input(ip_text):
    """Preprocess the SMS text for TF-IDF model."""
    import string
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # Ensure stopwords are available
    nltk.download('stopwords', quiet=True)
    ps = PorterStemmer()

    # Use the input text for processing
    text = ip_text.lower()
    text = nltk.word_tokenize(text)

    y = [word for word in text if word.isalnum()]
    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

def predict_sms(sms):
    """Predict whether an SMS is spam or ham."""
    lstm_model, classical_ml, fcnn, tokenizer, tfidf = Load_Model()

    # Preprocess and transform the text for the LSTM model
    sequence = tokenizer.texts_to_sequences([sms])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    lstm_prob = lstm_model.predict(padded_sequence)[0, 0]

    # Preprocess and transform the text for the classical model
    transformed_sms = preprocess_input(sms)
    tfidf_features = tfidf.transform([transformed_sms]).toarray()
    classical_prob = classical_ml.predict_proba(tfidf_features)[:, 1][0]

    # Combine predictions and pass them through the FCNN
    combined_features = np.hstack(([[lstm_prob]], [[classical_prob]]))
    final_prob = fcnn.predict(combined_features)[0, 0]
    prediction = "Spam" if final_prob > 0.5 else "Ham"
    
    return {"LSTM Probability": lstm_prob,
            "Classical Probability": classical_prob,
            "Final Probability": final_prob,
            "Prediction": prediction}

st.title("SMS Spam-Ham Classifier")

user_input = st.text_input("Input Text:")

if st.button("Predict"):
    if user_input:
        prediction = predict_sms(user_input)
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please enter valid input.")
