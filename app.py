# ============================================================
# STREAMLIT WEB APP — Fake News Detection System
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ─────────────────────────────────────────
# CUSTOM CSS STYLING
# ─────────────────────────────────────────
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* Title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #aaa;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Result cards */
    .result-real {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border-left: 5px solid #52b788;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 20px;
    }
    .result-fake {
        background: linear-gradient(135deg, #4a1c1c, #7b2d2d);
        border-left: 5px solid #e63946;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 20px;
    }

    /* Info box */
    .info-box {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Text area label */
    label {
        color: #ddd !important;
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD OR TRAIN MODEL
# We cache the model so it doesn't retrain every time
# the user interacts with the app (Streamlit best practice)
# ─────────────────────────────────────────

@st.cache_resource
def load_or_train_model():
    """Load the saved model, or train a fresh one if not found."""
    model_path = "model/model.pkl"
    vec_path = "model/vectorizer.pkl"

    # If saved model exists, load it
    if os.path.exists(model_path) and os.path.exists(vec_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)
        accuracy = None  # We don't recompute accuracy here
        return model, vectorizer, accuracy

    # Otherwise, train from scratch
    df = pd.read_csv("data/news.csv")
    df['label_num'] = df['label'].map({'REAL': 1, 'FAKE': 0})

    X = df['text']
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Save model
    os.makedirs("model", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer, accuracy


# ─────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────

def predict(text, model, vectorizer):
    """Run prediction and return label + confidence."""
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100
    label = "REAL" if pred == 1 else "FAKE"
    return label, confidence, proba


# ─────────────────────────────────────────
# MAIN APP UI
# ─────────────────────────────────────────

# Title
st.markdown('<h1 class="main-title">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by TF-IDF + Logistic Regression · Built for AI/ML Beginners</p>', unsafe_allow_html=True)

st.markdown("---")

# Load model
with st.spinner("⚙️ Loading AI model..."):
    model, vectorizer, accuracy = load_or_train_model()

# Show accuracy badge
if accuracy:
    st.success(f"✅ Model trained! Accuracy on test data: **{accuracy:.1f}%**")
else:
    st.info("✅ Pre-trained model loaded successfully!")

# ─────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────

st.markdown("### 📰 Enter News Text Below")
user_text = st.text_area(
    label="Paste a news headline or article excerpt:",
    placeholder="e.g. Scientists discover a new treatment that reduces cancer risk by 40% in clinical trials...",
    height=160
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    detect_button = st.button("🔍 Detect Now", use_container_width=True)

# ─────────────────────────────────────────
# RESULT SECTION
# ─────────────────────────────────────────

if detect_button:
    if len(user_text.strip()) < 10:
        st.warning("⚠️ Please enter a longer news text (at least 10 characters).")
    else:
        label, confidence, proba = predict(user_text, model, vectorizer)

        if label == "REAL":
            st.markdown(f"""
            <div class="result-real">
                <h2>✅ This looks REAL</h2>
                <p style="font-size:1.1rem; color:#b7e4c7;">
                    The model classified this news as <strong>REAL</strong> 
                    with <strong>{confidence:.1f}% confidence</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fake">
                <h2>❌ This looks FAKE</h2>
                <p style="font-size:1.1rem; color:#ffb3b3;">
                    The model classified this news as <strong>FAKE</strong> 
                    with <strong>{confidence:.1f}% confidence</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown("#### 📊 Prediction Confidence")
        confidence_df = pd.DataFrame({
            "Category": ["FAKE 🚫", "REAL ✅"],
            "Confidence (%)": [round(proba[0] * 100, 1), round(proba[1] * 100, 1)]
        })
        st.bar_chart(confidence_df.set_index("Category"))

        st.caption("⚠️ This is an AI prediction trained on a small dataset. Always verify news from trusted sources.")

# ─────────────────────────────────────────
# HOW IT WORKS SECTION
# ─────────────────────────────────────────

st.markdown("---")
with st.expander("🧠 How does this work? (Click to learn!)"):
    st.markdown("""
    This system uses **Natural Language Processing (NLP)** and **Machine Learning**:

    **Step 1 — Data Collection 📂**  
    We have a CSV file with news articles labeled as REAL or FAKE.

    **Step 2 — TF-IDF Vectorization 🔢**  
    Text is converted to numbers using TF-IDF. Words like *"shocking"*, *"exposed"*, 
    *"aliens"* score high in FAKE news. Words like *"research"*, *"study"*, *"confirmed"* 
    score high in REAL news.

    **Step 3 — Logistic Regression 🤖**  
    The model learns which word patterns belong to REAL vs FAKE news during training.

    **Step 4 — Prediction 🔍**  
    For new text, it converts to TF-IDF and runs through the trained model to predict.

    **Accuracy** depends on the size and quality of the training dataset.  
    With real-world large datasets (like Kaggle's LIAR dataset), accuracy can exceed 90%!
    """)

# ─────────────────────────────────────────
# SAMPLE NEWS EXAMPLES
# ─────────────────────────────────────────

st.markdown("---")
st.markdown("### 💡 Try These Examples")

examples = {
    "🟢 Real News": "NASA confirms the James Webb Space Telescope has captured the deepest infrared image of the universe ever taken.",
    "🔴 Fake News": "SHOCKING: Scientists confirm 5G towers are mind control devices installed by secret government to track citizens.",
}

col_r, col_f = st.columns(2)
for (title, example), col in zip(examples.items(), [col_r, col_f]):
    with col:
        st.markdown(f"**{title}**")
        st.code(example, language=None)
        if st.button(f"Try this →", key=title):
            st.session_state["prefill"] = example

if "prefill" in st.session_state:
    st.info(f"📋 Copied! Paste this in the text box above:\n\n_{st.session_state['prefill']}_")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🎓 Built for B.Tech AI/ML Students · Python + scikit-learn + Streamlit"
    "</p>",
    unsafe_allow_html=True
)
