import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Just grabbing the model and tokenizer from disk (local fine-tuned model)
MODEL_PATH = "model/distilbert_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Basic prediction wrapper
def classify_message(msg_text):
    # Tokenize the input, pad it just in case, and make sure it fits the model
    encoded = tokenizer(msg_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():  # Don't need gradients for inference
        result = model(**encoded).logits
    pred_idx = torch.argmax(result, dim=1).item()
    conf_score = torch.nn.functional.softmax(result, dim=1)[0][pred_idx].item()
    label = "Spam" if pred_idx == 1 else "Ham"
    return (label, round(conf_score * 100, 2))  # Convert to percentage just for UI clarity

# --- Streamlit UI bits ---
st.set_page_config(page_title="SMS Spam Detector", page_icon="📱")
st.title("📱 SMS Spam Detection")

# Little UI nicety – tab layout
tab_live, tab_adv = st.tabs(["🔍 Live SMS Detection", "📊 Adversarial Testing Results"])

# --- First Tab: Interactive Prediction ---
with tab_live:
    st.subheader("Type a message below to classify it:")

    user_input = st.text_area("✉️ Enter SMS message here:")

    if st.button("🚀 Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")  # Handles empty input gracefully
        else:
            label, confidence = classify_message(user_input)
            st.success(f"**Prediction:** {label} ({confidence}% confident)")

# --- Second Tab: Visualizing Adversarial Results ---
with tab_adv:
    st.subheader("Adversarial Attack Results")

    try:
        # NOTE: Make sure this CSV is there – should be generated from adversarial_train.py
        adv_data = pd.read_csv("adversarial_results.csv")

        st.dataframe(adv_data)

        # Count how many predictions changed vs stayed the same
        changed = adv_data[adv_data['Changed'] == 'Yes'].shape[0]
        same = adv_data[adv_data['Changed'] == 'No'].shape[0]

        st.markdown(f"""
        - ✅ **Unchanged Predictions**: {same}  
        - ❌ **Flipped Predictions**: {changed}
        """)

        # Simple bar chart – nothing fancy
        st.markdown("### 📉 Bar Chart")
        fig1, ax1 = plt.subplots()
        ax1.bar(['Unchanged', 'Flipped'], [same, changed], color=['green', 'red'])
        ax1.set_ylabel("Count")
        ax1.set_title("Prediction Change After Adversarial Attack")
        st.pyplot(fig1)

        # Pie chart – good for quick glance
        st.markdown("### 🥧 Pie Chart")
        fig2, ax2 = plt.subplots()
        ax2.pie([same, changed],
                labels=['Unchanged', 'Flipped'],
                colors=['lightgreen', 'lightcoral'],
                autopct='%1.1f%%',
                startangle=90,
                shadow=True)
        ax2.set_title("Prediction Stability")
        st.pyplot(fig2)

        # Bonus: manually inspect any specific example
        st.markdown("### 🔬 Inspect an Example")
        picked_row = st.selectbox("Select a row to compare:", adv_data.index)
        st.write("**Original Text:**", adv_data.loc[picked_row, 'Original Text'])
        st.write("**Original Prediction:**", adv_data.loc[picked_row, 'Original Prediction'])
        st.write("**Adversarial Text:**", adv_data.loc[picked_row, 'Adversarial Text'])
        st.write("**Adversarial Prediction:**", adv_data.loc[picked_row, 'Adversarial Prediction'])

    except FileNotFoundError:
        # If the file isn’t there, probably haven’t run the attack script yet
        st.warning("⚠️ 'adversarial_results.csv' not found. Please run adversarial_train.py first.")
