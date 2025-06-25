import streamlit as st
import json
import pandas as pd
import pickle
import tempfile
import os
from pdf_processor import extract_structured_data  
import numpy as np
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from run_rag_from_json import generate_rag_explanation
import google.generativeai as genai 

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# ✅ Load model and feature names from dictionary
with open("logistic regression (1).pkl", "rb") as f:
    package = pickle.load(f)
    model = package["model"]
    feature_names = package["feature_names"]

# ✅ Load SHAP background dataset
try:
    with open("background_data.pkl", "rb") as f:
        background = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load background data. SHAP values may not be accurate.")
    background = None

# 🧠 App Title
st.title("🧪 Semen Analysis Fertility Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload your semen analysis PDF", type=["pdf", "jpg", "png"])

if uploaded_file:
    file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info("🔍 Extracting data from uploaded file...")

    try:
        # Extract structured data
        result = extract_structured_data(tmp_file_path)
        extracted_data = result.model_dump()

        # Save extracted JSON
        json_path = tmp_file_path.replace(".pdf", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        st.success("✅ Data extracted successfully!")

        # Extract values
        volume = extracted_data['semen_analysis']['volume']['value']
        concentration = extracted_data['semen_analysis']['concentration']['value']
        motility = extracted_data['semen_analysis']['motility']['value']

        # Create input DataFrame using correct feature names
        input_data = pd.DataFrame([{
            feature_names[0]: volume,
            feature_names[1]: concentration,
            feature_names[2]: motility
        }])

        # Predict fertility probability
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("🎯 Fertility Score")
        st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{probability*100:.2f}%</h1>", unsafe_allow_html=True)

        # Sidebar interpretation
        st.sidebar.title("📊 Interpretation Guide")
        st.sidebar.subheader("0-40%: Low Fertility Probability")
        st.sidebar.write("Unfavorable indicators. Recommend follow-up consultation.")
        st.sidebar.subheader("40-70%: Borderline")
        st.sidebar.write("Results are inconclusive; consider retesting.")
        st.sidebar.subheader("70-100%: High Fertility Probability")
        st.sidebar.write("Strong indicators of fertility.")

        # Display table of values
        st.table(pd.DataFrame({
            "Label Name": ["Volume", "Concentration", "Motility"],
            "Value": [volume, concentration, motility]
        }).set_index("Label Name"))

        # 🔍 SHAP Explanation
        st.subheader("🔎 Feature Importance Analysis (SHAP)")
        if background is not None:
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(input_data)
            base_value = explainer.expected_value

            feature_impact = pd.DataFrame({
                "Feature": feature_names,
                "Value": input_data.iloc[0].values,
                "SHAP_Impact": shap_values[0]
            }).sort_values(by="SHAP_Impact")

            feature_impact["Abs_Impact"] = np.abs(feature_impact["SHAP_Impact"])
            colors = ['#ff4d4d' if x < 0 else '#28a745' for x in feature_impact["SHAP_Impact"]]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(feature_impact["Feature"], feature_impact["SHAP_Impact"], color=colors, edgecolor='black')

            for bar, impact in zip(bars, feature_impact["SHAP_Impact"]):
                x_offset = 0.08 if impact < 0 else 0
                ax.text(bar.get_width() + x_offset, bar.get_y() + bar.get_height() / 2,
                        f'{impact:+.2f}', va='center', ha='left', fontsize=10, fontweight='bold')

            ax.axvline(0, color='black', linewidth=1)
            ax.set_title("Feature Contribution to Fertility Prediction")
            ax.set_xlabel("SHAP Impact")
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 💬 Text Explanation
            def get_impact_strength(abs_impact):
                if abs_impact > 0.15:
                    return "Very Strong"
                elif abs_impact > 0.10:
                    return "Strong"
                elif abs_impact > 0.05:
                    return "Moderate"
                else:
                    return "Mild"

            st.subheader("📋 DETAILED FEATURE ANALYSIS")
            st.markdown("---")

            positive_factors = feature_impact[feature_impact['SHAP_Impact'] >= 0]
            negative_factors = feature_impact[feature_impact['SHAP_Impact'] < 0]
            if not positive_factors.empty:
                st.markdown("✅ **POSITIVE INFLUENCES (Supporting Fertility)**")
                for _, row in positive_factors.iterrows():
                    st.markdown(f"💪 **{row['Feature']}** = {row['Value']:.2f}")
                    st.markdown(f"→ {get_impact_strength(row['Abs_Impact'])} **positive** impact (**+{row['SHAP_Impact']:.2f}**)")

            if not negative_factors.empty:
                st.markdown("⚠️ **NEGATIVE INFLUENCES (Areas for Improvement)**")
                for _, row in negative_factors.iterrows():
                    st.markdown(f"🎯 **{row['Feature']}** = {row['Value']:.2f}")
                    st.markdown(f"→ {get_impact_strength(row['Abs_Impact'])} **negative** impact (**{row['SHAP_Impact']:.2f}**)  \n💡 Consider improving this parameter if possible.")

            # ✅ Save structured JSON for RAG
            shap_json = {
                "fertility_score": round(probability * 100, 2),
                "features": {}
            }

            for feature, value in zip(feature_names, input_data.iloc[0]):
                shap_json["features"][feature] = {
                    "value": float(value),
                    "impact": float(shap_values[0][feature_names.index(feature)])
                }

            with open("shap_inputs.json", "w") as f:
                json.dump(shap_json, f, indent=2)

            st.success("✅ SHAP explanation saved for RAG")
            st.json(shap_json)
            # ✅ Generate RAG Explanation
       

            try:
                st.subheader("📖 AI Fertility Explanation")
                with st.spinner("Generating explanation..."):
                    explanation = generate_rag_explanation()
                    st.markdown(explanation)
                    with open("rag_output.txt", "w") as f:
                        f.write(explanation)
                st.success("✅ Explanation complete and saved!")
            except Exception as e:
                st.error(f"❌ Failed to generate explanation: {e}")

        else:
            st.warning("⚠️ SHAP explanation not available due to missing background data.")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")

    os.remove(tmp_file_path)

# Footer
st.caption("🧠 Disclaimer: This tool is for educational use only. Not a substitute for professional diagnosis.")
st.markdown("[Terms of Service](https://herafertility.co/policies/terms-of-service) | [Privacy Policy](https://herafertility.co/policies/privacy-policy)")

# import os
# import json
# import tempfile
# import pickle
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# import streamlit as st
# from dotenv import load_dotenv

# from pdf_processor import extract_structured_data

# # LangChain + FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

# # ✅ Environment setup
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

# # ✅ Load model
# with open("logistic regression (1).pkl", "rb") as f:
#     package = pickle.load(f)
#     model = package["model"]
#     feature_names = package["feature_names"]

# # ✅ Load SHAP background data
# try:
#     with open("background_data.pkl", "rb") as f:
#         background = pickle.load(f)
# except:
#     background = None
#     st.warning("⚠️ SHAP background data not loaded.")

# # ✅ LangChain LLM setup
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
# vectorstore = FAISS.load_local("faiss_index_both", embedding_model, allow_dangerous_deserialization=True)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4000)

# # ✅ Prompt Template
# def categorize_score(score):
#     if score < 20:
#         return "very low", "IVF with ICSI is highly likely. This means doctors use special methods to help with pregnancy."
#     elif 20 <= score < 40:
#         return "low", "IVF (with or without ICSI) is recommended, depending on age and egg quality."
#     elif 40 <= score < 60:
#         return "moderate", "Try IUI, and consider vitamins/lifestyle improvements."
#     elif 60 <= score < 80:
#         return "good", "Try naturally for 6–12 months or IUI if time-sensitive."
#     else:
#         return "high", "Timed intercourse and ovulation tracking is suggested."

# def create_prompt_template(score):
#     category, treatment = categorize_score(score)
#     return PromptTemplate(
#         input_variables=["context", "question"],
#         template=f"""
# A machine learning model predicted a {category} fertility score of {{score:.2f}}%, representing the probability of achieving successful natural pregnancy within 12 months.

# {{question}}

# Strictly follow this structure:
# 1. Introduction: Restate the {category} fertility score ({{score:.2f}}%) and its probability of natural pregnancy within 12 months.
# 2. Numbered subheadings for each factor (e.g., "1. Motility", "2. Volume", "3. Concentration"):
#    - State the SHAP value and quantify its impact (strong, moderate, weak) on the fertility probability.
#    - Explain how the factor's value and impact drive the {{score:.2f}}% score.
#    - Suggest personalized interventions to improve the fertility probability, tailored to the factor's impact.
#    - Cite claims inline: (Title by Author).
# 3. Summary: Prioritize factors with stronger negative impacts (if any), recommend steps to maximize the fertility probability, and include the following treatment recommendation: {treatment}

# Use the medical research below. Do not deviate from this structure.

# Context:
# {{context}}
# """
#     )

# # ✅ Streamlit App UI
# st.set_page_config(page_title="Fertility Predictor + RAG", layout="centered")
# st.title("🧪 Fertility Score + AI Explanation")

# uploaded_file = st.file_uploader("Upload Semen Analysis (PDF or Image)", type=["pdf", "jpg", "png"])

# if uploaded_file:
#     file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name

#     try:
#         # ✅ Step 1: Extract Semen Analysis
#         extracted = extract_structured_data(tmp_file_path).model_dump()
#         volume = extracted['semen_analysis']['volume']['value']
#         concentration = extracted['semen_analysis']['concentration']['value']
#         motility = extracted['semen_analysis']['motility']['value']

#         # ✅ Step 2: Predict Fertility Score
#         input_data = pd.DataFrame([{
#             feature_names[0]: volume,
#             feature_names[1]: concentration,
#             feature_names[2]: motility
#         }])
#         score = model.predict_proba(input_data)[0][1] * 100
#         st.subheader("🎯 Predicted Fertility Score")
#         st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{score:.2f}%</h1>", unsafe_allow_html=True)

#         # ✅ Step 3: Compute SHAP
#         explainer = shap.LinearExplainer(model, background)
#         shap_values = explainer.shap_values(input_data)
#         shap_values = shap_values.ravel()  # ✅ Fix for saving float-compatible values

#         # ✅ Step 4: SHAP bar plot
#         st.subheader("🔍 SHAP Feature Contributions")
#         fig, ax = plt.subplots()
#         shap.summary_plot([shap_values], input_data, feature_names=feature_names, plot_type="bar", show=False)
#         st.pyplot(fig)

#         # ✅ Step 5: Save SHAP JSON
#         shap_json = {
#             "fertility_score": round(score, 2),
#             "features": {
#                 feature_names[i]: {
#                     "value": float(input_data.iloc[0][i]),
#                     "impact": float(shap_values[i])
#                 } for i in range(len(feature_names))
#             }
#         }

#         with open("shap_inputs.json", "w") as f:
#             json.dump(shap_json, f, indent=2)

#         # ✅ Step 6: Format SHAP input for RAG
#         shap_formatted = f"Fertility score: {score:.2f}%\n\nThis score represents the model's prediction of the chance of having a natural pregnancy within 12 months.\n\n✅ Positive Factors:\n"
#         for k, v in shap_json["features"].items():
#             if v["impact"] > 0:
#                 shap_formatted += f"- {k}: {v['value']} → SHAP Impact: +{v['impact']:.3f} (positive)\n"
#         shap_formatted += "\n⚠️ Negative Factors:\n"
#         for k, v in shap_json["features"].items():
#             if v["impact"] < 0:
#                 shap_formatted += f"- {k}: {v['value']} → SHAP Impact: {v['impact']:.3f} (negative)\n"
#         shap_formatted += "\nPlease explain the fertility score and contributing factors using SHAP. Follow the exact structure from the prompt."

#         # ✅ Step 7: Generate RAG Explanation
#         st.subheader("📖 AI-Generated Explanation (RAG)")
#         rag_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff",
#             chain_type_kwargs={
#                 "prompt": create_prompt_template(score).partial(score=score),
#                 "document_variable_name": "context"
#             },
#             input_key="query"
#         )
#         with st.spinner("🔍 Generating explanation using medical literature..."):
#             response = rag_chain.invoke({"query": shap_formatted})
#             st.markdown(response["result"])

#         # ✅ Final Display
#         st.success("✅ Done! All stages completed.")
#         st.json(shap_json)

#     except Exception as e:
#         st.error(f"❌ Error: {e}")
#     finally:
#         os.remove(tmp_file_path)

# st.caption("📘 Educational tool only. Not a medical diagnosis.")
