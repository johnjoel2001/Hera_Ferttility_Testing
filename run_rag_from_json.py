import os
import json
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ✅ Define function
def generate_rag_explanation(json_path="shap_inputs.json", faiss_path="faiss_index_both"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Load from .env

    # Load SHAP JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    score = data["fertility_score"]
    features = data["features"]

    # Format query
    shap_formatted = f"Fertility score: {score:.2f}%\n\nThis score represents the model's prediction of the chance of having a natural pregnancy within 12 months.\n\n✅ Positive Factors:\n"
    for k, v in features.items():
        if v["impact"] > 0:
            shap_formatted += f"- {k}: {v['value']} → SHAP Impact: +{v['impact']:.3f} (positive)\n"
    shap_formatted += "\n⚠️ Negative Factors:\n"
    for k, v in features.items():
        if v["impact"] < 0:
            shap_formatted += f"- {k}: {v['value']} → SHAP Impact: {v['impact']:.3f} (negative)\n"
    shap_formatted += "\nPlease explain the fertility score and contributing factors using SHAP. Follow the exact structure from the prompt."

    # Categorize fertility
    def categorize_score(score):
        if score < 20:
            return "very low", "IVF with ICSI is highly likely. This means doctors use special methods to help with pregnancy."
        elif 20 <= score < 40:
            return "low", "IVF (with or without ICSI) is recommended, depending on age and egg quality."
        elif 40 <= score < 60:
            return "moderate", "Try IUI, and consider vitamins/lifestyle improvements."
        elif 60 <= score < 80:
            return "good", "Try naturally for 6–12 months or IUI if time-sensitive."
        else:
            return "high", "Timed intercourse and ovulation tracking is suggested."

    category, treatment = categorize_score(score)

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
A machine learning model predicted a **{category} fertility score of {{score:.2f}}%**, representing the probability of achieving successful natural pregnancy within 12 months.

{{question}}

Strictly follow this structure:

---

**1. Introduction**
- Restate the **{category} fertility score** (**{{score:.2f}}%**) and its probability of natural pregnancy within 12 months.

---

**2. Numbered subheadings for each factor (e.g., "1. Motility", "2. Volume", "3. Concentration"):**
- **State the SHAP value** and quantify its impact (**strong**, **moderate**, **weak**) on the fertility probability.
- **Explain** how the factor's value and impact drive the **{{score:.2f}}%** score.
- **Suggest personalized interventions** to improve the fertility probability, tailored to the factor's impact.
- **Cite claims inline**: (Title by Author).

Repeat this structure for each factor.

---

**3. Summary**
- **Prioritize** factors with stronger negative impacts (if any).
- **Recommend** steps to maximize the fertility probability.
- **Include** the following treatment recommendation: **{treatment}**

---

Use the medical research below. Do not deviate from this structure.  
Please do put everything in **bulletin**. I want neatly formatted. **Headers in bold, spacing**, and the **summary should be in bulletin** as well.

**Context**:  
{{context}}
"""

    )

    # Load vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(faiss_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4000)

    # Build RAG pipeline
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt.partial(score=score),
            "document_variable_name": "context"
        },
        input_key="query"
    )

    response = rag_chain.invoke({"query": shap_formatted})
    return response["result"]



