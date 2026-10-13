
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")
st.title("📚 RAG Document Question Answering")
st.write("Ask questions about your Employee Handbook.")

@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("document.index")

    with open("chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    return model, index, chunks

try:
    model, index, chunks = load_resources()
except Exception as e:
    st.error(f"Could not load the index or chunks: {e}")
    st.stop()

question = st.text_input(
    "Enter your question",
    placeholder="How many annual leave days do employees receive?"
)

if st.button("Search Document"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        query_vector = model.encode(
            [question],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = index.search(query_vector, k=3)

        st.subheader("Relevant passages")

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                continue

            st.markdown(f"### Result {rank}")
            st.caption(f"Similarity score: {score:.3f}")
            st.write(chunks[idx])
            st.divider()