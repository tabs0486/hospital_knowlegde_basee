import os
import json
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="Hospital Policy Assistant",
    page_icon="🏥",
    layout="centered"
)

FAISS_DIR = "faiss_index"
INDEX_FILE = os.path.join(FAISS_DIR, "hospital_policy.index")
METADATA_FILE = os.path.join(FAISS_DIR, "metadata.json")

MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

st.markdown(
    """
    <style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
    }
    .title {
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<h1 class="title">🏥 Hospital Policy Assistant</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Ask questions about hospital policies and procedures.</p>',
    unsafe_allow_html=True
)

# Read Groq API key from Streamlit Secrets.
# The key is never displayed in the application.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_faiss_index():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )
    return faiss.read_index(INDEX_FILE)


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    embedding_model = load_embedding_model()
    index = load_faiss_index()
    metadata = load_metadata()
except Exception as e:
    st.error(f"Unable to load the knowledge base: {e}")
    st.stop()


def retrieve_chunks(question, top_k=TOP_K):
    """Convert the question to an embedding and search FAISS."""

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx].copy()
        item["similarity"] = float(score)
        results.append(item)

    return results


def build_context(results):
    """Prepare retrieved chunks for the Groq model."""

    context_parts = []

    for number, item in enumerate(results, start=1):
        context_parts.append(
            f"""
SOURCE {number}

Department:
{item.get("department", "Unknown")}

Document:
{item.get("source", "Unknown")}

Page:
{item.get("page", "Unknown")}

Content:
{item.get("text", "")}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, results):
    """Send retrieved policy context to Groq."""

    context = build_context(results)

    system_prompt = """
You are a Hospital Policy Knowledge Base Assistant.

Answer questions using ONLY the hospital policy information supplied
in the retrieved context.

Rules:
1. Use the supplied policy context as your primary source.
2. Do not invent hospital policies, procedures, rules, or requirements.
3. If the answer is not available in the retrieved context, say:
   "I could not find this information in the hospital policy knowledge base."
4. Give a clear and concise answer.
5. When useful, mention the relevant department, document, and page.
6. Do not claim that a policy exists when it is not present in the context.
7. For medical or patient-safety questions, distinguish hospital policy
   information from clinical advice and recommend following applicable
   hospital procedures and qualified professional guidance.
"""

    user_prompt = f"""
Use the following hospital policy documents to answer the question.

================ RETRIEVED POLICY CONTEXT ================

{context}

================ END POLICY CONTEXT =======================

QUESTION:

{question}

Answer the question based only on the retrieved policy context.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1200
    )

    return response.choices[0].message.content


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"""
**Document:** {source["source"]}

**Department:** {source["department"]}

**Page:** {source["page"]}
"""
                    )


# User input
question = st.chat_input(
    "Ask a question about hospital policies..."
)


if question:

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Retrieve relevant chunks
    with st.spinner("Searching hospital policies..."):
        results = retrieve_chunks(question, TOP_K)

    if not results:
        answer = (
            "I could not find relevant information "
            "in the hospital policy knowledge base."
        )
        sources = []

    else:
        # Generate answer from retrieved chunks
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(
                    question,
                    results
                )
            except Exception as e:
                answer = (
                    "An error occurred while generating the answer: "
                    f"{e}"
                )

        # Collect unique document/page sources
        sources = []
        seen = set()

        for item in results:
            source_key = (
                item.get("source"),
                item.get("page"),
                item.get("department")
            )

            if source_key not in seen:
                seen.add(source_key)

                sources.append({
                    "source": item.get(
                        "source",
                        "Unknown"
                    ),
                    "department": item.get(
                        "department",
                        "Unknown"
                    ),
                    "page": item.get(
                        "page",
                        "Unknown"
                    )
                })

    # Display answer
    with st.chat_message("assistant"):
        st.markdown(answer)

        if sources:
            with st.expander("📚 Documents used"):
                for source in sources:
                    st.markdown(
                        f"""
**Document:** {source["source"]}

**Department:** {source["department"]}

**Page:** {source["page"]}
"""
                    )

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
