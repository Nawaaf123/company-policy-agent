import os
import re
import math
import pandas as pd
import requests
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END


load_dotenv()


# -----------------------------
# 1. Load company documents
# -----------------------------
print("Loading company documents...")

docs = []

for root, dirs, files in os.walk("docs"):
    for filename in files:
        if filename.lower().endswith((".txt", ".jsonl")):
            file_path = os.path.join(root, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"Loaded: {file_path}")

print(f"Total documents loaded: {len(docs)}")


# -----------------------------
# 2. Split documents into chunks
# -----------------------------
print("\nSplitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")


# -----------------------------
# 3. Create embeddings and vector database
# -----------------------------
print("\nCreating embeddings and vector database...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policy_docs"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 15,
        "fetch_k": 50
    }
)


print("Vector database ready.")


# -----------------------------
# 4. Load LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("LLM ready.")


# -----------------------------
# 5. Define Agent State
# -----------------------------
class AgentState(TypedDict):
    question: str
    intent: str
    documents: List[str]
    answer: str
    grounded: bool


# -----------------------------
# 6. Supervisor Agent
# -----------------------------
def supervisor_agent(state: AgentState):
    intent = "document_question"

    print(f"\n[Supervisor Agent] Intent detected: {intent}")

    return {
        "intent": intent
    }

def keyword_search_chunks(question: str, all_chunks, max_results: int = 10):
    stopwords = {
        "what", "is", "the", "for", "with", "all", "list", "show",
        "me", "a", "an", "of", "and", "or", "to", "in", "we", "have"
    }

    tokens = re.findall(r"[a-zA-Z0-9_-]+", question.lower())
    tokens = [token for token in tokens if len(token) > 2 and token not in stopwords]

    scored_docs = []

    for doc in all_chunks:
        content_lower = doc.page_content.lower()
        score = 0

        for token in tokens:
            if token in content_lower:
                score += 1

        # Strong boost for SKU/product catalog questions
        if "sku" in question.lower() and "sku" in content_lower:
            score += 5

        if "nova" in question.lower() and "nova" in content_lower:
            score += 3

        if "product" in question.lower() and "product" in content_lower:
            score += 2

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:max_results]]


# -----------------------------
# 7. Retrieval Agent
# -----------------------------
def retrieval_agent(state: AgentState):
    print("[Retrieval Agent] Searching company documents...")

    # Vector search
    vector_results = retriever.invoke(state["question"])

    # Keyword fallback search for SKU/product/catalog exact matches
    keyword_results = keyword_search_chunks(state["question"], chunks, max_results=10)

    # Merge results and remove duplicates
    combined_results = []
    seen = set()

    for doc in vector_results + keyword_results:
        source = doc.metadata.get("source", "Unknown source")
        key = source + doc.page_content[:150]

        if key not in seen:
            seen.add(key)
            combined_results.append(doc)

    retrieved_docs = []

    print("\n--- Retrieved Documents Preview ---")

    for index, doc in enumerate(combined_results[:20], start=1):
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content

        print(f"\nResult {index}")
        print(f"Source: {source}")
        print(content[:700])

        retrieved_docs.append(f"Source: {source}\nContent: {content}")

    print(f"\n[Retrieval Agent] Retrieved {len(retrieved_docs)} chunks.")

    return {
        "documents": retrieved_docs
    }

# -----------------------------
# 8. Answer Agent
# -----------------------------
def answer_agent(state: AgentState):
    print("[Answer Agent] Generating final answer...")

    context = "\n\n".join(state.get("documents", []))

    prompt = f"""
You are an enterprise company knowledge assistant.

Answer the user's question only using the provided company context.
Do not make up information.

If the user asks about MR FOG products, product categories, flavors, devices, specifications, wholesale, distributor information, contact information, warranty, returns, or FAQ, answer using the retrieved MR FOG website context.

If the retrieved context contains related MR FOG information, summarize it clearly.
If internal MR FOG documents and public website documents contain different information, prefer the internal MR FOG documents because they are more structured and company-specific.
Only say "I do not have enough information in the company documents" when the retrieved context has no relevant information at all.


At the end of your answer, mention the source document name from the context.

If the answer is not available in the company context, say:
"I do not have enough information in the company documents."

User question:
{state["question"]}

Company context:
{context}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -----------------------------
# 9. Guardrail Agent
# -----------------------------
def guardrail_agent(state: AgentState):
    print("[Guardrail Agent] Checking answer grounding...")

    documents = state.get("documents", [])
    answer = state.get("answer", "")

    if state["intent"] == "document_question" and len(documents) == 0:
        return {
            "grounded": False,
            "answer": "I do not have enough information in the company documents."
        }

    if "I do not have enough information" in answer:
        return {
            "grounded": False
        }

    return {
        "grounded": True
    }


# -----------------------------
# 10. Routing Function
# -----------------------------
def route_after_supervisor(state: AgentState):
    return "retrieve"

# -----------------------------
# 11. Build LangGraph Workflow
# -----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("retrieve", retrieval_agent)
workflow.add_node("answer", answer_agent)
workflow.add_node("guardrail", guardrail_agent)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "retrieve": "retrieve",
        "answer": "answer"
    }
)

workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", "guardrail")
workflow.add_edge("guardrail", END)

agent_app = workflow.compile()


# -----------------------------
# 12. Function for UI or CLI
# -----------------------------

WHOLESALER_FILE = os.path.join("data", "wholesalers_geocoded.csv")


def is_nearest_wholesaler_question(question: str) -> bool:
    q = question.lower()

    keywords = [
        "nearest wholesaler",
        "closest wholesaler",
        "nearby wholesaler",
        "where can i get",
        "where can i buy",
        "near me",
        "near my address"
    ]

    return any(keyword in q for keyword in keywords)


def geocode_user_address(address: str):
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")

    if not mapbox_token:
        raise ValueError("MAPBOX_ACCESS_TOKEN is missing")

    url = "https://api.mapbox.com/geocoding/v5/mapbox.places/" + requests.utils.quote(address) + ".json"

    params = {
        "access_token": mapbox_token,
        "country": "us",
        "limit": 1
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    features = data.get("features", [])

    if not features:
        return None, None

    coordinates = features[0]["center"]

    longitude = coordinates[0]
    latitude = coordinates[1]

    return latitude, longitude


def haversine_miles(lat1, lon1, lat2, lon2):
    radius_miles = 3958.8

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return radius_miles * c


def find_nearest_wholesaler(user_address: str):
    if not os.path.exists(WHOLESALER_FILE):
        return "Wholesaler location file is missing. Please create data/wholesalers_geocoded.csv first."

    user_lat, user_lon = geocode_user_address(user_address)

    if user_lat is None or user_lon is None:
        return "I could not understand that address. Please enter a complete address with city and state."

    df = pd.read_csv(WHOLESALER_FILE)

    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        return "No geocoded wholesaler locations are available."

    nearest = None
    nearest_distance = None

    for _, row in df.iterrows():
        distance = haversine_miles(
            user_lat,
            user_lon,
            row["latitude"],
            row["longitude"]
        )

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest = row

    if nearest is None:
        return "I could not find a nearby wholesaler."

    return f"""The nearest wholesaler is:

{nearest["name"]}

Address:
{nearest["address"]}"""


def ask_question(user_question: str):
    if is_nearest_wholesaler_question(user_question):
        answer = find_nearest_wholesaler(user_question)

        return {
            "question": user_question,
            "intent": "nearest_wholesaler",
            "documents": [],
            "answer": answer,
            "grounded": True
        }

    result = agent_app.invoke({
        "question": user_question,
        "intent": "",
        "documents": [],
        "answer": "",
        "grounded": False
    })

    return result


# -----------------------------
# 13. Run Chatbot in Terminal
# -----------------------------
if __name__ == "__main__":
    while True:
        user_question = input("\nUser Question: ")

        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        result = ask_question(user_question)

        print("\nFinal Answer:")
        print(result["answer"])

        print("\nGrounded:")
        print(result["grounded"])