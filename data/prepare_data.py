from datasets import load_dataset
import pandas as pd
import pickle
import os

def load_and_prepare():
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )
    df = pd.DataFrame(dataset["train"])
    df = df[["instruction", "category", "intent", "response"]].dropna()
    df = df.head(5000)

    print(f"Dataset size: {len(df)} rows")

    # Build chunk objects (each Q&A pair = one chunk)
    chunks = []
    for _, row in df.iterrows():
        chunks.append({
            "text": f"Question: {row['instruction']}\nAnswer: {row['response']}",
            "question": row["instruction"],
            "answer": row["response"],
            "category": row["category"],
            "intent": row["intent"],
        })

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks to artifacts/chunks.pkl")
    return chunks

if __name__ == "__main__":
    load_and_prepare()