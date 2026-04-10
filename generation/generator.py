from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def build_prompt(self, query: str, reranked: list[dict]) -> str:
        context = ""
        for i, r in enumerate(reranked, 1):
            context += f"\n[Context {i}]\n{r['chunk']['text']}\n"

        return (
            f"You are a helpful customer support assistant.\n"
            f"Use the context below to answer the customer's question.\n"
            f"The context may contain placeholders like {{Order Number}} or {{Customer Name}} "
            f"— treat these as real values and still provide a helpful general answer.\n"
            f"If you truly cannot answer, say 'Please contact our support team directly.'\n\n"
            f"{context}\n"
            f"Customer question: {query}\n\n"
            f"Answer:"
        )

    def generate(self, query: str, reranked: list[dict]) -> str:
        prompt = self.build_prompt(query, reranked)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()