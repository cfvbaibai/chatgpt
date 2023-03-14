import json

import numpy as np
import openai
import pandas as pd
from numpy import ndarray

CHAT_COMPLETION_MODEL = "gpt-3.5-turbo"
COMPLETIONS_MODEL = 'text-davinci-003'
EMBEDDING_MODEL = "text-embedding-ada-002"

df_qa = pd.read_csv('data/md_QA_embedded.csv')
df_qa['embedding'] = df_qa['embedding'].apply(lambda x: json.loads(x))


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def vector_similarity(x: list[float], y: list[float]) -> ndarray:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def calc_embeddings(df):
    df['embedding'] = df['Questions'].apply(lambda s: get_embedding(s))
    df.to_csv('../data/embedded.csv', index=False)
    return df.head()


class Dialog:
    messages = []

    @staticmethod
    def _get_most_similar_qa(question):
        q_embedding = get_embedding(question)
        df = df_qa.copy()
        df['similarity'] = df['embedding'].apply(lambda x: vector_similarity(x, q_embedding))
        sorted_df = df.sort_values(by='similarity', ascending=False)
        best_q, best_a, similarity = sorted_df[['Questions', 'Answers', 'similarity']].iloc[0]
        return best_q, best_a, similarity

    @staticmethod
    def _get_prompt_by_best_qa(q, best_q, best_a, similarity):
        lines = []
        if similarity > 0.9:
            lines.append(f"Q: {best_q}")
            lines.append(f"A: {best_a}")
            lines.append("")
        lines.append(f"Q: {q}")
        lines.append("A: ")
        prompt = "\n".join(lines)
        print(prompt)
        return prompt

    def set_cpu_role(self, message):
        self.messages = [item for item in self.messages if item["role"] != "system"]
        self.messages.insert(0, {"role": "system", "content": message})

    def ask(self, question):
        best_q, best_a, similarity = Dialog._get_most_similar_qa(question)
        if similarity > 0.925:
            answer = best_a
        else:
            prompt = Dialog._get_prompt_by_best_qa(question, best_q, best_a, similarity)
            self.messages.append({"role": "user", "content": prompt})
            result = openai.ChatCompletion.create(
                model=CHAT_COMPLETION_MODEL,
                messages=self.messages
            )
            answer = result.choices[0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": answer})
        return {"final_a": answer, "best_q": best_q, "best_a": best_a, "similarity": similarity}
