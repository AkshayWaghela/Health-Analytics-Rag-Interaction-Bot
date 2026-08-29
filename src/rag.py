import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def extract_user_id(message):

    patterns = [
        r"user\s*id\s*(\d+)",
        r"user\s*(\d+)",
        r"person\s*(\d+)",
        r"profile\s*(\d+)"
    ]

    message = message.lower()

    for pattern in patterns:

        match = re.search(
            pattern,
            message
        )

        if match:
            return int(match.group(1))

    return None


class HealthRAG:

    def __init__(
        self,
        df,
        documents,
        embeddings,
        index
    ):

        self.df = df
        self.documents = documents
        self.embeddings = embeddings
        self.index = index

        self.embedding_model = (
            SentenceTransformer(MODEL_NAME)
        )

        self.llm = pipeline(
            "text-generation",
            model=LLM_NAME,
            device_map="auto"
        )


    def get_user(self, user_id):

        result = self.df[
            self.df["user_id"] == user_id
        ]

        if result.empty:
            return None

        return result.iloc[0]


    def retrieve_users(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True
            )
        )

        query_embedding = (
            query_embedding.astype(np.float32)
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            if idx >= len(self.documents):
                continue

            results.append({

                "score": round(
                    float(score),
                    4
                ),

                "user_id": int(
                    self.df.iloc[idx]["user_id"]
                ),

                "document": self.documents[idx]
            })

        return results


    def get_user_context(
        self,
        user_id
    ):

        user = self.get_user(
            user_id
        )

        if user is None:
            return None

        context = {

            "User": {

                "User ID":
                    user.get("user_id"),

                "Age":
                    user.get("age_first"),

                "Sex":
                    user.get("sex_first"),

                "BMI":
                    user.get("bmi_mean"),

                "Average Heart Rate":
                    user.get(
                        "avg_heart_rate_mean"
                    ),

                "Average Steps":
                    user.get(
                        "steps_mean"
                    ),

                "Average Sleep":
                    user.get(
                        "sleep_hours_mean"
                    ),

                "Water Intake":
                    user.get(
                        "water_intake_l_mean"
                    ),

                "Risk Probability":
                    user.get(
                        "risk_probability"
                    ),

                "Risk State":
                    user.get(
                        "cardiometabolic_risk_state_max"
                    ),

                "Health Score":
                    user.get(
                        "health_score"
                    ),

                "Health Tier":
                    user.get(
                        "health_tier"
                    ),

                "Health Category":
                    user.get(
                        "health_category"
                    )
            }
        }

        return context


    def generate_answer(
        self,
        question,
        context
    ):

        prompt = f"""
You are a health data analytics assistant.

Answer the user's question using ONLY the DATA provided.

RULES:

1. Never invent numbers.
2. Never invent users.
3. Never assume missing information.
4. Do not diagnose diseases.
5. Do not provide medical certainty.
6. Do not use outside information.
7. If information is unavailable, say:
   "That information is not available in the dataset."
8. Only compare values explicitly present in DATA.
9. Keep the answer simple and concise.
10. Use actual dataset values whenever relevant.

USER QUESTION:
{question}

DATA:
{context}

Answer using ONLY the DATA.
"""

        output = self.llm(
            [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_new_tokens=250,
            do_sample=False
        )

        generated = output[0]["generated_text"]

        if isinstance(
            generated,
            list
        ):

            for message in reversed(
                generated
            ):

                if (
                    isinstance(
                        message,
                        dict
                    )
                    and
                    message.get("role")
                    == "assistant"
                ):

                    return message.get(
                        "content",
                        ""
                    ).strip()

        return str(
            generated
        ).strip()


    def answer_user_question(
        self,
        message
    ):

        user_id = extract_user_id(
            message
        )

        if user_id is not None:

            context = self.get_user_context(
                user_id
            )

            if context is None:

                return (
                    f"❌ User {user_id} "
                    "was not found in the dataset."
                )

            return self.generate_answer(
                message,
                json.dumps(
                    context,
                    indent=2,
                    default=str
                )
            )


        results = self.retrieve_users(
            message,
            top_k=5
        )

        if not results:

            return (
                "I couldn't find relevant "
                "information in the dataset."
            )


        retrieved_context = "\n\n".join(

            [
                f"""
User ID: {r['user_id']}
Similarity Score: {r['score']}

{r['document']}
"""
                for r in results
            ]
        )


        return self.generate_answer(
            message,
            retrieved_context
        )
