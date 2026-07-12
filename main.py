from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import io
import base64
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
def answer(req: ImageRequest):
    try:
        img_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(img_bytes))

        response = model.generate_content([
            f"""
Answer ONLY this question.

Question:
{req.question}

Return ONLY the raw answer.
""",
            image
        ])

        answer = ""

        if hasattr(response, "text") and response.text:
            answer = response.text.strip()

        return {
            "answer": answer
        }

    except Exception:
        return {
            "answer": ""
        }
        }
