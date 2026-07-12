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

        prompt = f"""
Answer the question from the image.

Question:
{req.question}

Rules:
- Return ONLY the answer.
- No explanation.
- If numeric, return only digits/decimal.
- Never include units or currency.
"""

        response = model.generate_content([prompt, image])

        return {"answer": response.text.strip()}

    except Exception as e:
        return {"error": str(e)}
    return {
        "answer": response.text.strip()
    }
