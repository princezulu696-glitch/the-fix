import base64
import os
import requests

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
VISION_MODEL = "qwen3-vl:2b"


@router.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    question: str = "Describe and analyze this image in detail."
):
    try:
        # Read uploaded image
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image was uploaded.")

        # Basic safety limit
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image is too large. Maximum size is 20 MB."
            )

        # Convert image to Base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Ask the vision model
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                    "images": [image_base64]
                }
            ],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.1,
                "num_predict": 500
            }
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama error: {response.text}"
            )

        data = response.json()

        answer = (
            data.get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            answer = "The Fix could not produce an answer for this image."

        return {
            "success": True,
            "filename": image.filename,
            "content_type": image.content_type,
            "model": VISION_MODEL,
            "question": question,
            "answer": answer
        }

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The vision model took too long to respond."
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="The Fix cannot connect to Ollama."
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}"
        )