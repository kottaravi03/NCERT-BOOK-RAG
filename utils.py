from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
import base64

load_dotenv()


def get_model_from_gcp(
    model_name: str = "gemini-2.5-flash-lite",
) -> ChatGoogleGenerativeAI:
    """Returns the Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=model_name,
    )


def get_embeddings_from_gcp(
    model_name: str = "text-embedding-005",
) -> GoogleGenerativeAIEmbeddings:
    """Returns the embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
    )


def image_to_base64(image_path: str) -> str:
    """Convert an image file to Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")