from groq import Groq
from src.config import GROQ_API_KEY


def get_client() -> Groq:
    """Return a configured Groq client.
    
    Swap this function in a later evaluation to change providers.
    Nothing outside this file needs to know which SDK is used.
    """
    return Groq(api_key=GROQ_API_KEY)