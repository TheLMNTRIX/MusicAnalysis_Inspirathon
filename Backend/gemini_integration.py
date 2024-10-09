import logging
import google.generativeai as genai
from models import AudioSegment
from config import GEMINI_API_KEY, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash')

def get_gemini_analysis(segment: AudioSegment) -> str:
    logger.info(f"Requesting Gemini analysis for segment {segment.start_time}-{segment.end_time}")
    prompt = SYSTEM_PROMPT.format(
        start_time=segment.start_time,
        end_time=segment.end_time,
        tempo=segment.tempo,
        key=segment.key,
        segments=segment.segments[:5]
    )
    
    response = model.generate_content(prompt)
    logger.info("Gemini analysis received")
    logger.info(f"Gemini response: {response.text[:500]}...")  # Log the first 500 characters of the response
    return response.text