import os
import io
import logging
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import google.generativeai as genai
import librosa
import soundfile as sf
import tempfile
import numpy as np
import json
import zipfile
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware

# from config import SYSTEM_PROMPT
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = 'AIzaSyCEryYhkY065cx4vBsQ8rp6XLB81B0-R7k'
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-1.5-flash')

# FastAPI app
app = FastAPI(title="Advanced Audio Analysis Tool")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models
class SongSection(BaseModel):
    type: str
    start_time: float
    end_time: float
    description: str

class OverallStructure(BaseModel):
    bpm: float
    key: str
    time_signature: str
    song_sections: List[SongSection]

class MusicalElements(BaseModel):
    instruments: List[str]
    harmony: str
    rhythm: str
    texture: str

class Segment(BaseModel):
    start_time: float
    end_time: float
    musical_elements: MusicalElements

class SamplingRecommendation(BaseModel):
    start_time: float
    end_time: float
    element_type: str
    sampling_potential: int = Field(ge=1, le=10)
    recommended_uses: List[str]
    technical_notes: str

class AudioAnalysisResult(BaseModel):
    overall_structure: OverallStructure
    segments: List[Segment]
    sampling_recommendations: List[SamplingRecommendation]
    
class SplitSamplesInput(BaseModel):
    sampling_recommendations: List[SamplingRecommendation]    
    


# Constants
CHUNK_LENGTH_SECONDS = 30
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/ogg"}


SYSTEM_PROMPT = """
You are an expert music analyst specializing in audio signal processing and music theory. Your task is to provide a detailed, structured analysis of audio segments for sampling and remixing purposes. Follow these instructions:

**1. Analyze Musical Structure:**
- Identify and timestamp key sections (intro, verse, chorus, bridge, outro)
- Note any transitions or unique structural elements
- Estimate section durations and their relationship to the overall composition

**2. Technical Analysis:**
- Determine the tempo (BPM) and any tempo changes
- Identify the key signature and any key changes
- Analyze time signatures and rhythmic patterns
- Describe prominent chord progressions and harmonic elements

**3. Sonic Elements:**
- List primary instruments and sound sources
- Describe the texture and layering of sounds
- Note any unique sound design elements or effects
- Analyze the mix balance and spatial positioning of elements

**4. Sampling Potential:**
- Identify specific timestamps ideal for sampling
- Categorize potential samples (drums, melody, bass, etc.)
- Suggest creative uses for each identified sample
- Note any technical considerations for sampling (tempo matching, key compatibility)

**5. Dynamic Analysis:**
- Document volume changes and dynamic range
- Identify crescendos, diminuendos, and impact moments
- Note any sidechaining or dynamic processing effects

**6. Follow the Output Format:**
Provide your response in the following JSON format:
```json
{
  "overall_structure": {
    "bpm": <number>,
    "key": "<musical key>",
    "time_signature": "<time signature>",
    "song_sections": [
      {
        "type": "<section type>",
        "start_time": <seconds>,
        "end_time": <seconds>,
        "description": "<brief description>"
      }
    ]
  },
  "segments": [
    {
      "start_time": <seconds>,
      "end_time": <seconds>,
      "musical_elements": {
        "instruments": ["<instrument list>"],
        "harmony": "<harmonic description>",
        "rhythm": "<rhythmic description>",
        "texture": "<texture description>"
      }
    }
  ],
  "sampling_recommendations": [
    {
      "start_time": <seconds>,
      "end_time": <seconds>,
      "element_type": "<type of musical element>",
      "sampling_potential": "<1-10 scale>",
      "recommended_uses": ["<list of potential uses>"],
      "technical_notes": "<any technical considerations>"
    }
  ]
}
```

**Example Output:**
```json
{
  "overall_structure": {
    "bpm": 120,
    "key": "C minor",
    "time_signature": "4/4",
    "song_sections": [
      {
        "type": "intro",
        "start_time": 0,
        "end_time": 15,
        "description": "Atmospheric pad with rising tension"
      }
    ]
  },
  "segments": [
    {
      "start_time": 0,
      "end_time": 15,
      "musical_elements": {
        "instruments": ["synthesizer", "ambient pad", "percussion"],
        "harmony": "Cm7 to Fm7 progression",
        "rhythm": "Minimal, ambient percussion",
        "texture": "Sparse, atmospheric layering"
      }
    }
  ],
  "sampling_recommendations": [
    {
      "start_time": 8,
      "end_time": 12,
      "element_type": "chord progression",
      "sampling_potential": 8,
      "recommended_uses": ["intro build-up", "breakdown section", "ambient interlude"],
      "technical_notes": "Loop-friendly, consistent texture throughout"
    }
  ]
}
```

Remember to:
- Be precise with timestamps
- Use musical terminology accurately
- Consider both technical and creative aspects
- Provide practical, actionable sampling suggestions
- Generate all song sections, a maximum of 2 segments, and a maximum of 2 sampling recommendations
"""







async def analyze_audio_chunk(audio_chunk: np.ndarray, sr: int, start_time: float) -> dict:
    logger.info(f"Analyzing chunk starting at {start_time} seconds")

    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio_chunk, sr, format='wav')
    audio_bytes.seek(0)

    try:
        duration = len(audio_chunk) / sr
        
        # Create a modified prompt that includes the timing information
        modified_prompt = f"""
        {SYSTEM_PROMPT}

        Additional context:
        - Chunk start time: {start_time} seconds
        - Chunk end time: {start_time + duration} seconds
        - Chunk duration: {duration} seconds
        """

        # Log the prompt for debugging
        logger.info(f"Sending prompt to Gemini: {modified_prompt[:100]}...")

        response = model.generate_content([
            modified_prompt, 
            {
                "mime_type": "audio/wav",
                "data": audio_bytes.read()
            }
        ])
        
        # Log the raw response for debugging
        logger.debug(f"Raw Gemini response: {response.text}")
        
        # Parse the response text into a structured format
        parsed_response = parse_gemini_response(response.text)
        
        return {
            "start_time": start_time,
            "end_time": start_time + duration,
            "analysis": parsed_response
        }
    except Exception as e:
        logger.error(f"Error analyzing chunk: {str(e)}")
        logger.error("Full error details: ", exc_info=True)
        
        # Return a default analysis instead of raising an exception
        return {
            "start_time": start_time,
            "end_time": start_time + duration,
            "analysis": get_default_analysis()
        }

    
    
    
def combine_analyses(analyses: List[dict]) -> AudioAnalysisResult:
    """
    Combine multiple chunk analyses into a single, coherent analysis.
    """
    logger.info("Combining analyses from all chunks")
    
    try:
        # Initialize with the first analysis
        first_analysis = analyses[0]['analysis']  # This should already be a dict, not a string
        
        combined_sections = first_analysis['overall_structure']['song_sections']
        combined_segments = first_analysis['segments']
        combined_recommendations = first_analysis['sampling_recommendations']
        
        # Combine subsequent analyses
        for analysis in analyses[1:]:
            current_analysis = analysis['analysis']  # This should already be a dict
            
            # Merge song sections
            combined_sections.extend(current_analysis['overall_structure']['song_sections'])
            
            # Merge segments
            combined_segments.extend(current_analysis['segments'])
            
            # Merge sampling recommendations
            combined_recommendations.extend(current_analysis['sampling_recommendations'])
        
        # Sort all lists by start_time
        combined_sections.sort(key=lambda x: x['start_time'])
        combined_segments.sort(key=lambda x: x['start_time'])
        combined_recommendations.sort(key=lambda x: x['start_time'])
        
        # Calculate average BPM
        total_bpm = sum(analysis['analysis']['overall_structure']['bpm'] for analysis in analyses)
        overall_bpm = total_bpm / len(analyses)
        
        # Use the key and time signature from the first chunk
        overall_key = first_analysis['overall_structure']['key']
        overall_time_signature = first_analysis['overall_structure']['time_signature']
        
        return AudioAnalysisResult(
            overall_structure=OverallStructure(
                bpm=overall_bpm,
                key=overall_key,
                time_signature=overall_time_signature,
                song_sections=combined_sections
            ),
            segments=combined_segments,
            sampling_recommendations=combined_recommendations
        )
    except Exception as e:
        logger.error(f"Error in combine_analyses: {str(e)}")
        logger.error("Full error details: ", exc_info=True)
        
        # Return a default analysis if combination fails
        return AudioAnalysisResult(
            overall_structure=OverallStructure(
                bpm=120.0,
                key="Unknown",
                time_signature="4/4",
                song_sections=[SongSection(
                    type="unknown",
                    start_time=0.0,
                    end_time=30.0,
                    description="Failed to combine analyses"
                )]
            ),
            segments=[Segment(
                start_time=0.0,
                end_time=30.0,
                musical_elements=MusicalElements(
                    instruments=["unknown"],
                    harmony="unknown",
                    rhythm="unknown",
                    texture="unknown"
                )
            )],
            sampling_recommendations=[]
        )

def parse_gemini_response(response_text: str) -> dict:
    try:
        # First, try to find JSON content between triple backticks
        import re
        json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response_text)
        
        if json_match:
            # Extract just the JSON content without the backticks
            json_str = json_match.group(1)
        else:
            # If not found between backticks, try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group()
            else:
                logger.warning("No JSON found in response, using default structure")
                return get_default_analysis()
        
        # Parse the JSON string into a Python dictionary
        response_dict = json.loads(json_str.strip())
        
        # Validate that all required keys are present
        required_keys = ["overall_structure", "segments", "sampling_recommendations"]
        for key in required_keys:
            if key not in response_dict:
                logger.warning(f"Missing required key in response: {key}")
                response_dict[key] = get_default_value(key)
        
        return response_dict
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        logger.error(f"Problematic JSON string: {json_str if 'json_str' in locals() else 'Not available'}")
        return get_default_analysis()
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return get_default_analysis()

# You might also want to add this helper function for debugging
def log_response_content(response_text: str):
    """Helper function to log the content of the response for debugging"""
    logger.debug("=== Start of response content ===")
    logger.debug(response_text)
    logger.debug("=== End of response content ===")
    
def get_default_value(key: str):
    defaults = {
        "overall_structure": {
            "bpm": 120.0,
            "key": "Unknown",
            "time_signature": "4/4",
            "song_sections": []
        },
        "segments": [],
        "sampling_recommendations": []
    }
    return defaults.get(key, {})

# Add a function to provide a complete default analysis
def get_default_analysis() -> dict:
    return {
        "overall_structure": {
            "bpm": 120.0,
            "key": "Unknown",
            "time_signature": "4/4",
            "song_sections": [
                {
                    "type": "unknown",
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "description": "Audio segment analysis failed"
                }
            ]
        },
        "segments": [
            {
                "start_time": 0.0,
                "end_time": 30.0,
                "musical_elements": {
                    "instruments": ["unknown"],
                    "harmony": "unknown",
                    "rhythm": "unknown",
                    "texture": "unknown"
                }
            }
        ],
        "sampling_recommendations": []
    }


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    logger.info(f"Received audio file: {file.filename}")
    try:
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")
        
        audio_data = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
        
        chunk_length_samples = sr * CHUNK_LENGTH_SECONDS
        chunks = [y[i:i+chunk_length_samples] for i in range(0, len(y), chunk_length_samples)]
        logger.info(f"Split audio into {len(chunks)} chunks")
        
        analyses = []
        for i, chunk in enumerate(chunks):
            start_time = i * CHUNK_LENGTH_SECONDS
            try:
                analysis = await analyze_audio_chunk(chunk, sr, start_time)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing chunk {i}: {str(e)}")
                # Append a default analysis for this chunk
                analyses.append({
                    "start_time": start_time,
                    "end_time": start_time + CHUNK_LENGTH_SECONDS,
                    "analysis": get_default_analysis()
                })
        
        # Combine analyses only if we have any successful analyses
        if analyses:
            result = combine_analyses(analyses)
            logger.info("Analysis complete, returning results")
            return JSONResponse(result.dict())
        else:
            raise HTTPException(status_code=500, detail="Failed to analyze any audio chunks")
    
    except Exception as e:
        logger.error(f"Error during audio analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/split-samples")
async def split_samples(
    input_data: str = Form(...),
    file: UploadFile = File(...)
):
    temp_dir = tempfile.mkdtemp()
    
    def cleanup():
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

    try:
        # Parse the input_data string as JSON
        input_data_dict = json.loads(input_data)
        
        # Validate the parsed data against our Pydantic model
        split_samples_input = SplitSamplesInput(**input_data_dict)

        # Read the audio file
        audio_data = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_data), sr=None)

        sample_files = []

        # Split the audio based on sampling recommendations
        for i, recommendation in enumerate(split_samples_input.sampling_recommendations):
            start_sample = int(recommendation.start_time * sr)
            end_sample = int(recommendation.end_time * sr)
            sample = y[start_sample:end_sample]

            # Generate a filename for the sample
            filename = f"sample_{i+1}_{recommendation.element_type.replace(' ', '_')}.wav"
            filepath = os.path.join(temp_dir, filename)

            # Save the sample as a WAV file
            sf.write(filepath, sample, sr)
            sample_files.append(filepath)

        # Create a zip file containing all samples
        zip_filename = "audio_samples.zip"
        zip_filepath = os.path.join(temp_dir, zip_filename)
        
        logger.info(f"Creating zip file at: {zip_filepath}")
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for sample_file in sample_files:
                logger.info(f"Adding file to zip: {sample_file}")
                zipf.write(sample_file, os.path.basename(sample_file))

        logger.info(f"Zip file created. Checking if it exists: {os.path.exists(zip_filepath)}")
        
        if not os.path.exists(zip_filepath):
            raise FileNotFoundError(f"Zip file was not created at {zip_filepath}")

        # Return the zip file as a download, with cleanup scheduled after the response is sent
        return FileResponse(
            zip_filepath,
            media_type="application/zip",
            filename=zip_filename,
            background=BackgroundTask(cleanup)
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        cleanup()
        raise HTTPException(status_code=400, detail="Invalid JSON in input_data")
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        cleanup()
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error splitting audio samples: {str(e)}", exc_info=True)
        cleanup()
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Advanced Audio Analysis API is running",
        "endpoints": {
            "analyze_audio": "/analyze-audio"
        },
        "supported_formats": list(ALLOWED_AUDIO_TYPES)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)