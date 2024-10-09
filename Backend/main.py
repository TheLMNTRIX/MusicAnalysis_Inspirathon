import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from models import AnalysisResult
from audio_processing import split_audio, analyze_chunk
from analysis import SophisticatedMusicParser, combine_analyses
from gemini_integration import get_gemini_analysis
from config import ALLOWED_AUDIO_TYPES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Audio Analysis Tool")

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    logger.info(f"Received audio file: {file.filename}")
    try:
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            logger.warning(f"Unsupported file type: {file.content_type}")
            raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")
        
        audio_data = await file.read()
        logger.info(f"Audio file read, size: {len(audio_data)} bytes")
        
        chunks, sr = split_audio(audio_data)
        
        segment_analyses = []
        gemini_analyses = []
        
        for i, (chunk, start_time) in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                segment = analyze_chunk(chunk, start_time, sr)
                segment_analyses.append(segment)
                
                analysis = get_gemini_analysis(segment)
                gemini_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        if not segment_analyses:
            raise HTTPException(status_code=500, detail="Failed to analyze any audio segments")
        
        logger.info("All chunks processed, combining analyses")
        final_result = combine_analyses(segment_analyses, gemini_analyses)
        
        logger.info("Analysis complete, returning results")
        return JSONResponse(final_result.dict())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during audio analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during audio analysis")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Advanced Audio Analysis API is running",
        "endpoints": {
            "analyze_audio": "/analyze-audio"
        },
        "supported_formats": ALLOWED_AUDIO_TYPES,
        "features": [
            "Tempo analysis",
            "Key detection",
            "Segment identification",
            "Genre suggestions",
            "Processing recommendations"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)