import logging
import librosa
import numpy as np
import io
from typing import List, Tuple
from models import AudioSegment
from config import CHUNK_DURATION

logger = logging.getLogger(__name__)

def split_audio(audio_data: bytes) -> Tuple[List[Tuple[np.ndarray, float]], int]:
    logger.info("Starting audio splitting process")
    try:
        with io.BytesIO(audio_data) as audio_io:
            y, sr = librosa.load(audio_io)
        
        total_samples = len(y)
        samples_per_chunk = int(sr * CHUNK_DURATION)
        chunks = []
        
        for i in range(0, total_samples, samples_per_chunk):
            chunk = y[i:i + samples_per_chunk]
            if len(chunk) < sr:  # Skip chunks shorter than 1 second
                continue
            chunks.append((chunk, i/sr))
        
        logger.info(f"Audio split into {len(chunks)} chunks")
        return chunks, sr
    except Exception as e:
        logger.error(f"Error in split_audio: {str(e)}")
        raise

def analyze_chunk(chunk: np.ndarray, start_time: float, sr: int) -> AudioSegment:
    try:
        logger.info(f"Analyzing chunk starting at {start_time} seconds")
        
        # Tempo analysis with fallback
        try:
            tempo, beats = librosa.beat.beat_track(y=chunk, sr=sr)
            logger.info(f"Tempo analysis complete: {tempo}")
        except Exception as e:
            logger.warning(f"Tempo analysis failed: {str(e)}. Using default tempo.")
            tempo = 120.0
        
        # Key estimation with fallback
        try:
            y_harmonic = librosa.effects.harmonic(y=chunk)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            chroma_sum = np.sum(chroma, axis=1)
            key_index = np.argmax(chroma_sum)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key = key_names[key_index]
            logger.info(f"Key estimation complete: {key}")
        except Exception as e:
            logger.warning(f"Key estimation failed: {str(e)}. Using default key.")
            key = 'Unknown'
        
        # Segment analysis with fallback
        segments = []
        try:
            onset_envelope = librosa.onset.onset_strength(y=chunk, sr=sr)
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_envelope,
                sr=sr,
                units='time'
            )
            
            if len(onset_frames) > 1:
                for i in range(len(onset_frames) - 1):
                    segment_start = onset_frames[i]
                    segment_end = onset_frames[i+1]
                    
                    absolute_start = start_time + segment_start
                    absolute_end = start_time + segment_end
                    
                    segments.append({
                        "start": float(absolute_start),
                        "end": float(absolute_end),
                        "duration": float(segment_end - segment_start)
                    })
            else:
                segments.append({
                    "start": float(start_time),
                    "end": float(start_time + len(chunk)/sr),
                    "duration": float(len(chunk)/sr)
                })
            
            logger.info(f"Segment analysis complete. Found {len(segments)} segments")
        except Exception as e:
            logger.warning(f"Segment analysis failed: {str(e)}. Using single segment.")
            segments = [{
                "start": float(start_time),
                "end": float(start_time + len(chunk)/sr),
                "duration": float(len(chunk)/sr)
            }]
        
        return AudioSegment(
            start_time=start_time,
            end_time=start_time + len(chunk)/sr,
            tempo=float(tempo),
            key=key,
            segments=segments
        )
    except Exception as e:
        logger.error(f"Error in analyze_chunk: {str(e)}")
        raise