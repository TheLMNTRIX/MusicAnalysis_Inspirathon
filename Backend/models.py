from pydantic import BaseModel
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TimeRange:
    start: float
    end: float
    
    def __str__(self):
        return f"{self.start:.1f}-{self.end:.1f}"

@dataclass
class MusicalElement:
    name: str
    description: str
    confidence: float = 1.0

@dataclass
class Recommendation:
    element_type: str
    time_range: TimeRange
    description: str
    musical_elements: List[MusicalElement]
    genre_suggestions: List[str]
    processing_suggestions: List[str]
    confidence: float

class AudioSegment(BaseModel):
    start_time: float
    end_time: float
    tempo: float
    key: str
    segments: List[Dict[str, float]]

class AnalysisResult(BaseModel):
    overall_tempo: float
    overall_key: str
    duration: float
    segments: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]