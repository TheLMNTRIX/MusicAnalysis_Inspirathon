import logging
import re
import numpy as np
from typing import List, Optional
from models import TimeRange, MusicalElement, Recommendation, AudioSegment, AnalysisResult

logger = logging.getLogger(__name__)

class SophisticatedMusicParser:
    def __init__(self):
        logger.info("Initializing SophisticatedMusicParser")
        self.time_patterns = {
            'range': r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*seconds?',
            'point': r'at\s+(\d+\.?\d*)\s*seconds?',
            'duration': r'for\s+(\d+\.?\d*)\s*seconds?'
        }
        
        self.musical_elements = {
            'rhythm': r'(rhythm|beat|tempo|bpm)',
            'melody': r'(melody|melodic|tune|hook)',
            'harmony': r'(harmony|chord|key|scale)',
            'texture': r'(texture|timbre|sound)',
            'dynamics': r'(dynamics|volume|intensity)'
        }
        
        self.genre_keywords = {
            'hip_hop': ['hip hop', 'rap', 'trap'],
            'electronic': ['edm', 'electronic', 'techno', 'house'],
            'rock': ['rock', 'metal', 'punk'],
            'jazz': ['jazz', 'blues', 'swing']
        }
        
        self.processing_keywords = {
            'eq': ['equalizer', 'eq', 'frequency'],
            'compression': ['compress', 'dynamics'],
            'reverb': ['reverb', 'space', 'room'],
            'delay': ['delay', 'echo']
        }
        
        self.confidence_modifiers = {
            'high': ['definitely', 'clearly', 'strong', 'perfect'],
            'medium': ['could', 'might', 'maybe'],
            'low': ['possibly', 'perhaps', 'subtle']
        }

    def extract_time_range(self, text: str) -> Optional[TimeRange]:
        logger.info(f"Extracting time range from: {text[:100]}...")  # Log the first 100 characters
        for pattern_type, pattern in self.time_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if pattern_type == 'range':
                    result = TimeRange(float(match.group(1)), float(match.group(2)))
                elif pattern_type == 'point':
                    point = float(match.group(1))
                    result = TimeRange(point, point)
                elif pattern_type == 'duration':
                    duration = float(match.group(1))
                    result = TimeRange(0, duration)
                logger.info(f"Time range extracted: {result}")
                return result
        logger.warning("No time range found")
        return None

    def extract_musical_elements(self, text: str) -> List[MusicalElement]:
        logger.info("Extracting musical elements")
        elements = []
        for element_type, pattern in self.musical_elements.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                sentence = re.findall(r'[^.!?]+[.!?]', text[max(0, match.start()-50):match.end()+50])
                if sentence:
                    description = sentence[0].strip()
                    confidence = self.calculate_confidence(description)
                    elements.append(MusicalElement(element_type, description, confidence))
        logger.info(f"Extracted {len(elements)} musical elements")
        return elements

    def suggest_genres(self, text: str) -> List[str]:
        logger.info("Suggesting genres")
        suggestions = []
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                suggestions.append(genre)
        logger.info(f"Suggested genres: {suggestions}")
        return suggestions

    def extract_processing_suggestions(self, text: str) -> List[str]:
        logger.info("Extracting processing suggestions")
        suggestions = []
        for proc_type, keywords in self.processing_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                sentence = re.findall(r'[^.!?]+[.!?]', text)
                if sentence:
                    suggestions.append(f"{proc_type}: {sentence[0].strip()}")
        logger.info(f"Extracted {len(suggestions)} processing suggestions")
        return suggestions

    def calculate_confidence(self, text: str) -> float:
        logger.info("Calculating confidence")
        base_confidence = 0.5
        for level, modifiers in self.confidence_modifiers.items():
            if any(modifier in text.lower() for modifier in modifiers):
                if level == 'high':
                    confidence = min(base_confidence + 0.3, 1.0)
                elif level == 'medium':
                    confidence = base_confidence
                else:  # low
                    confidence = max(base_confidence - 0.2, 0.1)
                logger.info(f"Calculated confidence: {confidence}")
                return confidence
        logger.info(f"Default confidence: {base_confidence}")
        return base_confidence

    def parse_analysis(self, analysis: str, segment_start_time: float = 0) -> List[Recommendation]:
        logger.info(f"Parsing analysis for segment starting at {segment_start_time}")
        logger.info(f"Raw Gemini response: {analysis[:500]}...")  # Log the first 500 characters of the response
        recommendations = []
        sentences = re.split(r'(?<=[.!?])\s+', analysis)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            time_range = self.extract_time_range(sentence)
            if not time_range:
                logger.info(f"No time range found in sentence: {sentence}")
                continue
                
            adjusted_time_range = TimeRange(
                time_range.start + segment_start_time,
                time_range.end + segment_start_time
            )
                
            musical_elements = self.extract_musical_elements(sentence)
            if not musical_elements:
                logger.info(f"No musical elements found in sentence: {sentence}")
                continue
                
            element_type = musical_elements[0].name
            confidence = sum(elem.confidence for elem in musical_elements) / len(musical_elements)
                
            recommendation = Recommendation(
                element_type=element_type,
                time_range=adjusted_time_range,
                description=sentence.strip(),
                musical_elements=musical_elements,
                genre_suggestions=self.suggest_genres(sentence),
                processing_suggestions=self.extract_processing_suggestions(sentence),
                confidence=confidence
            )
            recommendations.append(recommendation)
        
        logger.info(f"Parsed {len(recommendations)} recommendations")
        return recommendations

def combine_analyses(segments: List[AudioSegment], gemini_analyses: List[str]) -> AnalysisResult:
    logger.info("Combining analyses from all segments")
    parser = SophisticatedMusicParser()
    all_recommendations = []
    tempos = []
    keys = []
    all_segments = []
    
    for segment, analysis in zip(segments, gemini_analyses):
        tempos.append(segment.tempo)
        keys.append(segment.key)
        all_segments.extend(segment.segments)
        
        recommendations = parser.parse_analysis(analysis, segment.start_time)
        all_recommendations.extend(recommendations)
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: x['start'])
    
    # Calculate overall metrics
    overall_tempo = float(np.median(tempos))
    overall_key = max(set(keys), key=keys.count)
    duration = max(segment.end_time for segment in segments)
    
    # Convert recommendations to dictionary format
    formatted_recommendations = [
        {
            "type": rec.element_type,
            "time_range": str(rec.time_range),
            "description": rec.description,
            "musical_elements": [
                {"name": elem.name, "description": elem.description}
                for elem in rec.musical_elements
            ],
            "genre_suggestions": rec.genre_suggestions,
            "processing_suggestions": rec.processing_suggestions,
            "confidence": rec.confidence
        }
        for rec in all_recommendations
    ]
    
    logger.info(f"Analysis combination complete. Overall tempo: {overall_tempo}, Key: {overall_key}, Duration: {duration}")
    return AnalysisResult(
        overall_tempo=overall_tempo,
        overall_key=overall_key,
        duration=duration,
        segments=all_segments,
        recommendations=formatted_recommendations
    )