"""
VisionService: Gemini 1.5 Flash for workout screenshot OCR.
Extracts exercise data from workout app screenshots.
"""

import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import re


class VisionService:
    """
    Service for extracting workout data from screenshots using Gemini Vision.
    """
    
    EXTRACTION_PROMPT = """Analyze this workout screenshot and extract all exercise data.

Return a JSON object with the following structure:
{
    "exercises": [
        {
            "name": "Exercise Name",
            "sets": [
                {"weight": 225, "reps": 8},
                {"weight": 245, "reps": 6}
            ]
        }
    ],
    "date": "YYYY-MM-DD" (if visible, otherwise null),
    "notes": "any additional notes visible"
}

Rules:
1. Extract ALL exercises and sets visible in the screenshot
2. Weight should be in pounds (convert if in kg)
3. Include warm-up sets - they will be filtered later
4. If prescription info is visible (e.g., "3x8", "1x5"), include it in notes

Return ONLY valid JSON, no markdown formatting."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initialize VisionService.
        
        Args:
            api_key: Google AI API key. If None, uses GOOGLE_API_KEY env var.
            model: Model name to use.
        """
        if api_key:
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(model)
    
    def extract_workout_data(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extract workout data from screenshot image.
        
        Args:
            image_bytes: Raw image bytes from screenshot.
            
        Returns:
            Dict with extracted exercise data.
        """
        # Create image part for multimodal input
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        
        response = self.model.generate_content([
            self.EXTRACTION_PROMPT,
            image_part
        ])
        
        return self._parse_response(response.text)
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini response into structured data.
        
        Args:
            response_text: Raw response from Gemini.
            
        Returns:
            Parsed exercise data dict.
        """
        # Clean up response - remove markdown code blocks if present
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```json\s*', '', cleaned)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        try:
            data = json.loads(cleaned)
            return data
        except json.JSONDecodeError:
            # Return empty structure on parse failure
            return {
                "exercises": [],
                "date": None,
                "notes": f"Parse error: {response_text[:100]}"
            }
