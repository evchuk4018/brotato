"""
AliasService: Fuzzy matching for exercise names.
Maps variations and aliases to canonical exercise names.
"""

from thefuzz import fuzz, process
from typing import Optional, Dict, List


class AliasService:
    """
    Service for fuzzy matching exercise names against a canonical alias map.
    
    Example:
        "Leg Press (Machine)" -> "Leg Press"
        "Bench press" -> "Bench Press"
        "DB Rows" -> "Dumbbell Row"
    """
    
    # Default alias map: canonical_name -> list of aliases
    DEFAULT_ALIAS_MAP: Dict[str, List[str]] = {
        "Leg Press": ["Leg Press (Machine)", "Leg Press Machine", "LP"],
        "Bench Press": ["Bench", "Flat Bench", "BB Bench Press", "Barbell Bench"],
        "Squat": ["Back Squat", "BB Squat", "Barbell Squat"],
        "Deadlift": ["DL", "Conventional Deadlift", "BB Deadlift"],
        "Dumbbell Row": ["DB Row", "DB Rows", "Dumbbell Rows", "One Arm Row"],
        "Overhead Press": ["OHP", "Shoulder Press", "Military Press"],
        "Pull Up": ["Pullup", "Pull-up", "Chin Up", "Chinup"],
        "Romanian Deadlift": ["RDL", "Romanian DL", "Stiff Leg Deadlift"],
        "Lat Pulldown": ["Lat Pull Down", "Cable Pulldown", "Pulldown"],
        "Cable Row": ["Seated Row", "Seated Cable Row", "Low Row"],
    }
    
    def __init__(self, alias_map: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the AliasService.
        
        Args:
            alias_map: Custom alias map. Uses DEFAULT_ALIAS_MAP if not provided.
        """
        self.alias_map = alias_map or self.DEFAULT_ALIAS_MAP
        self._build_reverse_map()
    
    def _build_reverse_map(self) -> None:
        """Build reverse lookup: alias -> canonical name."""
        self._reverse_map: Dict[str, str] = {}
        self._all_names: List[str] = []
        
        for canonical, aliases in self.alias_map.items():
            # Add canonical name itself
            self._reverse_map[canonical.lower()] = canonical
            self._all_names.append(canonical)
            
            # Add all aliases
            for alias in aliases:
                self._reverse_map[alias.lower()] = canonical
                self._all_names.append(alias)
    
    def match(self, exercise_name: str, threshold: int = 80) -> Optional[str]:
        """
        Find the canonical exercise name for a given input.
        
        Args:
            exercise_name: The exercise name to match (from OCR/user input).
            threshold: Minimum fuzzy match score (0-100). Default 80.
            
        Returns:
            Canonical exercise name if match found, None otherwise.
        """
        if not exercise_name:
            return None
        
        normalized = exercise_name.strip().lower()
        
        # Try exact match first
        if normalized in self._reverse_map:
            return self._reverse_map[normalized]
        
        # Fuzzy match against all known names
        result = process.extractOne(
            exercise_name,
            self._all_names,
            scorer=fuzz.token_sort_ratio
        )
        
        if result and result[1] >= threshold:
            matched_name = result[0]
            # Return canonical name
            return self._reverse_map.get(matched_name.lower(), matched_name)
        
        return None
    
    def get_all_canonical_names(self) -> List[str]:
        """Return list of all canonical exercise names."""
        return list(self.alias_map.keys())
