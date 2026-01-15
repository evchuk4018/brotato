"""
WorkoutProcessor: Smart selection logic for workout sets.
Handles warm-up filtering and set selection based on prescription.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class SetPrescription(Enum):
    """Prescription type for exercise sets."""
    SINGLE = "1x"  # Take top set only
    TRIPLE = "3x"  # Take top 3 sets
    ALL = "all"    # Take all working sets


@dataclass
class ExerciseSet:
    """Represents a single set of an exercise."""
    weight: float
    reps: int
    is_warmup: bool = False
    
    @property
    def estimated_1rm(self) -> float:
        """Estimate 1RM using Brzycki formula."""
        if self.reps >= 37:
            return self.weight
        return self.weight * (36 / (37 - self.reps))


@dataclass 
class Exercise:
    """Represents an exercise with all its sets."""
    name: str
    sets: List[ExerciseSet]
    prescription: SetPrescription = SetPrescription.ALL


class WorkoutProcessor:
    """
    Processes workout data with smart selection logic.
    
    Rules:
    1. Filter warm-ups: Sets with weight < 50% of max weight
    2. If prescription is "1x": Return top set only (highest weight)
    3. If prescription is "3x": Return top 3 sets
    4. Otherwise: Return all working sets (non-warmup)
    """
    
    WARMUP_THRESHOLD = 0.5  # 50% of max weight
    
    def __init__(self, warmup_threshold: float = 0.5):
        """
        Initialize WorkoutProcessor.
        
        Args:
            warmup_threshold: Percentage of max weight below which sets are warm-ups.
        """
        self.warmup_threshold = warmup_threshold
    
    def identify_warmups(self, sets: List[ExerciseSet]) -> List[ExerciseSet]:
        """
        Mark warm-up sets based on weight threshold.
        
        Args:
            sets: List of exercise sets.
            
        Returns:
            Same list with is_warmup flags updated.
        """
        if not sets:
            return sets
        
        max_weight = max(s.weight for s in sets)
        threshold_weight = max_weight * self.warmup_threshold
        
        for s in sets:
            s.is_warmup = s.weight <= threshold_weight
        
        return sets
    
    def select_working_sets(
        self, 
        sets: List[ExerciseSet], 
        prescription: SetPrescription = SetPrescription.ALL
    ) -> List[ExerciseSet]:
        """
        Select working sets based on prescription.
        
        Args:
            sets: List of exercise sets (should have warmups already identified).
            prescription: How many sets to select.
            
        Returns:
            Filtered list of working sets.
        """
        # Filter out warm-ups
        working_sets = [s for s in sets if not s.is_warmup]
        
        if not working_sets:
            return []
        
        # Sort by weight descending
        working_sets.sort(key=lambda s: s.weight, reverse=True)
        
        if prescription == SetPrescription.SINGLE:
            return working_sets[:1]
        elif prescription == SetPrescription.TRIPLE:
            return working_sets[:3]
        else:
            return working_sets
    
    def process_exercise(self, exercise: Exercise) -> List[ExerciseSet]:
        """
        Full processing pipeline for an exercise.
        
        Args:
            exercise: Exercise with raw sets.
            
        Returns:
            List of selected working sets.
        """
        # Step 1: Identify warm-ups
        self.identify_warmups(exercise.sets)
        
        # Step 2: Select based on prescription
        return self.select_working_sets(exercise.sets, exercise.prescription)
    
    def process_workout(self, exercises: List[Exercise]) -> List[Exercise]:
        """
        Process entire workout.
        
        Args:
            exercises: List of exercises.
            
        Returns:
            Exercises with filtered sets.
        """
        result = []
        for exercise in exercises:
            processed_sets = self.process_exercise(exercise)
            if processed_sets:
                result.append(Exercise(
                    name=exercise.name,
                    sets=processed_sets,
                    prescription=exercise.prescription
                ))
        return result
