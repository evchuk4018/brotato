"""Tests for WorkoutProcessor smart selection logic."""

import pytest
from services.workout_processor import (
    WorkoutProcessor,
    Exercise,
    ExerciseSet,
    SetPrescription
)


class TestWorkoutProcessor:
    """Test suite for WorkoutProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create WorkoutProcessor instance."""
        return WorkoutProcessor()
    
    @pytest.fixture
    def sample_sets(self):
        """Create sample exercise sets including warm-ups."""
        return [
            ExerciseSet(weight=135, reps=10),  # Warm-up (< 50% of 315)
            ExerciseSet(weight=185, reps=8),   # Warm-up
            ExerciseSet(weight=225, reps=5),   # Working set
            ExerciseSet(weight=275, reps=3),   # Working set
            ExerciseSet(weight=315, reps=1),   # Top set (max)
        ]
    
    def test_identify_warmups(self, processor, sample_sets):
        """Test that warm-up sets are correctly identified."""
        processor.identify_warmups(sample_sets)
        
        # 50% of 315 = 157.5
        assert sample_sets[0].is_warmup is True   # 135 < 157.5
        assert sample_sets[1].is_warmup is False  # 185 > 157.5
        assert sample_sets[2].is_warmup is False  # 225 > 157.5
        assert sample_sets[3].is_warmup is False  # 275 > 157.5
        assert sample_sets[4].is_warmup is False  # 315 > 157.5
    
    def test_select_working_sets_single_prescription(self, processor, sample_sets):
        """
        REQUIREMENT: If prescription is "1x", take top set only.
        """
        processor.identify_warmups(sample_sets)
        result = processor.select_working_sets(sample_sets, SetPrescription.SINGLE)
        
        assert len(result) == 1, "1x prescription should return exactly 1 set"
        assert result[0].weight == 315, "Should return the heaviest set"
    
    def test_select_working_sets_triple_prescription(self, processor, sample_sets):
        """
        REQUIREMENT: If prescription is "3x", take top 3 sets.
        """
        processor.identify_warmups(sample_sets)
        result = processor.select_working_sets(sample_sets, SetPrescription.TRIPLE)
        
        assert len(result) == 3, "3x prescription should return exactly 3 sets"
        weights = [s.weight for s in result]
        assert weights == [315, 275, 225], "Should return top 3 heaviest sets in order"
    
    def test_select_working_sets_all_prescription(self, processor, sample_sets):
        """Test that ALL prescription returns all working sets."""
        processor.identify_warmups(sample_sets)
        result = processor.select_working_sets(sample_sets, SetPrescription.ALL)
        
        # Should exclude only the 135 warm-up
        assert len(result) == 4
        assert all(s.weight >= 185 for s in result)
    
    def test_warmup_threshold_filtering(self, processor):
        """
        REQUIREMENT: Filter warm-ups (sets with weight < 50% of max).
        """
        sets = [
            ExerciseSet(weight=100, reps=10),  # Warm-up (< 50% of 200)
            ExerciseSet(weight=99, reps=10),   # Warm-up
            ExerciseSet(weight=200, reps=5),   # Working set (max)
        ]
        
        processor.identify_warmups(sets)
        result = processor.select_working_sets(sets, SetPrescription.ALL)
        
        assert len(result) == 1
        assert result[0].weight == 200
    
    def test_1x_vs_3x_logic(self, processor):
        """
        CRITICAL TEST: Verify 1x vs 3x prescription logic.
        
        Given the same sets:
        - "1x" prescription should return only the top set
        - "3x" prescription should return top 3 sets
        """
        sets = [
            ExerciseSet(weight=100, reps=10),
            ExerciseSet(weight=200, reps=8),
            ExerciseSet(weight=250, reps=5),
            ExerciseSet(weight=275, reps=3),
            ExerciseSet(weight=300, reps=2),
        ]
        
        processor.identify_warmups(sets)
        
        # Test 1x
        result_1x = processor.select_working_sets(sets, SetPrescription.SINGLE)
        assert len(result_1x) == 1
        assert result_1x[0].weight == 300
        
        # Test 3x
        result_3x = processor.select_working_sets(sets, SetPrescription.TRIPLE)
        assert len(result_3x) == 3
        assert [s.weight for s in result_3x] == [300, 275, 250]
    
    def test_process_exercise_full_pipeline(self, processor):
        """Test full processing pipeline for an exercise."""
        exercise = Exercise(
            name="Squat",
            sets=[
                ExerciseSet(weight=135, reps=10),
                ExerciseSet(weight=225, reps=5),
                ExerciseSet(weight=315, reps=3),
                ExerciseSet(weight=365, reps=1),
            ],
            prescription=SetPrescription.TRIPLE
        )
        
        result = processor.process_exercise(exercise)
        
        assert len(result) == 3
        assert result[0].weight == 365
    
    def test_empty_sets(self, processor):
        """Test handling of empty sets."""
        result = processor.identify_warmups([])
        assert result == []
        
        result = processor.select_working_sets([], SetPrescription.SINGLE)
        assert result == []
    
    def test_estimated_1rm_calculation(self):
        """Test 1RM estimation using Brzycki formula."""
        # 225 lbs x 10 reps
        exercise_set = ExerciseSet(weight=225, reps=10)
        
        # Brzycki: 225 * (36 / (37 - 10)) = 225 * (36/27) = 300
        expected_1rm = 225 * (36 / 27)
        assert abs(exercise_set.estimated_1rm - expected_1rm) < 0.01
