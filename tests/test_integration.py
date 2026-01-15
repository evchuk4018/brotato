"""End-to-end integration tests for the workout tracker."""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime
import pytz
import json

from services.alias_service import AliasService
from services.workout_processor import WorkoutProcessor, Exercise, ExerciseSet, SetPrescription
from services.sheet_manager import SheetManager, BatchPayload


class TestEndToEndIntegration:
    """
    End-to-end integration tests.
    
    Tests the complete flow from mock input to batch output,
    without hitting real external APIs.
    """
    
    @pytest.fixture
    def alias_service(self):
        return AliasService()
    
    @pytest.fixture
    def workout_processor(self):
        return WorkoutProcessor()
    
    @pytest.fixture
    def sheet_manager(self):
        return SheetManager(
            sheet_name="Workouts",
            start_col="B",
            start_row=2
        )
    
    @pytest.fixture
    def mock_vision_response(self):
        """Simulates Gemini Vision extraction output."""
        return {
            "exercises": [
                {
                    "name": "Leg Press (Machine)",  # Should fuzzy match to "Leg Press"
                    "sets": [
                        {"weight": 180, "reps": 15},   # Warm-up (< 50% of 450)
                        {"weight": 270, "reps": 10},   # Working
                        {"weight": 360, "reps": 8},    # Working
                        {"weight": 450, "reps": 5},    # Top set
                    ]
                },
                {
                    "name": "DB Rows",  # Should fuzzy match to "Dumbbell Row"
                    "sets": [
                        {"weight": 50, "reps": 15},    # Warm-up (< 50% of 120)
                        {"weight": 75, "reps": 10},    # Working
                        {"weight": 100, "reps": 8},    # Working
                        {"weight": 120, "reps": 6},    # Top set
                    ]
                }
            ],
            "date": "2025-01-15",
            "notes": "3x working sets"
        }
    
    def test_full_pipeline_mock_to_batch(
        self, 
        alias_service, 
        workout_processor, 
        sheet_manager,
        mock_vision_response
    ):
        """
        CRITICAL TEST: Full pipeline from mock input to batch output.
        
        Workflow:
        1. Mock vision data comes in
        2. Fuzzy match exercise names
        3. Process sets (filter warm-ups, apply prescription)
        4. Generate batch payload for Sheets
        """
        processed_exercises = []
        
        for ex_data in mock_vision_response["exercises"]:
            # Step 1: Fuzzy match name
            raw_name = ex_data["name"]
            canonical_name = alias_service.match(raw_name)
            
            # Verify fuzzy matching works
            if raw_name == "Leg Press (Machine)":
                assert canonical_name == "Leg Press", \
                    f"Expected 'Leg Press' but got '{canonical_name}'"
            elif raw_name == "DB Rows":
                assert canonical_name == "Dumbbell Row", \
                    f"Expected 'Dumbbell Row' but got '{canonical_name}'"
            
            # Step 2: Create exercise with sets
            sets = [
                ExerciseSet(weight=s["weight"], reps=s["reps"])
                for s in ex_data["sets"]
            ]
            
            # Parse prescription from notes
            notes = mock_vision_response.get("notes", "")
            prescription = SetPrescription.TRIPLE if "3x" in notes else SetPrescription.ALL
            
            exercise = Exercise(
                name=canonical_name,
                sets=sets,
                prescription=prescription
            )
            
            # Step 3: Process exercise
            working_sets = workout_processor.process_exercise(exercise)
            
            # Convert to sheet format
            for s in working_sets:
                processed_exercises.append({
                    "name": canonical_name,
                    "weight": s.weight,
                    "reps": s.reps,
                    "sets": 1,
                    "notes": notes
                })
        
        # Step 4: Generate batch payload
        workout_date = datetime(2025, 1, 15, tzinfo=pytz.UTC)
        payload = sheet_manager.prepare_batch_payload(
            exercises=processed_exercises,
            existing_data=None,
            workout_date=workout_date
        )
        
        # Verify batch payload structure
        assert isinstance(payload, BatchPayload)
        assert payload.range_notation.startswith("Workouts!B2:")
        
        # Leg Press: 3x prescription, top 3 sets = 450, 360, 270
        # Dumbbell Row: 3x prescription, top 3 sets = 100, 75, 50
        # Total: 6 rows
        assert len(payload.values) == 6
        
        # Verify exercise names in payload
        exercise_names = [row[1] for row in payload.values]
        assert exercise_names.count("Leg Press") == 3
        assert exercise_names.count("Dumbbell Row") == 3
    
    def test_1x_prescription_pipeline(
        self, 
        alias_service, 
        workout_processor, 
        sheet_manager
    ):
        """Test pipeline with 1x prescription - should return only top sets."""
        vision_data = {
            "exercises": [
                {
                    "name": "Bench Press",
                    "sets": [
                        {"weight": 135, "reps": 10},
                        {"weight": 185, "reps": 5},
                        {"weight": 225, "reps": 3},
                        {"weight": 245, "reps": 1},
                    ]
                }
            ],
            "notes": "1x top set only"
        }
        
        processed = []
        for ex in vision_data["exercises"]:
            canonical_name = alias_service.match(ex["name"])
            sets = [ExerciseSet(weight=s["weight"], reps=s["reps"]) for s in ex["sets"]]
            
            exercise = Exercise(
                name=canonical_name,
                sets=sets,
                prescription=SetPrescription.SINGLE  # 1x
            )
            
            working_sets = workout_processor.process_exercise(exercise)
            for s in working_sets:
                processed.append({
                    "name": canonical_name,
                    "weight": s.weight,
                    "reps": s.reps,
                    "sets": 1,
                    "notes": ""
                })
        
        payload = sheet_manager.prepare_batch_payload(exercises=processed)
        
        # Should only have 1 row (top set)
        assert len(payload.values) == 1
        assert payload.values[0][2] == 245  # Weight of top set
    
    def test_warmup_filtering_pipeline(
        self, 
        alias_service, 
        workout_processor, 
        sheet_manager
    ):
        """Test that warm-ups are properly filtered throughout pipeline."""
        vision_data = {
            "exercises": [
                {
                    "name": "Squat",
                    "sets": [
                        {"weight": 95, "reps": 15},   # Warm-up (< 50% of 315)
                        {"weight": 135, "reps": 10},  # Warm-up
                        {"weight": 155, "reps": 8},   # Warm-up (155 < 157.5)
                        {"weight": 225, "reps": 5},   # Working
                        {"weight": 275, "reps": 3},   # Working
                        {"weight": 315, "reps": 1},   # Top set
                    ]
                }
            ],
            "notes": ""
        }
        
        processed = []
        for ex in vision_data["exercises"]:
            canonical_name = alias_service.match(ex["name"]) or ex["name"]
            sets = [ExerciseSet(weight=s["weight"], reps=s["reps"]) for s in ex["sets"]]
            
            exercise = Exercise(
                name=canonical_name,
                sets=sets,
                prescription=SetPrescription.ALL
            )
            
            working_sets = workout_processor.process_exercise(exercise)
            for s in working_sets:
                processed.append({
                    "name": canonical_name,
                    "weight": s.weight,
                    "reps": s.reps,
                    "sets": 1,
                    "notes": ""
                })
        
        payload = sheet_manager.prepare_batch_payload(exercises=processed)
        
        # Should have 3 working sets (225, 275, 315)
        # 50% of 315 = 157.5, so 95, 135, 155 are all warm-ups
        assert len(payload.values) == 3
        weights = [row[2] for row in payload.values]
        assert 95 not in weights
        assert 135 not in weights
        assert 155 not in weights
        assert 225 in weights
        assert 275 in weights
        assert 315 in weights
    
    def test_batch_policy_read_once_write_once(self, sheet_manager):
        """
        Verify the batch policy is enforced:
        - Read Once: Single fetch of existing data
        - Write Once: Single batch update
        """
        # Mock gspread
        mock_worksheet = MagicMock()
        mock_worksheet.get.return_value = [
            ["2025-01-14", "Squat", 300, 5, 3, "", 340],
            ["2025-01-14", "Bench", 200, 8, 3, "", 240],
        ]
        
        mock_spreadsheet = MagicMock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_spreadsheet
        
        sheet_manager.set_client(mock_client, "test-id")
        
        # Read block (should be called once)
        existing_data = sheet_manager.read_block()
        
        # Prepare batch
        exercises = [
            {"name": "Deadlift", "weight": 405, "reps": 3, "sets": 1, "notes": ""}
        ]
        payload = sheet_manager.prepare_batch_payload(
            exercises=exercises,
            existing_data=existing_data
        )
        
        # Execute batch update (should be called once)
        sheet_manager.execute_batch_update(payload)
        
        # Verify: get() called once (Read Once)
        mock_worksheet.get.assert_called_once()
        
        # Verify: update() called once (Write Once)
        mock_worksheet.update.assert_called_once()
    
    def test_complete_flow_with_mocks(self):
        """
        Full end-to-end test simulating the complete webhook flow.
        """
        # Simulate incoming data
        mock_vision_output = {
            "exercises": [
                {
                    "name": "Romanian DL",  # -> "Romanian Deadlift"
                    "sets": [
                        {"weight": 135, "reps": 12},
                        {"weight": 185, "reps": 10},
                        {"weight": 225, "reps": 8},
                        {"weight": 275, "reps": 6},
                    ]
                }
            ],
            "notes": "3x working sets"
        }
        
        # Initialize services
        alias_svc = AliasService()
        processor = WorkoutProcessor()
        sheet_mgr = SheetManager()
        
        # Mock sheets client
        mock_ws = MagicMock()
        mock_ws.get.return_value = []
        mock_ss = MagicMock()
        mock_ss.worksheet.return_value = mock_ws
        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_ss
        sheet_mgr.set_client(mock_client, "test-sheet-id")
        
        # Process pipeline
        final_exercises = []
        for ex in mock_vision_output["exercises"]:
            name = alias_svc.match(ex["name"])
            assert name == "Romanian Deadlift"
            
            sets = [ExerciseSet(weight=s["weight"], reps=s["reps"]) for s in ex["sets"]]
            exercise = Exercise(name=name, sets=sets, prescription=SetPrescription.TRIPLE)
            working = processor.process_exercise(exercise)
            
            for s in working:
                final_exercises.append({
                    "name": name,
                    "weight": s.weight,
                    "reps": s.reps,
                    "sets": 1,
                    "notes": ""
                })
        
        # Generate and execute batch
        existing = sheet_mgr.read_block()
        payload = sheet_mgr.prepare_batch_payload(
            exercises=final_exercises,
            existing_data=existing
        )
        result = sheet_mgr.execute_batch_update(payload)
        
        # Assertions
        assert result["updated_rows"] == 3  # Top 3 sets
        mock_ws.update.assert_called_once()
        
        # Verify payload structure
        call_args = mock_ws.update.call_args
        updated_range = call_args[0][0]
        updated_values = call_args[0][1]
        
        assert updated_range == "Workouts!B2:H4"
        assert len(updated_values) == 3
        
        # All exercises should be "Romanian Deadlift"
        for row in updated_values:
            assert row[1] == "Romanian Deadlift"
