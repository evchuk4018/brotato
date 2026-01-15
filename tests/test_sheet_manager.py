"""Tests for SheetManager batch operations."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
import pytz

from services.sheet_manager import SheetManager, BatchPayload


class TestSheetManager:
    """Test suite for SheetManager."""
    
    @pytest.fixture
    def sheet_manager(self):
        """Create SheetManager instance."""
        return SheetManager(
            sheet_name="Workouts",
            start_col="B",
            start_row=2,
            timezone="America/New_York"
        )
    
    @pytest.fixture
    def sample_exercises(self):
        """Sample exercises for testing."""
        return [
            {
                "name": "Squat",
                "weight": 315,
                "reps": 5,
                "sets": 3,
                "notes": "Felt strong"
            },
            {
                "name": "Bench Press",
                "weight": 225,
                "reps": 8,
                "sets": 3,
                "notes": ""
            },
            {
                "name": "Deadlift",
                "weight": 405,
                "reps": 3,
                "sets": 1,
                "notes": "PR attempt"
            }
        ]
    
    def test_prepare_batch_payload_structure(self, sheet_manager, sample_exercises):
        """
        REQUIREMENT: Verify batch payload has correct GridRange and Value structure.
        
        Mock the gspread response - do not hit real API.
        """
        workout_date = datetime(2025, 1, 15, 10, 0, 0, tzinfo=pytz.UTC)
        
        # No existing data - start from row 2
        payload = sheet_manager.prepare_batch_payload(
            exercises=sample_exercises,
            existing_data=None,
            workout_date=workout_date
        )
        
        # Verify payload is BatchPayload type
        assert isinstance(payload, BatchPayload)
        
        # Verify range notation
        assert payload.range_notation == "Workouts!B2:H4"
        
        # Verify values structure
        assert len(payload.values) == 3  # 3 exercises
        assert len(payload.values[0]) == 7  # 7 columns (B-H)
    
    def test_prepare_batch_payload_values(self, sheet_manager, sample_exercises):
        """Test that payload values are correctly formatted."""
        workout_date = datetime(2025, 1, 15, tzinfo=pytz.UTC)
        
        payload = sheet_manager.prepare_batch_payload(
            exercises=sample_exercises,
            existing_data=None,
            workout_date=workout_date
        )
        
        # Check first row (Squat)
        squat_row = payload.values[0]
        assert squat_row[0] == "2025-01-15"  # Date
        assert squat_row[1] == "Squat"        # Exercise
        assert squat_row[2] == 315            # Weight
        assert squat_row[3] == 5              # Reps
        assert squat_row[4] == 3              # Sets
        assert squat_row[5] == "Felt strong"  # Notes
        
        # Check 1RM calculation: 315 * (36 / (37-5)) = 354.375
        expected_1rm = round(315 * (36 / 32), 1)
        assert squat_row[6] == expected_1rm
    
    def test_prepare_batch_payload_with_existing_data(self, sheet_manager, sample_exercises):
        """Test that payload starts after existing data."""
        existing_data = [
            ["2025-01-14", "Squat", 300, 5, 3, "", 337.5],
            ["2025-01-14", "Bench", 200, 8, 3, "", 246.6],
            [],  # Empty row - this is where new data should start
        ]
        
        workout_date = datetime(2025, 1, 15, tzinfo=pytz.UTC)
        
        payload = sheet_manager.prepare_batch_payload(
            exercises=sample_exercises,
            existing_data=existing_data,
            workout_date=workout_date
        )
        
        # Should start at row 4 (start_row=2 + 2 existing rows)
        assert payload.range_notation == "Workouts!B4:H6"
    
    def test_find_next_empty_row(self, sheet_manager):
        """Test finding the next empty row in existing data."""
        existing_data = [
            ["2025-01-14", "Squat", 300, 5, 3, "", 337.5],
            ["2025-01-14", "Bench", 200, 8, 3, "", 246.6],
        ]
        
        next_row = sheet_manager.find_next_empty_row(existing_data)
        assert next_row == 4  # start_row (2) + len(data) (2)
    
    def test_find_next_empty_row_with_gap(self, sheet_manager):
        """Test finding empty row when there's a gap in data."""
        existing_data = [
            ["2025-01-14", "Squat", 300, 5, 3, "", 337.5],
            [],  # Empty row
            ["2025-01-14", "Bench", 200, 8, 3, "", 246.6],
        ]
        
        next_row = sheet_manager.find_next_empty_row(existing_data)
        assert next_row == 3  # start_row (2) + 1 (first empty)
    
    def test_1rm_calculation_in_payload(self, sheet_manager):
        """Test 1RM calculation using Brzycki formula in payload."""
        exercises = [
            {"name": "Test", "weight": 225, "reps": 10, "sets": 1, "notes": ""}
        ]
        
        payload = sheet_manager.prepare_batch_payload(exercises=exercises)
        
        # Brzycki: 225 * (36 / (37-10)) = 225 * (36/27) = 300
        expected_1rm = round(225 * (36 / 27), 1)
        assert payload.values[0][6] == expected_1rm
    
    def test_1rm_edge_case_high_reps(self, sheet_manager):
        """Test 1RM when reps >= 37 (formula breaks down)."""
        exercises = [
            {"name": "Test", "weight": 100, "reps": 40, "sets": 1, "notes": ""}
        ]
        
        payload = sheet_manager.prepare_batch_payload(exercises=exercises)
        
        # Should return weight itself when reps >= 37
        assert payload.values[0][6] == 100
    
    def test_batch_payload_for_mock_gspread(self, sheet_manager, sample_exercises):
        """
        Test that payload can be used with mocked gspread client.
        
        This simulates the actual usage pattern without hitting the API.
        """
        # Mock gspread client and worksheet
        mock_worksheet = MagicMock()
        mock_spreadsheet = MagicMock()
        mock_spreadsheet.worksheet.return_value = mock_worksheet
        mock_client = MagicMock()
        mock_client.open_by_key.return_value = mock_spreadsheet
        
        # Set up the client
        sheet_manager.set_client(mock_client, "test-spreadsheet-id")
        
        # Prepare payload
        payload = sheet_manager.prepare_batch_payload(
            exercises=sample_exercises,
            existing_data=None
        )
        
        # Execute batch update
        result = sheet_manager.execute_batch_update(payload)
        
        # Verify the mock was called correctly
        mock_worksheet.update.assert_called_once_with(
            payload.range_notation,
            payload.values,
            value_input_option="USER_ENTERED"
        )
        
        # Verify result
        assert result["updated_range"] == payload.range_notation
        assert result["updated_rows"] == 3


class TestBatchPayload:
    """Tests for BatchPayload dataclass."""
    
    def test_batch_payload_creation(self):
        """Test BatchPayload can be created with required fields."""
        payload = BatchPayload(
            range_notation="Sheet1!A1:B2",
            values=[["a", "b"], ["c", "d"]]
        )
        
        assert payload.range_notation == "Sheet1!A1:B2"
        assert payload.values == [["a", "b"], ["c", "d"]]
