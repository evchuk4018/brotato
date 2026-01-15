"""
SheetManager: Google Sheets batch operations.
Implements the "Read Once, Write Once" batch policy for data integrity.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pytz


@dataclass
class CellUpdate:
    """Represents a single cell update."""
    row: int
    col: int
    value: Any


@dataclass
class BatchPayload:
    """
    Batch update payload for Google Sheets.
    
    Contains the range and values to update in a single API call.
    """
    range_notation: str  # e.g., "Sheet1!B2:H20"
    values: List[List[Any]]  # 2D array of values


class SheetManager:
    """
    Manages Google Sheets operations with batch updates.
    
    Batch Policy (Critical):
    - Read Once: Fetch entire block (e.g., B2:H20) in single call
    - Write Once: Push all updates in single batch_update call
    
    This ensures data integrity and minimizes API quota usage.
    """
    
    # Column mapping for workout data
    COLUMNS = {
        "date": 0,        # Column B
        "exercise": 1,    # Column C  
        "weight": 2,      # Column D
        "reps": 3,        # Column E
        "sets": 4,        # Column F
        "notes": 5,       # Column G
        "estimated_1rm": 6  # Column H
    }
    
    def __init__(
        self, 
        sheet_name: str = "Workouts",
        start_col: str = "B",
        start_row: int = 2,
        timezone: str = "America/New_York"
    ):
        """
        Initialize SheetManager.
        
        Args:
            sheet_name: Name of the worksheet.
            start_col: Starting column letter.
            start_row: Starting row number.
            timezone: Timezone for date formatting.
        """
        self.sheet_name = sheet_name
        self.start_col = start_col
        self.start_row = start_row
        self.timezone = pytz.timezone(timezone)
        self._client = None  # gspread client, set externally
        self._worksheet = None
    
    def set_client(self, client: Any, spreadsheet_id: str) -> None:
        """
        Set the gspread client and open worksheet.
        
        Args:
            client: Authenticated gspread client.
            spreadsheet_id: Google Sheets document ID.
        """
        self._client = client
        spreadsheet = client.open_by_key(spreadsheet_id)
        self._worksheet = spreadsheet.worksheet(self.sheet_name)
    
    def read_block(self, end_row: int = 100) -> List[List[Any]]:
        """
        Read entire data block in single API call.
        
        Args:
            end_row: Last row to read.
            
        Returns:
            2D array of cell values.
        """
        if not self._worksheet:
            raise RuntimeError("Worksheet not initialized. Call set_client first.")
        
        range_notation = f"{self.start_col}{self.start_row}:H{end_row}"
        return self._worksheet.get(range_notation)
    
    def find_next_empty_row(self, data: List[List[Any]]) -> int:
        """
        Find the next empty row from existing data.
        
        Args:
            data: Existing sheet data from read_block.
            
        Returns:
            Row number for next entry.
        """
        for i, row in enumerate(data):
            if not row or all(cell == "" or cell is None for cell in row):
                return self.start_row + i
        return self.start_row + len(data)
    
    def prepare_batch_payload(
        self,
        exercises: List[Dict[str, Any]],
        existing_data: Optional[List[List[Any]]] = None,
        workout_date: Optional[datetime] = None
    ) -> BatchPayload:
        """
        Prepare batch update payload for exercises.
        
        Args:
            exercises: List of exercise dictionaries with keys:
                - name: Exercise name
                - weight: Weight used
                - reps: Reps performed  
                - sets: Number of sets
                - notes: Optional notes
            existing_data: Data from read_block (to find next row).
            workout_date: Date of workout (defaults to now).
            
        Returns:
            BatchPayload with range and values for batch_update.
        """
        if workout_date is None:
            workout_date = datetime.now(self.timezone)
        
        date_str = workout_date.strftime("%Y-%m-%d")
        
        # Determine starting row
        if existing_data:
            start_row = self.find_next_empty_row(existing_data)
        else:
            start_row = self.start_row
        
        # Build values array
        values = []
        for exercise in exercises:
            weight = exercise.get("weight", 0)
            reps = exercise.get("reps", 0)
            
            # Calculate estimated 1RM (Brzycki formula)
            if reps > 0 and reps < 37:
                estimated_1rm = round(weight * (36 / (37 - reps)), 1)
            else:
                estimated_1rm = weight
            
            row = [
                date_str,
                exercise.get("name", ""),
                weight,
                reps,
                exercise.get("sets", 1),
                exercise.get("notes", ""),
                estimated_1rm
            ]
            values.append(row)
        
        # Calculate end row and column
        end_row = start_row + len(exercises) - 1
        end_col = chr(ord(self.start_col) + len(self.COLUMNS) - 1)  # H
        
        range_notation = f"{self.sheet_name}!{self.start_col}{start_row}:{end_col}{end_row}"
        
        return BatchPayload(
            range_notation=range_notation,
            values=values
        )
    
    def execute_batch_update(self, payload: BatchPayload) -> Dict[str, Any]:
        """
        Execute batch update to Google Sheets.
        
        Args:
            payload: BatchPayload with range and values.
            
        Returns:
            API response dict.
        """
        if not self._worksheet:
            raise RuntimeError("Worksheet not initialized. Call set_client first.")
        
        # Use gspread's update method with batch semantics
        self._worksheet.update(
            payload.range_notation,
            payload.values,
            value_input_option="USER_ENTERED"
        )
        
        return {
            "updated_range": payload.range_notation,
            "updated_rows": len(payload.values)
        }
