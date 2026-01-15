"""
Snap-to-Sheet Workout Tracker v1.1
Serverless workout logger that syncs screenshots to Google Sheets using Gemini Vision.
"""

import os
import functions_framework
from flask import Request, jsonify
import gspread
from google.oauth2.service_account import Credentials

from services.alias_service import AliasService
from services.workout_processor import WorkoutProcessor, Exercise, ExerciseSet, SetPrescription
from services.sheet_manager import SheetManager
from services.vision_service import VisionService
from services.telegram_service import TelegramService


# Initialize services (singleton pattern for Cloud Functions)
alias_service = AliasService()
workout_processor = WorkoutProcessor()
sheet_manager = SheetManager()


def get_prescription_from_notes(notes: str) -> SetPrescription:
    """
    Parse prescription from notes string.
    
    Args:
        notes: Notes string that may contain "1x" or "3x".
        
    Returns:
        Appropriate SetPrescription enum value.
    """
    if notes:
        notes_lower = notes.lower()
        if "1x" in notes_lower or "single" in notes_lower:
            return SetPrescription.SINGLE
        elif "3x" in notes_lower or "triple" in notes_lower:
            return SetPrescription.TRIPLE
    return SetPrescription.ALL


def process_vision_data(vision_data: dict) -> list:
    """
    Process vision extraction data through the workout pipeline.
    
    Args:
        vision_data: Dict from VisionService with exercises.
        
    Returns:
        List of exercise dicts ready for SheetManager.
    """
    processed_exercises = []
    
    for ex_data in vision_data.get("exercises", []):
        raw_name = ex_data.get("name", "")
        
        # Fuzzy match to canonical name
        canonical_name = alias_service.match(raw_name)
        if not canonical_name:
            canonical_name = raw_name  # Use original if no match
        
        # Convert sets to ExerciseSet objects
        sets = [
            ExerciseSet(weight=s.get("weight", 0), reps=s.get("reps", 0))
            for s in ex_data.get("sets", [])
        ]
        
        if not sets:
            continue
        
        # Determine prescription from notes
        notes = ex_data.get("notes", "") or vision_data.get("notes", "")
        prescription = get_prescription_from_notes(notes)
        
        # Create Exercise and process
        exercise = Exercise(name=canonical_name, sets=sets, prescription=prescription)
        processed_sets = workout_processor.process_exercise(exercise)
        
        # Convert processed sets to sheet format
        for s in processed_sets:
            processed_exercises.append({
                "name": canonical_name,
                "weight": s.weight,
                "reps": s.reps,
                "sets": 1,
                "notes": notes
            })
    
    return processed_exercises


@functions_framework.http
def handle_telegram_webhook(request: Request):
    """
    Main entry point for Telegram webhook.
    
    Workflow:
    1. Receive photo from Telegram
    2. Extract workout data via Gemini Vision
    3. Fuzzy match exercises to canonical names
    4. Apply smart selection logic (warm-up filtering, set selection)
    5. Batch update Google Sheets (Read Once, Write Once)
    """
    try:
        # Parse incoming Telegram update
        update = request.get_json(silent=True)
        
        if not update:
            return jsonify({"status": "error", "message": "No payload"}), 400
        
        # Initialize Telegram service
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            return jsonify({"status": "error", "message": "Bot token not configured"}), 500
        
        telegram = TelegramService(bot_token)
        file_id, chat_id, text = telegram.parse_update(update)
        
        if not file_id:
            # No photo in message
            return jsonify({"status": "ok", "message": "No photo to process"}), 200
        
        # Get photo bytes (sync wrapper for Cloud Functions)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        image_bytes = loop.run_until_complete(telegram.get_file_bytes(file_id))
        
        # Extract workout data via Gemini Vision
        gemini_key = os.environ.get("GOOGLE_API_KEY")
        vision = VisionService(api_key=gemini_key)
        vision_data = vision.extract_workout_data(image_bytes)
        
        # Process through workout pipeline
        exercises = process_vision_data(vision_data)
        
        if not exercises:
            loop.run_until_complete(telegram.send_message(
                chat_id, "No exercises found in screenshot."
            ))
            return jsonify({"status": "ok", "message": "No exercises extracted"}), 200
        
        # Setup Google Sheets client
        spreadsheet_id = os.environ.get("GOOGLE_SHEET_ID")
        creds_json = os.environ.get("GOOGLE_CREDENTIALS")
        
        if spreadsheet_id and creds_json:
            import json
            creds_dict = json.loads(creds_json)
            creds = Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            client = gspread.authorize(creds)
            sheet_manager.set_client(client, spreadsheet_id)
            
            # Batch Policy: Read Once, Write Once
            existing_data = sheet_manager.read_block()
            payload = sheet_manager.prepare_batch_payload(
                exercises=exercises,
                existing_data=existing_data
            )
            result = sheet_manager.execute_batch_update(payload)
            
            # Send confirmation
            msg = f"✅ Logged {len(exercises)} exercise(s) to sheet!"
            loop.run_until_complete(telegram.send_message(chat_id, msg))
            
            return jsonify({
                "status": "ok", 
                "updated": result["updated_rows"]
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Google Sheets not configured"
            }), 500
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
