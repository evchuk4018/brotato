#!/usr/bin/env python
"""Quick sanity check for imports."""

try:
    from services.alias_service import AliasService
    print("✓ AliasService imported")
    
    from services.workout_processor import WorkoutProcessor
    print("✓ WorkoutProcessor imported")
    
    from services.sheet_manager import SheetManager
    print("✓ SheetManager imported")
    
    # Try to instantiate
    alias_svc = AliasService()
    print("✓ AliasService instantiated")
    
    processor = WorkoutProcessor()
    print("✓ WorkoutProcessor instantiated")
    
    sheet_mgr = SheetManager()
    print("✓ SheetManager instantiated")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
