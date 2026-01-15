"""Tests for AliasService fuzzy matching."""

import pytest
from services.alias_service import AliasService


class TestAliasService:
    """Test suite for AliasService."""
    
    @pytest.fixture
    def alias_service(self):
        """Create AliasService instance with default map."""
        return AliasService()
    
    def test_exact_match_canonical(self, alias_service):
        """Test exact match of canonical name."""
        result = alias_service.match("Leg Press")
        assert result == "Leg Press"
    
    def test_exact_match_alias(self, alias_service):
        """Test exact match of an alias."""
        result = alias_service.match("Leg Press (Machine)")
        assert result == "Leg Press"
    
    def test_fuzzy_match_leg_press_machine(self, alias_service):
        """
        REQUIREMENT: "Leg Press (Machine)" must map to "Leg Press".
        This is a critical test for the alias service.
        """
        result = alias_service.match("Leg Press (Machine)")
        assert result == "Leg Press", (
            f"Expected 'Leg Press (Machine)' to map to 'Leg Press', got '{result}'"
        )
    
    def test_fuzzy_match_case_insensitive(self, alias_service):
        """Test that matching is case insensitive."""
        result = alias_service.match("leg press (machine)")
        assert result == "Leg Press"
        
        result = alias_service.match("LEG PRESS")
        assert result == "Leg Press"
    
    def test_fuzzy_match_abbreviation(self, alias_service):
        """Test matching abbreviations."""
        result = alias_service.match("DB Row")
        assert result == "Dumbbell Row"
        
        result = alias_service.match("RDL")
        assert result == "Romanian Deadlift"
    
    def test_fuzzy_match_with_typo(self, alias_service):
        """Test fuzzy matching handles minor typos."""
        result = alias_service.match("Bech Press")  # typo
        assert result == "Bench Press"
    
    def test_no_match_below_threshold(self, alias_service):
        """Test that low-quality matches return None."""
        result = alias_service.match("Jumping Jacks")
        assert result is None
    
    def test_no_match_empty_input(self, alias_service):
        """Test empty input returns None."""
        result = alias_service.match("")
        assert result is None
        
        result = alias_service.match(None)
        assert result is None
    
    def test_custom_alias_map(self):
        """Test using a custom alias map."""
        custom_map = {
            "Custom Exercise": ["CE", "Custom Ex"],
        }
        service = AliasService(alias_map=custom_map)
        
        result = service.match("CE")
        assert result == "Custom Exercise"
    
    def test_get_all_canonical_names(self, alias_service):
        """Test retrieving all canonical names."""
        names = alias_service.get_all_canonical_names()
        assert "Leg Press" in names
        assert "Bench Press" in names
        assert "Squat" in names
