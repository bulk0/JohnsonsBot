import pytest
from bot import parse_variable_list, match_variables_case_insensitive

@pytest.mark.unit
def test_parse_variable_list():
    """Test variable list parsing functionality"""
    # Test empty input
    assert parse_variable_list("") == []
    assert parse_variable_list("   ") == []
    
    # Test single variable
    assert parse_variable_list("var1") == ["var1"]
    
    # Test multiple variables
    assert parse_variable_list("var1, var2, var3") == ["var1", "var2", "var3"]
    
    # Test with extra spaces
    assert parse_variable_list(" var1 ,  var2 , var3 ") == ["var1", "var2", "var3"]
    
    # Test with empty elements
    assert parse_variable_list("var1,,var2") == ["var1", "var2"]
    
    # Test with quotes
    assert parse_variable_list("'var1', \"var2\"") == ["var1", "var2"]

@pytest.mark.unit
def test_match_variables_case_insensitive():
    """Test case-insensitive variable matching"""
    available_vars = ["Var1", "VAR2", "var3"]
    
    # Test exact match
    matched, invalid = match_variables_case_insensitive(["Var1"], available_vars)
    assert matched == ["Var1"]
    assert invalid == []
    
    # Test case-insensitive match
    matched, invalid = match_variables_case_insensitive(["var1", "VAR2"], available_vars)
    assert matched == ["Var1", "VAR2"]
    assert invalid == []
    
    # Test invalid variables
    matched, invalid = match_variables_case_insensitive(["var1", "nonexistent"], available_vars)
    assert matched == ["Var1"]
    assert invalid == ["nonexistent"]
    
    # Test empty input
    matched, invalid = match_variables_case_insensitive([], available_vars)
    assert matched == []
    assert invalid == []