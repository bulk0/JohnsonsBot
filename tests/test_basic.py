import pytest
from bot import parse_variable_list, match_variables_case_insensitive

def test_parse_variable_list():
    # Test empty input
    assert parse_variable_list("") == []
    
    # Test single variable
    assert parse_variable_list("var1") == ["var1"]
    
    # Test multiple variables
    assert parse_variable_list("var1, var2, var3") == ["var1", "var2", "var3"]
    
    # Test with extra spaces
    assert parse_variable_list(" var1 ,  var2 , var3 ") == ["var1", "var2", "var3"]
    
    # Test with empty elements
    assert parse_variable_list("var1,,var2") == ["var1", "var2"]

def test_match_variables_case_insensitive():
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
