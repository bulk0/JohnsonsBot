def test_simple():
    """A simple test to verify testing setup"""
    assert True

def test_parse_variable_list():
    """Test basic variable list parsing"""
    from bot import parse_variable_list
    
    # Test empty input
    assert parse_variable_list("") == []
    
    # Test single variable
    assert parse_variable_list("var1") == ["var1"]