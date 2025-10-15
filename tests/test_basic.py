def test_simple():
    """A simple test to verify testing setup"""
    assert True

def test_string_operations():
    """Test basic string operations that don't require external dependencies"""
    # Test string strip
    assert " test ".strip() == "test"
    
    # Test string split
    assert "a,b,c".split(",") == ["a", "b", "c"]