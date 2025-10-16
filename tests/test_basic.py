def test_simple():
    """A simple test to verify testing setup"""
    assert True

def test_list_operations():
    """Test basic list operations"""
    test_list = [1, 2, 3]
    assert len(test_list) == 3
    assert sum(test_list) == 6

def test_string_operations():
    """Test basic string operations"""
    test_str = "  hello  "
    assert test_str.strip() == "hello"
    assert test_str.split() == ["hello"]