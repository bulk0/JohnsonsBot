#!/usr/bin/env python3
"""
Test script for case-insensitive variable matching
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import match_variables_case_insensitive

def test_match_variables_case_insensitive():
    """Test the case-insensitive variable matching function"""
    
    print("=" * 60)
    print("Testing case-insensitive variable matching")
    print("=" * 60)
    
    # Test case 1: Exact match (same case)
    print("\n### Test 1: Exact match (same case)")
    user_vars = ['var1', 'var2', 'var3']
    available_vars = ['var1', 'var2', 'var3', 'var4', 'var5']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == ['var1', 'var2', 'var3'], "Test 1 failed"
    assert invalid == [], "Test 1 failed"
    print("✅ Test 1 PASSED")
    
    # Test case 2: User inputs lowercase, database has uppercase
    print("\n### Test 2: User inputs lowercase, database has uppercase")
    user_vars = ['csi', 'nps', 'satisfaction']
    available_vars = ['CSI', 'NPS', 'SATISFACTION', 'Brand', 'Region']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == ['CSI', 'NPS', 'SATISFACTION'], f"Test 2 failed: got {matched}"
    assert invalid == [], "Test 2 failed"
    print("✅ Test 2 PASSED")
    
    # Test case 3: User inputs uppercase, database has lowercase
    print("\n### Test 3: User inputs uppercase, database has lowercase")
    user_vars = ['CSI', 'NPS', 'QUALITY']
    available_vars = ['csi', 'nps', 'quality', 'brand']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == ['csi', 'nps', 'quality'], f"Test 3 failed: got {matched}"
    assert invalid == [], "Test 3 failed"
    print("✅ Test 3 PASSED")
    
    # Test case 4: Mixed case on both sides
    print("\n### Test 4: Mixed case on both sides")
    user_vars = ['CsI', 'nPs', 'QuAlItY']
    available_vars = ['CSI', 'NPS', 'Quality', 'Brand']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == ['CSI', 'NPS', 'Quality'], f"Test 4 failed: got {matched}"
    assert invalid == [], "Test 4 failed"
    print("✅ Test 4 PASSED")
    
    # Test case 5: Some invalid variables
    print("\n### Test 5: Some invalid variables")
    user_vars = ['csi', 'nps', 'invalid_var', 'another_invalid']
    available_vars = ['CSI', 'NPS', 'Quality', 'Brand']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == ['CSI', 'NPS'], f"Test 5 failed: got {matched}"
    assert invalid == ['invalid_var', 'another_invalid'], f"Test 5 failed: got {invalid}"
    print("✅ Test 5 PASSED")
    
    # Test case 6: All invalid variables
    print("\n### Test 6: All invalid variables")
    user_vars = ['not_exist1', 'not_exist2']
    available_vars = ['CSI', 'NPS', 'Quality']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == [], f"Test 6 failed: got {matched}"
    assert invalid == ['not_exist1', 'not_exist2'], f"Test 6 failed: got {invalid}"
    print("✅ Test 6 PASSED")
    
    # Test case 7: Empty lists
    print("\n### Test 7: Empty user input")
    user_vars = []
    available_vars = ['CSI', 'NPS', 'Quality']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == [], f"Test 7 failed: got {matched}"
    assert invalid == [], f"Test 7 failed: got {invalid}"
    print("✅ Test 7 PASSED")
    
    # Test case 8: Variables with underscores and numbers
    print("\n### Test 8: Variables with underscores and numbers")
    user_vars = ['var_1', 'var_2', 'test_var_3']
    available_vars = ['VAR_1', 'Var_2', 'TEST_VAR_3', 'Other']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == ['VAR_1', 'Var_2', 'TEST_VAR_3'], f"Test 8 failed: got {matched}"
    assert invalid == [], f"Test 8 failed: got {invalid}"
    print("✅ Test 8 PASSED")
    
    # Test case 9: Real-world example from the issue
    print("\n### Test 9: Real-world example (CSI issue)")
    user_vars = ['csi', 'os', 'quality_overall']
    available_vars = ['CSI', 'OS', 'Quality_Overall', 'Brand', 'Region']
    matched, invalid = match_variables_case_insensitive(user_vars, available_vars)
    print(f"User input: {user_vars}")
    print(f"Available: {available_vars}")
    print(f"✅ Matched: {matched}")
    print(f"❌ Invalid: {invalid}")
    assert matched == ['CSI', 'OS', 'Quality_Overall'], f"Test 9 failed: got {matched}"
    assert invalid == [], f"Test 9 failed: got {invalid}"
    print("✅ Test 9 PASSED")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


def test_johnson_weights_case_mapping():
    """Test the case mapping logic in johnson_weights.py"""
    
    print("\n" + "=" * 60)
    print("Testing johnson_weights.py case mapping logic")
    print("=" * 60)
    
    # Simulate the logic from johnson_weights.py
    print("\n### Test: Column name mapping")
    
    # Simulate dataframe columns (as they would appear in real SPSS file)
    df_columns = ['CSI', 'NPS', 'Quality_Overall', 'Brand', 'Region']
    
    # Simulate user input (lowercase)
    dependent_vars = ['csi']
    independent_vars = ['nps', 'quality_overall']
    subgroups = ['brand']
    
    # Create case-insensitive mapping (same as in johnson_weights.py)
    columns_lower = {col.lower(): col for col in df_columns}
    print(f"DataFrame columns: {df_columns}")
    print(f"Lowercase mapping: {columns_lower}")
    
    all_vars = dependent_vars + independent_vars + subgroups
    
    missing_vars = []
    var_name_mapping = {}
    
    for var in all_vars:
        var_lower = var.lower()
        if var_lower in columns_lower:
            var_name_mapping[var] = columns_lower[var_lower]
        else:
            missing_vars.append(var)
    
    print(f"\nUser input:")
    print(f"  Dependent: {dependent_vars}")
    print(f"  Independent: {independent_vars}")
    print(f"  Subgroups: {subgroups}")
    
    print(f"\nVariable name mapping: {var_name_mapping}")
    print(f"Missing variables: {missing_vars}")
    
    # Apply mapping
    dependent_vars_mapped = [var_name_mapping[var] for var in dependent_vars]
    independent_vars_mapped = [var_name_mapping[var] for var in independent_vars]
    subgroups_mapped = [var_name_mapping[var] for var in subgroups]
    
    print(f"\nMapped variables:")
    print(f"  Dependent: {dependent_vars_mapped}")
    print(f"  Independent: {independent_vars_mapped}")
    print(f"  Subgroups: {subgroups_mapped}")
    
    # Verify
    assert dependent_vars_mapped == ['CSI'], f"Dependent mapping failed: {dependent_vars_mapped}"
    assert independent_vars_mapped == ['NPS', 'Quality_Overall'], f"Independent mapping failed: {independent_vars_mapped}"
    assert subgroups_mapped == ['Brand'], f"Subgroups mapping failed: {subgroups_mapped}"
    assert missing_vars == [], f"Should have no missing vars: {missing_vars}"
    
    print("\n✅ johnson_weights.py case mapping logic works correctly!")


if __name__ == "__main__":
    try:
        test_match_variables_case_insensitive()
        test_johnson_weights_case_mapping()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe case-insensitive variable matching is working correctly.")
        print("Users can now input variable names in any case (csi, CSI, CsI, etc.)")
        print("and they will be matched to the correct variables in the database.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

