import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from weights_handler import WeightsCalculationHandler
import pandas as pd
import pyreadstat

def test_file(file_path: str) -> dict:
    """
    Test file handling for a single SPSS file
    
    Args:
        file_path (str): Path to SPSS file
        
    Returns:
        dict: Test results
    """
    print(f"\nTesting file: {os.path.basename(file_path)}")
    print("-" * 50)
    
    results = {
        "file_name": os.path.basename(file_path),
        "tests": [],
        "errors": []
    }
    
    # Initialize handler
    handler = WeightsCalculationHandler()
    
    # Test 1: Basic file validation
    is_valid, error_msg = handler.validate_input_file(file_path)
    results["tests"].append({
        "name": "File validation",
        "passed": is_valid,
        "message": error_msg if not is_valid else "File is valid"
    })
    print(f"File validation: {'✅' if is_valid else '❌'} {error_msg if not is_valid else ''}")
    
    if not is_valid:
        return results
    
    try:
        # Test 2: Variable detection
        available_vars = handler.get_available_variables(file_path)
        if "error" in available_vars:
            results["tests"].append({
                "name": "Variable detection",
                "passed": False,
                "message": available_vars["error"]
            })
            print(f"Variable detection: ❌ {available_vars['error']}")
            return results
            
        results["tests"].append({
            "name": "Variable detection",
            "passed": True,
            "message": f"Found {len(available_vars['numeric'])} numeric and {len(available_vars['categorical'])} categorical variables"
        })
        print(f"Variable detection: ✅ Found {len(available_vars['numeric'])} numeric and {len(available_vars['categorical'])} categorical variables")
        
        # Test 3: Metadata handling
        if available_vars['metadata']:
            results["tests"].append({
                "name": "Metadata detection",
                "passed": True,
                "message": f"Detected {len(available_vars['metadata'])} metadata columns"
            })
            print(f"Metadata detection: ✅ Found {len(available_vars['metadata'])} metadata columns")
            print("Metadata columns:", ", ".join(available_vars['metadata'][:5]) + ("..." if len(available_vars['metadata']) > 5 else ""))
        
        # Test 4: Read with different encodings
        df, meta = handler._try_encodings(file_path)
        results["tests"].append({
            "name": "Encoding handling",
            "passed": True,
            "message": f"Successfully read file with shape {df.shape}"
        })
        print(f"Encoding handling: ✅ Successfully read file with shape {df.shape}")
        
        # Test 5: Special codes detection
        special_codes = []
        for var in available_vars['numeric']:
            unique_vals = set(df[var].dropna().unique())
            if any(code in unique_vals for code in [98, 99]):
                special_codes.append(var)
        
        if special_codes:
            results["tests"].append({
                "name": "Special codes detection",
                "passed": True,
                "message": f"Found {len(special_codes)} variables with special codes"
            })
            print(f"Special codes detection: ✅ Found {len(special_codes)} variables with special codes")
            print("Variables with special codes:", ", ".join(special_codes[:5]) + ("..." if len(special_codes) > 5 else ""))
        
        # Test 6: Variable validation
        if len(available_vars['numeric']) >= 3:
            # Test with first 3 numeric variables
            test_vars = available_vars['numeric'][:3]
            is_valid, error_msg = handler.validate_analysis_parameters(
                file_path,
                dependent_vars=[test_vars[0]],
                independent_vars=test_vars[1:3]
            )
            results["tests"].append({
                "name": "Variable validation",
                "passed": is_valid,
                "message": error_msg if not is_valid else "Variables validated successfully"
            })
            print(f"Variable validation: {'✅' if is_valid else '❌'} {error_msg if not is_valid else ''}")
        
    except Exception as e:
        results["errors"].append(str(e))
        print(f"Error during testing: {str(e)}")
    
    return results

def test_all_files():
    """Test all SPSS files in the test directories"""
    test_dirs = ["scenarios", "edge_cases", "error_cases"]
    all_results = []
    
    for dir_name in test_dirs:
        dir_path = os.path.join(project_root, "test_data", dir_name)
        if not os.path.exists(dir_path):
            continue
            
        print(f"\nTesting files in {dir_name}")
        print("=" * 50)
        
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".sav"):
                file_path = os.path.join(dir_path, file_name)
                results = test_file(file_path)
                all_results.append(results)
    
    return all_results

def print_summary(results):
    """Print summary of all test results"""
    print("\nTest Summary")
    print("=" * 50)
    
    total_files = len(results)
    total_tests = sum(len(r["tests"]) for r in results)
    passed_tests = sum(sum(1 for t in r["tests"] if t["passed"]) for r in results)
    failed_tests = total_tests - passed_tests
    files_with_errors = sum(1 for r in results if r["errors"])
    
    print(f"Total files tested: {total_files}")
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {failed_tests}")
    print(f"Files with errors: {files_with_errors}")
    
    if failed_tests > 0 or files_with_errors > 0:
        print("\nFailed Tests:")
        for result in results:
            failed = [t for t in result["tests"] if not t["passed"]]
            if failed or result["errors"]:
                print(f"\nFile: {result['file_name']}")
                for test in failed:
                    print(f"- {test['name']}: {test['message']}")
                for error in result["errors"]:
                    print(f"- Error: {error}")

if __name__ == "__main__":
    results = test_all_files()
    print_summary(results)
