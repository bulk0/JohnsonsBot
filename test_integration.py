"""
Integration tests for Johnson's Bot components.
Tests the interaction between different components and full workflow.
"""

import os
import sys
import unittest
import tempfile
import shutil
from typing import Dict, Any
import pandas as pd
import pyreadstat

from weights_handler import WeightsCalculationHandler
from spss_handlers import read_spss_with_fallbacks, validate_spss_file
from johnson_weights import calculate_johnson_weights
import output_generator

class TestFullWorkflow(unittest.TestCase):
    """Test complete workflow from file reading to output generation"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.handler = WeightsCalculationHandler(base_dir=cls.test_dir)
        
        # Copy test files to temp directory
        cls.test_files = {
            'standard': 'test_data/scenarios/Scen1-B_aggr.sav',
            'cyrillic': 'test_data/error_cases/База Johnson_верхний.sav',
            'complex': 'test_data/edge_cases/Yandex_CSI 2025 Poisk Finsrez_(FINAL_for_client)_v2.sav'
        }
        
        cls.temp_files = {}
        for key, path in cls.test_files.items():
            if os.path.exists(path):
                dest = os.path.join(cls.test_dir, os.path.basename(path))
                shutil.copy2(path, dest)
                cls.temp_files[key] = dest
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    def test_standard_workflow(self):
        """Test standard file processing workflow"""
        file_path = self.temp_files.get('standard')
        if not file_path:
            self.skipTest("Standard test file not available")
        
        # 1. Validate file
        is_valid, message, file_info = validate_spss_file(file_path)
        self.assertTrue(is_valid, f"File validation failed: {message}")
        
        # 2. Read file
        df, meta = read_spss_with_fallbacks(file_path)
        self.assertIsNotNone(df, "Failed to read SPSS file")
        
        # 3. Get variables
        vars_info = self.handler.get_available_variables(file_path)
        self.assertIn('numeric', vars_info, "No numeric variables found")
        self.assertTrue(len(vars_info['numeric']) >= 3, "Insufficient numeric variables")
        
        # 4. Calculate weights
        dependent_vars = vars_info['numeric'][:1]
        independent_vars = vars_info['numeric'][1:4]
        
        result = self.handler.calculate_weights(
            input_file=file_path,
            dependent_vars=dependent_vars,
            independent_vars=independent_vars
        )
        
        self.assertEqual(result['status'], 'success', f"Calculation failed: {result.get('error')}")
        self.assertTrue(os.path.exists(result['results']), "Results file not created")

    def test_cyrillic_workflow(self):
        """Test workflow with Cyrillic file"""
        file_path = self.temp_files.get('cyrillic')
        if not file_path:
            self.skipTest("Cyrillic test file not available")
        
        # 1. Validate file
        is_valid, message, file_info = validate_spss_file(file_path)
        self.assertTrue(is_valid, f"File validation failed: {message}")
        self.assertEqual(file_info.get('type'), 'cyrillic', "File type not detected as cyrillic")
        
        # 2. Read file
        df, meta = read_spss_with_fallbacks(file_path)
        self.assertIsNotNone(df, "Failed to read Cyrillic SPSS file")
        
        # Test rest of workflow...

    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test with invalid file
        with tempfile.NamedTemporaryFile(suffix='.sav') as temp:
            temp.write(b'Invalid content')
            temp.flush()
            
            # Should fail gracefully
            is_valid, message, _ = validate_spss_file(temp.name)
            self.assertFalse(is_valid)
            self.assertIn("Not a valid SPSS file", message)
    
    def test_concurrent_processing(self):
        """Test handling multiple files concurrently"""
        import concurrent.futures
        
        def process_file(file_path: str) -> Dict[str, Any]:
            try:
                df, meta = read_spss_with_fallbacks(file_path)
                return {'status': 'success', 'rows': len(df)}
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        
        # Process multiple files concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(process_file, path): name 
                for name, path in self.temp_files.items()
            }
            
            results = {}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {'status': 'error', 'error': str(e)}
        
        # Check results
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        self.assertGreater(successful, 0, "No files processed successfully")

    def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        for name, path in self.temp_files.items():
            if not os.path.exists(path):
                continue
                
            # Force garbage collection before test
            gc.collect()
            
            try:
                # Measure initial memory
                mem_before = process.memory_info().rss
                
                # Get file size for context
                file_size = os.path.getsize(path)
                
                # Process file
                df, meta = read_spss_with_fallbacks(path)
                
                # Basic file validation
                self.assertIsNotNone(df, "DataFrame should not be None")
                self.assertGreater(len(df), 0, "DataFrame should not be empty")
                
                # Measure peak memory
                mem_peak = process.memory_info().rss
                mem_diff = mem_peak - mem_before
                
                # Clean up
                del df, meta
                gc.collect()
                
                # Memory usage should be reasonable relative to file size
                max_ratio = 5  # Allow up to 5x file size in memory
                memory_ratio = mem_diff / file_size
                
                self.assertLess(
                    memory_ratio,
                    max_ratio,
                    f"High memory usage for {name}: "
                    f"{mem_diff / (1024*1024):.1f}MB "
                    f"(file size: {file_size / (1024*1024):.1f}MB, "
                    f"ratio: {memory_ratio:.1f}x)"
                )
                
            except Exception as e:
                self.fail(f"Failed to process {name}: {str(e)}")

def run_integration_tests():
    """Run integration tests with detailed reporting"""
    import sys
    import time
    
    # Configure test output
    unittest.TestLoader.sortTestMethodsUsing = None
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Time the tests
    start_time = time.time()
    result = runner.run(unittest.makeSuite(TestFullWorkflow))
    duration = time.time() - start_time
    
    # Print summary
    print("\nTest Summary:")
    print(f"Ran {result.testsRun} tests in {duration:.2f} seconds")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    sys.exit(run_integration_tests())
