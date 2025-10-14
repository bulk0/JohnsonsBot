"""
Performance tests for Johnson's Bot.
Tests system performance under various conditions.
"""

import os
import time
import psutil
import logging
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from spss_handlers import read_spss_with_fallbacks
from weights_handler import WeightsCalculationHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Track and report performance metrics"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_time': [],
            'file_sizes': [],
            'concurrent_ops': []
        }
    
    def start_measurement(self):
        """Start performance measurement"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.start_cpu = self.process.cpu_percent()
    
    def end_measurement(self, file_size: int = 0, concurrent: int = 1):
        """End performance measurement and record metrics"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        end_cpu = self.process.cpu_percent()
        
        self.metrics['execution_time'].append(end_time - self.start_time)
        self.metrics['memory_usage'].append(end_memory - self.start_memory)
        self.metrics['cpu_usage'].append(end_cpu - self.start_cpu)
        self.metrics['file_sizes'].append(file_size)
        self.metrics['concurrent_ops'].append(concurrent)
    
    def get_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        return {
            'avg_execution_time': np.mean(self.metrics['execution_time']),
            'max_execution_time': np.max(self.metrics['execution_time']),
            'avg_memory_usage_mb': np.mean(self.metrics['memory_usage']) / (1024 * 1024),
            'max_memory_usage_mb': np.max(self.metrics['memory_usage']) / (1024 * 1024),
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']),
            'max_cpu_usage': np.max(self.metrics['cpu_usage']),
            'total_operations': len(self.metrics['execution_time'])
        }

def test_file_processing_performance(
    test_files: List[str],
    metrics: PerformanceMetrics,
    concurrent: bool = False
) -> Dict[str, Any]:
    """
    Test file processing performance
    
    Args:
        test_files: List of test file paths
        metrics: Performance metrics tracker
        concurrent: Whether to process files concurrently
        
    Returns:
        Dict with test results
    """
    results = {'success': 0, 'failed': 0, 'errors': []}
    
    def process_file(file_path: str) -> Dict[str, Any]:
        try:
            metrics.start_measurement()
            
            # Read file
            df, meta = read_spss_with_fallbacks(file_path)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            metrics.end_measurement(file_size=file_size)
            
            return {
                'status': 'success',
                'file': file_path,
                'rows': len(df),
                'columns': len(df.columns)
            }
        except Exception as e:
            return {
                'status': 'error',
                'file': file_path,
                'error': str(e)
            }
    
    if concurrent:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_file = {
                executor.submit(process_file, f): f 
                for f in test_files
            }
            
            for future in as_completed(future_to_file):
                result = future.result()
                if result['status'] == 'success':
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(result['error'])
    else:
        for file_path in test_files:
            result = process_file(file_path)
            if result['status'] == 'success':
                results['success'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(result['error'])
    
    return results

def test_calculation_performance(
    test_files: List[str],
    metrics: PerformanceMetrics
) -> Dict[str, Any]:
    """Test calculation engine performance"""
    results = {'success': 0, 'failed': 0, 'errors': []}
    handler = WeightsCalculationHandler()
    
    for file_path in test_files:
        try:
            metrics.start_measurement()
            
            # Get variables
            vars_info = handler.get_available_variables(file_path)
            if not vars_info.get('numeric'):
                continue
                
            # Select variables for test
            numeric_vars = vars_info['numeric']
            dependent_vars = numeric_vars[:1]
            independent_vars = numeric_vars[1:4]
            
            # Calculate weights
            result = handler.calculate_weights(
                input_file=file_path,
                dependent_vars=dependent_vars,
                independent_vars=independent_vars
            )
            
            metrics.end_measurement()
            
            if result['status'] == 'success':
                results['success'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(result.get('error'))
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(str(e))
    
    return results

def run_performance_tests():
    """Run all performance tests"""
    # Initialize metrics
    metrics = PerformanceMetrics()
    
    # Get test files
    test_files = []
    test_dirs = ['test_data/scenarios', 'test_data/edge_cases', 'test_data/error_cases']
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.sav'):
                    test_files.append(os.path.join(dir_path, file_name))
    
    if not test_files:
        logger.error("No test files found")
        return
    
    logger.info(f"Found {len(test_files)} test files")
    
    # Run tests
    tests = [
        ("Sequential File Processing", 
         lambda: test_file_processing_performance(test_files, metrics)),
        ("Concurrent File Processing", 
         lambda: test_file_processing_performance(test_files, metrics, concurrent=True)),
        ("Calculation Performance", 
         lambda: test_calculation_performance(test_files, metrics))
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        results[test_name] = test_func()
    
    # Generate report
    performance_report = metrics.get_report()
    
    # Print results
    print("\nPerformance Test Results")
    print("=" * 50)
    
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        print(f"  Successful: {result['success']}")
        print(f"  Failed: {result['failed']}")
        if result['errors']:
            print("  Errors:")
            for error in result['errors'][:3]:
                print(f"    - {error}")
            if len(result['errors']) > 3:
                print(f"    ... and {len(result['errors']) - 3} more")
    
    print("\nPerformance Metrics:")
    print(f"  Average execution time: {performance_report['avg_execution_time']:.2f}s")
    print(f"  Maximum execution time: {performance_report['max_execution_time']:.2f}s")
    print(f"  Average memory usage: {performance_report['avg_memory_usage_mb']:.1f}MB")
    print(f"  Maximum memory usage: {performance_report['max_memory_usage_mb']:.1f}MB")
    print(f"  Average CPU usage: {performance_report['avg_cpu_usage']:.1f}%")
    print(f"  Total operations: {performance_report['total_operations']}")

if __name__ == '__main__':
    run_performance_tests()
