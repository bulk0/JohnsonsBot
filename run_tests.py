import os
import pandas as pd
import numpy as np
from johnson_weights import calculate_johnson_weights

# Standard test configuration
TEST_CONFIG = {
    'dependent_vars': ['satisfaction'],
    'independent_vars': [
        'product_attr_1',
        'product_attr_2',
        'product_attr_3',
        'product_attr_4',
        'brand_attr_1'
    ],
    'subgroups': ['brand_id'],
    'layer_var': None,
    'min_sample_size': 100
}

# Create output directory for test results
output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all .sav files from test_data directory
test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
sav_files = [f for f in os.listdir(test_data_dir) if f.endswith('.sav')]

# Sort files by scenario number
sav_files.sort(key=lambda x: int(x.split('-')[0].replace('Scen', '')))

# Process each file
for sav_file in sav_files:
    input_file = os.path.join(test_data_dir, sav_file)
    print(f"\nProcessing {sav_file}...")
    
    try:
        # Run analysis
        output_file = calculate_johnson_weights(
            input_file=input_file,
            dependent_vars=TEST_CONFIG['dependent_vars'],
            independent_vars=TEST_CONFIG['independent_vars'],
            subgroups=TEST_CONFIG['subgroups'],
            layer_var=TEST_CONFIG['layer_var'],
            min_sample_size=TEST_CONFIG['min_sample_size'],
            output_dir=output_dir
        )
        
        if output_file:
            print(f"Analysis completed successfully for {sav_file}")
        else:
            print(f"Analysis failed for {sav_file}")
            
    except Exception as e:
        print(f"Error processing {sav_file}: {str(e)}")
        continue 