import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import re

def get_latest_results(results_dir):
    """Get the latest results files for each timestamp"""
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not csv_files:
        return []
        
    # Get the latest timestamp
    latest_timestamp = max(f.split('_')[1] + '_' + f.split('_')[2] for f in csv_files)
    
    # Get all files with this timestamp
    latest_files = [f for f in csv_files if latest_timestamp in f]
    return latest_files

def get_sample_sizes():
    """Get mapping of sample sizes to scenarios"""
    return {
        2715: 'Scen6-G',   # Smallest sample
        6501: 'Scen8-I',   # Largest sample
        3783: 'Scen4-E',
        4320: 'Scen3-D',
        4963: 'Scen10-K',
        6373: 'Scen7-H',
        3855: 'Scen11-L',
        3870: 'Scen2-C',
        4139: 'Scen9-J',
        4966: 'Scen1-B',
        5310: 'Scen12-M',
        5094: 'Scen5-F'
    }

def create_summary_table(results_dir):
    results = []
    
    # Get latest results files
    latest_files = get_latest_results(results_dir)
    if not latest_files:
        print("No CSV files found in results directory")
        return None
    
    # Get sample size mapping
    sample_size_map = get_sample_sizes()
    
    # Process each file
    seen_sample_sizes = {}
    for file in latest_files:
        file_path = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(file_path)
            
            # Get only the total sample rows
            total_rows = df[df['Group Type'] == 'Total']
            
            for _, total_row in total_rows.iterrows():
                sample_size = total_row['Sample Size']
                
                # Skip if we've seen this sample size before
                if sample_size in seen_sample_sizes:
                    continue
                seen_sample_sizes[sample_size] = True
                
                # Get scenario name from sample size
                scenario = sample_size_map.get(sample_size, 'Unknown')
                
                # Create summary row
                summary = {
                    'Database': scenario,
                    'Sample Size': sample_size,
                    'R²': total_row['R-squared']
                }
                
                # Add weights for each variable
                weight_cols = [col for col in df.columns if col.startswith('Weight_')]
                for col in weight_cols:
                    var_name = col.replace('Weight_', '')
                    summary[f'Weight_{var_name}'] = total_row[col]
                
                results.append(summary)
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    if not results:
        print("No results found to summarize")
        return None
        
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Sort by R² descending
    summary_df = summary_df.sort_values('R²', ascending=False)
    
    # Format the output
    for col in summary_df.columns:
        if col.startswith('Weight_') or col == 'R²':
            summary_df[col] = summary_df[col].map('{:.4f}'.format)
    
    # Save summary
    output_file = os.path.join(results_dir, 'weights_summary.xlsx')
    summary_df.to_excel(output_file, index=False)
    print(f"\nSummary saved to: {output_file}")
    
    # Format column names for display
    display_df = summary_df.copy()
    display_df.columns = [col.replace('Weight_', '').replace('_', ' ').title() for col in display_df.columns]
    
    return display_df

if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
    else:
        summary_df = create_summary_table(results_dir)
        if summary_df is not None:
            print("\nSummary of Results:")
            print(tabulate(summary_df, headers='keys', tablefmt='pipe', showindex=False)) 