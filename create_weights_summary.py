import os
import pandas as pd
import numpy as np
import re

# Get all results files from test_results directory
results_dir = os.path.join(os.path.dirname(__file__), 'test_results')
results = []

# Process each results file
for file in os.listdir(results_dir):
    if file.startswith('johnson_weights_') and file.endswith('.csv'):
        file_path = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Get total row (first row)
                total_row = df.iloc[0]
                
                # Extract weights and R²
                weight_cols = [col for col in df.columns if col.startswith('Weight_')]
                if weight_cols:  # Only process if we have weight columns
                    weights = {col.replace('Weight_', ''): total_row[col] for col in weight_cols}
                    
                    # Get scenario from the first row
                    for col in df.columns:
                        if 'Scen' in str(total_row[col]):
                            scenario_match = re.search(r'Scen(\d+)-([A-Z])', str(total_row[col]))
                            if scenario_match:
                                scenario_num = int(scenario_match.group(1))
                                scenario_letter = scenario_match.group(2)
                                scenario_name = f"{scenario_num}{scenario_letter}"
                                
                                result = {
                                    'Scenario': scenario_name,
                                    'Sample Size': int(total_row['Sample Size']),
                                    'R²': float(total_row['R-squared']),
                                    **weights
                                }
                                
                                results.append(result)
                                break
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

if not results:
    print("No results found to process")
    exit(0)

# Create summary DataFrame
summary_df = pd.DataFrame(results)

# Sort by scenario number
summary_df['Scenario'] = pd.to_numeric(summary_df['Scenario'].str.extract(r'(\d+)')[0])
summary_df = summary_df.sort_values('Scenario')

# Format numeric columns
numeric_cols = ['R²'] + [col for col in summary_df.columns if col not in ['Scenario', 'Sample Size']]
for col in numeric_cols:
    summary_df[col] = summary_df[col].round(3)

# Save to Excel with better formatting
output_file = os.path.join(results_dir, 'weights_summary.xlsx')
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    summary_df.to_excel(writer, index=False, sheet_name='Weights Summary')
    worksheet = writer.sheets['Weights Summary']
    
    # Adjust column widths
    for idx, col in enumerate(summary_df.columns):
        max_length = max(
            summary_df[col].astype(str).apply(len).max(),
            len(str(col))
        )
        worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2

print(f"\nWeights summary saved to: {output_file}")
print("\nWeights Summary:")
print(summary_df.to_string()) 