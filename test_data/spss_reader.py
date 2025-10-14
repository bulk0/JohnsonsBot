import os
import sys
import pyreadstat
import pandas as pd
import struct
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def read_spss_header(file_path):
    """Read and parse SPSS file header"""
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(4)
        if header != b'$FL2':
            return None, "Not a valid SPSS file"
            
        # Read version info
        f.seek(64)
        version = f.read(8)
        
        # Read encoding info (if available)
        f.seek(72)
        encoding_info = f.read(8)
        
        return {
            'header': header,
            'version': version,
            'encoding_info': encoding_info
        }, None

def fix_spss_encoding(input_file, output_file=None):
    """Try to fix SPSS file encoding"""
    if output_file is None:
        output_file = input_file + '.fixed'
        
    try:
        # Read the entire file
        with open(input_file, 'rb') as f:
            content = f.read()
            
        # Check if it's a compressed file
        if content[0:4] != b'$FL2':
            return None, "Not a valid SPSS file"
            
        # Try to detect encoding from content
        encodings = [
            ('utf-8', b'UTF-8'),
            ('cp1251', b'CP1251'),
            ('cp866', b'CP866'),
            ('koi8-r', b'KOI8-R'),
            ('iso-8859-5', b'ISO-8859-5'),
            ('windows-1251', b'WINDOWS-1251')
        ]
        
        detected_encoding = None
        for enc, marker in encodings:
            if marker in content:
                detected_encoding = enc
                break
                
        if not detected_encoding:
            # Default to windows-1251 for Cyrillic
            detected_encoding = 'windows-1251'
            
        # Modify encoding information in the file
        # SPSS encoding info is at offset 72
        modified_content = content[:72] + detected_encoding.encode().ljust(8, b'\x00') + content[80:]
        
        # Write modified file
        with open(output_file, 'wb') as f:
            f.write(modified_content)
            
        return output_file, None
        
    except Exception as e:
        return None, str(e)

def read_with_fallback(file_path):
    """Try multiple methods to read SPSS file"""
    # First try direct reading
    try:
        df, meta = pyreadstat.read_sav(file_path)
        return df, meta, None
    except Exception as e:
        print(f"Direct reading failed: {str(e)}")
    
    # Try reading header
    header_info, error = read_spss_header(file_path)
    if error:
        return None, None, error
    print("Header info:", header_info)
    
    # Try fixing encoding
    fixed_file, error = fix_spss_encoding(file_path)
    if error:
        return None, None, error
    
    try:
        df, meta = pyreadstat.read_sav(fixed_file)
        os.unlink(fixed_file)  # Clean up
        return df, meta, None
    except Exception as e:
        print(f"Reading fixed file failed: {str(e)}")
        try:
            os.unlink(fixed_file)  # Clean up
        except:
            pass
    
    # Try pandas as last resort
    try:
        df = pd.read_spss(file_path)
        meta = type('Meta', (), {
            'column_names': df.columns.tolist(),
            'column_labels': dict(zip(df.columns, df.columns)),
            'variable_value_labels': {},
            'variable_measure': dict(zip(df.columns, ['unknown'] * len(df.columns)))
        })
        return df, meta, None
    except Exception as e:
        print(f"Pandas reading failed: {str(e)}")
    
    return None, None, "All reading attempts failed"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python spss_reader.py <spss_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    df, meta, error = read_with_fallback(file_path)
    
    if error:
        print(f"Error: {error}")
        sys.exit(1)
        
    print("\nFile successfully read!")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col} ({df[col].dtype})")
