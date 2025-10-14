import os
import sys
import struct
import tempfile
from pathlib import Path

class SPSSBinaryReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.header = None
        self.var_count = 0
        self.case_count = 0
        self.compression = 0
        self.encoding = None
        self.variables = []
        
    def read_header(self):
        """Read SPSS file header"""
        with open(self.file_path, 'rb') as f:
            # Read magic number
            magic = f.read(4)
            if magic != b'$FL2':
                raise ValueError("Not a valid SPSS file")
            
            # Skip product info
            f.seek(60, 1)  # 64 total
            
            # Read basic info
            self.compression = struct.unpack('i', f.read(4))[0]
            self.case_count = struct.unpack('i', f.read(4))[0]
            
            # Try to detect encoding
            f.seek(72)
            enc_info = f.read(8)
            if b'UTF' in enc_info:
                self.encoding = 'utf-8'
            elif any(b in enc_info for b in [b'1251', b'866', b'KOI8']):
                self.encoding = 'windows-1251'
            else:
                self.encoding = 'cp1251'  # Default for Cyrillic
                
            return True
            
    def fix_encoding(self):
        """Create a copy of the file with fixed encoding"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.sav')
        try:
            with open(self.file_path, 'rb') as src, open(temp_file.name, 'wb') as dst:
                # Copy header
                header = src.read(72)
                dst.write(header)
                
                # Write encoding info
                dst.write(self.encoding.encode().ljust(8, b'\x00'))
                
                # Copy rest of file
                src.seek(80)
                while True:
                    chunk = src.read(8192)
                    if not chunk:
                        break
                    dst.write(chunk)
                    
            return temp_file.name
        except:
            try:
                os.unlink(temp_file.name)
            except:
                pass
            return None
            
    def detect_var_count(self):
        """Try to detect number of variables"""
        with open(self.file_path, 'rb') as f:
            f.seek(84)  # Skip header
            # Look for variable record markers
            count = 0
            while True:
                chunk = f.read(4)
                if not chunk or len(chunk) < 4:
                    break
                if chunk in [b'\x02\x00\x00\x00', b'\x03\x00\x00\x00']:
                    count += 1
            return count
            
    def create_fixed_copy(self):
        """Create a fixed copy of the file"""
        if not self.read_header():
            return None
            
        # Detect number of variables
        self.var_count = self.detect_var_count()
        if self.var_count == 0:
            return None
            
        # Create fixed copy
        return self.fix_encoding()

def main():
    if len(sys.argv) != 2:
        print("Usage: python spss_binary_reader.py <spss_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    reader = SPSSBinaryReader(file_path)
    
    try:
        print("Reading file header...")
        reader.read_header()
        print(f"Compression: {reader.compression}")
        print(f"Case count: {reader.case_count}")
        print(f"Detected encoding: {reader.encoding}")
        
        print("\nAnalyzing variable structure...")
        var_count = reader.detect_var_count()
        print(f"Detected variables: {var_count}")
        
        print("\nCreating fixed copy...")
        fixed_file = reader.create_fixed_copy()
        if fixed_file:
            print(f"Fixed file created: {fixed_file}")
            
            # Try to read with pyreadstat
            try:
                import pyreadstat
                df, meta = pyreadstat.read_sav(fixed_file)
                print("\nSuccessfully read fixed file!")
                print(f"Shape: {df.shape}")
                print("\nColumns:")
                for col in df.columns:
                    print(f"- {col}")
            except Exception as e:
                print(f"Error reading fixed file: {str(e)}")
            finally:
                try:
                    os.unlink(fixed_file)
                except:
                    pass
        else:
            print("Failed to create fixed copy")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
