"""
SPSS file handling with robust error handling and user feedback.
"""

import os
import struct
import tempfile
from typing import Dict, Any, Tuple, List, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class SPSSReadError(Exception):
    """Custom exception for SPSS reading errors with detailed information"""
    def __init__(self, message: str, attempts: List[Dict[str, str]], file_info: Dict[str, Any] = None):
        self.message = message
        self.attempts = attempts
        self.file_info = file_info or {}
        super().__init__(self.message)

    def get_user_message(self) -> str:
        """Get user-friendly error message with suggestions"""
        base_msg = "Unable to read your SPSS file."
        
        # Add file information
        if self.file_info:
            if "file_size" in self.file_info:
                size_mb = self.file_info["file_size"] / (1024 * 1024)
                base_msg += f"\nFile size: {size_mb:.2f} MB"
            
            if "header_valid" in self.file_info:
                if not self.file_info["header_valid"]:
                    base_msg += "\nThe file header is not valid for an SPSS file."
                    if "magic_bytes" in self.file_info:
                        base_msg += f"\nFound header: {self.file_info['magic_bytes']} (expected: 24464C32)"
            
            if "version_info" in self.file_info and self.file_info["version_info"]:
                base_msg += f"\nDetected SPSS version: {self.file_info['version_info']}"
            
            if "encoding" in self.file_info:
                base_msg += f"\nDetected encoding: {self.file_info['encoding']}"
                if "encoding_confidence" in self.file_info:
                    base_msg += f" (confidence: {self.file_info['encoding_confidence']})"
            
            if "type" in self.file_info and self.file_info["type"] == "cyrillic":
                base_msg += "\nThis file contains Cyrillic text."
        
        # Add specific error details
        if self.attempts:
            base_msg += "\n\nAttempted methods:"
            for attempt in self.attempts[:3]:  # Show only first 3 attempts
                base_msg += f"\n- {attempt['method']}: {attempt['error']}"
            if len(self.attempts) > 3:
                base_msg += f"\n(and {len(self.attempts) - 3} more attempts)"
        
        # Add suggestions
        suggestions = [
            "\nTroubleshooting steps:",
            "1. Verify file format:",
            "   - Ensure the file is a genuine SPSS (.sav) file",
            "   - Check if the file was exported correctly from SPSS",
            "   - Try re-saving the file in SPSS if possible",
            "",
            "2. Check file integrity:",
            "   - Verify the file isn't corrupted during transfer",
            "   - Try downloading/copying the file again",
            "",
            "3. Check compatibility:",
            "   - Verify SPSS version compatibility",
            "   - Try saving the file in an older SPSS version"
        ]
        
        if "cyrillic" in str(self.message).lower() or "utf-8" in str(self.message).lower():
            suggestions.extend([
                "",
                "4. For files with Cyrillic text:",
                "   - Save the file with CP1251 (Windows-1251) encoding",
                "   - Ensure variable names use Latin characters",
                "   - Remove special characters from variable labels",
                "   - Try using English locale when saving"
            ])
        
        return f"{base_msg}\n\n{chr(10).join(suggestions)}"

def detect_file_format(file_path: str) -> Dict[str, Any]:
    """
    Detect SPSS file format and characteristics. Performs detailed analysis of file structure
    to identify potential issues and provide specific feedback.
    
    Args:
        file_path: Path to SPSS file
        
    Returns:
        Dict with file format information including:
        - type: File type ('standard', 'cyrillic', 'unknown')
        - encoding: Detected encoding
        - error: Detailed error message if file is invalid
        - compression: Compression type
        - case_count: Number of cases
        - file_size: Size in bytes
        - header_valid: Whether header is valid
        - version_info: SPSS version info if available
    """
    try:
        with open(file_path, 'rb') as f:
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Check magic number
            magic = f.read(4)
            if magic != b'$FL2':
                error_msg = (
                    "Not a valid SPSS file. Common reasons:\n"
                    "1. File might be in a different format (Excel, CSV, etc.)\n"
                    "2. File might be corrupted during transfer\n"
                    "3. File might be from an incompatible SPSS version\n"
                    "4. File might be empty or truncated"
                )
                return {
                    'type': 'unknown',
                    'error': error_msg,
                    'file_size': file_size,
                    'header_valid': False,
                    'magic_bytes': magic.hex()
                }
            
            # Read header info
            f.seek(0)
            header = f.read(80)
            
            # Skip to compression info
            f.seek(64)
            compression = struct.unpack('i', f.read(4))[0]
            case_count = struct.unpack('i', f.read(4))[0]
            
            # Try to detect Cyrillic content
            def has_cyrillic(data: bytes) -> bool:
                return any(0x0400 <= byte <= 0x04FF for byte in data)
            
            # Read more content for analysis
            f.seek(80)
            content_sample = f.read(1000)
            
            # Try to detect SPSS version info
            version_info = None
            if len(header) >= 80:
                version_bytes = header[68:72]
                try:
                    version_num = struct.unpack('i', version_bytes)[0]
                    version_info = f"{version_num // 100}.{version_num % 100}"
                except:
                    pass

            # Detect format type and encoding
            if (any(b in header for b in [b'1251', b'866', b'KOI8']) or
                has_cyrillic(content_sample) or
                'база' in file_path.lower()):
                return {
                    'type': 'cyrillic',
                    'encoding': 'cp1251',
                    'compression': compression,
                    'case_count': case_count,
                    'file_size': file_size,
                    'header_valid': True,
                    'version_info': version_info,
                    'encoding_confidence': 'high' if any(b in header for b in [b'1251', b'866', b'KOI8']) else 'medium'
                }
            else:
                return {
                    'type': 'standard',
                    'encoding': 'utf-8',
                    'compression': compression,
                    'case_count': case_count,
                    'file_size': file_size,
                    'header_valid': True,
                    'version_info': version_info,
                    'encoding_confidence': 'high'
                }
    except Exception as e:
        return {'type': 'unknown', 'error': str(e)}

def create_minimal_meta(df: pd.DataFrame) -> object:
    """Create minimal metadata object for DataFrame"""
    return type('Meta', (), {
        'column_names': df.columns.tolist(),
        'column_labels': dict(zip(df.columns, df.columns)),
        'variable_value_labels': {},
        'variable_measure': dict(zip(df.columns, ['unknown'] * len(df.columns)))
    })

def read_spss_with_fallbacks(file_path: str) -> Tuple[pd.DataFrame, object]:
    """
    Read SPSS file with multiple fallback methods and detailed error tracking
    
    Args:
        file_path: Path to SPSS file
        
    Returns:
        Tuple[DataFrame, metadata]
        
    Raises:
        SPSSReadError: If file cannot be read with any method
    """
    # Defer heavy import to reduce cold start time
    import pyreadstat
    attempts = []
    file_info = detect_file_format(file_path)
    
    # Prepare temp file if needed
    temp_path = None
    if file_info['type'] == 'cyrillic':
        try:
            with open(file_path, 'rb') as src:
                content = src.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as temp:
                # Write header with fixed encoding
                temp.write(content[:72])
                temp.write(b'CP1251\x00\x00')
                temp.write(content[80:])
                temp_path = temp.name
        except Exception as e:
            attempts.append({
                'method': 'temp_file_creation',
                'error': str(e)
            })
    
    try:
        # Try different encodings with pyreadstat
        encodings = ['cp1251', 'windows-1251', 'utf-8', 'latin1', 'koi8-r']
        if file_info.get('encoding'):
            encodings.insert(0, file_info['encoding'])
        
        for encoding in encodings:
            try:
                kwargs = {'encoding': encoding}
                
                # No special file type handling needed
                
                # Try reading original and temp file
                for path in [file_path, temp_path] if temp_path else [file_path]:
                    try:
                        df, meta = pyreadstat.read_sav(path, **kwargs)
                        logger.info(f"Successfully read with encoding {encoding}")
                        return df, meta
                    except Exception as e:
                        attempts.append({
                            'method': f'pyreadstat_{encoding}{"_temp" if path == temp_path else ""}',
                            'error': str(e)
                        })
                        continue
                        
            except Exception as e:
                attempts.append({
                    'method': f'pyreadstat_{encoding}',
                    'error': str(e)
                })
                continue
        
        # Try pandas as fallback
        try:
            df = pd.read_spss(file_path, convert_categoricals=False)
            meta = create_minimal_meta(df)
            logger.info("Successfully read with pandas")
            return df, meta
        except Exception as e:
            attempts.append({
                'method': 'pandas',
                'error': str(e)
            })
        
        # If all attempts fail, raise error with details
        raise SPSSReadError(
            "Failed to read SPSS file with any method",
            attempts=attempts,
            file_info=file_info
        )
        
    finally:
        # Clean up temp file
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass

def validate_spss_file(file_path: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate SPSS file and provide user feedback
    
    Args:
        file_path: Path to SPSS file
        
    Returns:
        Tuple[is_valid, message, file_info]
    """
    try:
        # Check basic file properties
        if not os.path.exists(file_path):
            return False, "File does not exist", None
            
        if not file_path.lower().endswith('.sav'):
            return False, "File must be an SPSS (.sav) file", None
            
        # Detect format
        file_info = detect_file_format(file_path)
        if file_info['type'] == 'unknown':
            # Don't fail immediately, let the repair handler try to fix it
            file_info['needs_repair'] = True
            return True, "File needs repair", file_info
            
        # Try reading the file
        df, meta = read_spss_with_fallbacks(file_path)
        
        # Basic validation
        if df.shape[0] < 100:  # Minimum required sample size
            return False, f"Sample size too small: {df.shape[0]} rows (minimum required: 100)", file_info
            
        if df.shape[1] < 3:  # Minimum required variables
            return False, "Insufficient variables: need at least 3 variables", file_info
            
        # Success
        return True, "File is valid", file_info
        
    except SPSSReadError as e:
        return False, e.get_user_message(), e.file_info
    except Exception as e:
        return False, f"Validation error: {str(e)}", None


def validate_spss_file_fast(file_path: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Fast validation without loading full DataFrame. Checks existence, extension,
    and parses header/metadata to detect format. Intended to keep UI responsive
    and defer heavy reading to later steps.

    Returns:
        Tuple[is_valid, message, file_info]
        - If header is unknown, marks needs_repair but allows flow to attempt repair later
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist", None

        if not file_path.lower().endswith('.sav'):
            return False, "File must be an SPSS (.sav) file", None

        file_info = detect_file_format(file_path)
        if file_info.get('header_valid') is False:
            file_info['needs_repair'] = True
            return True, "File needs repair", file_info

        # Header looks fine; consider file valid for proceeding to lightweight read
        return True, "Header validated", file_info

    except Exception as e:
        return False, f"Fast validation error: {str(e)}", None


def read_spss_sample(file_path: str, row_limit: int = 500) -> Tuple[pd.DataFrame, object]:
    """
    Read a limited number of rows from SPSS to infer schema quickly.

    Uses pyreadstat.read_sav with row_limit and without applying value formats.
    Falls back through a limited set of encodings based on detect_file_format.
    """
    import pyreadstat

    file_info = detect_file_format(file_path)
    encodings = []
    if file_info.get('encoding'):
        encodings.append(file_info['encoding'])
    encodings.extend([e for e in ['cp1251', 'windows-1251', 'utf-8', 'latin1'] if e not in encodings])

    attempts: List[Dict[str, str]] = []
    for encoding in encodings:
        try:
            df, meta = pyreadstat.read_sav(
                file_path,
                encoding=encoding,
                apply_value_formats=False,
                row_limit=row_limit
            )
            logger.info(f"read_spss_sample: success with encoding {encoding}, rows={len(df)}")
            return df, meta
        except Exception as e:
            attempts.append({'method': f'sample_{encoding}', 'error': str(e)})
            continue

    # If all failed, raise a consistent error
    raise SPSSReadError(
        "Failed to read SPSS sample with any encoding",
        attempts=attempts,
        file_info=file_info
    )
