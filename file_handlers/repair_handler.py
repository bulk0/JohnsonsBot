import os
import tempfile
import logging
import pyreadstat
import pandas as pd
from typing import Tuple, List, Dict, Optional
from .base_handler import BaseFileHandler, EncodingError

logger = logging.getLogger(__name__)

class SPSSFileRepairHandler:
    """Handler class for attempting to repair and read problematic SPSS files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
        self.repair_attempts = []
        
    def attempt_repair(self) -> Tuple[Optional[pd.DataFrame], Optional[object], List[Dict]]:
        """
        Attempts various strategies to repair and read the SPSS file
        
        Returns:
            Tuple containing:
            - DataFrame if successful, None if failed
            - Metadata object if successful, None if failed
            - List of repair attempts and their outcomes
        """
        # Strategy 1: Try different encodings
        encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin1', 'koi8-r', 'cp866']
        for encoding in encodings:
            try:
                df, meta = pyreadstat.read_sav(
                    self.file_path,
                    encoding=encoding,
                    apply_value_formats=False
                )
                self._log_attempt("encoding_fix", encoding, True)
                return df, meta, self.repair_attempts
            except Exception as e:
                self._log_attempt("encoding_fix", encoding, False, str(e))

        # Strategy 2: Try fixing file header
        try:
            df, meta = self._try_fix_header()
            if df is not None:
                return df, meta, self.repair_attempts
        except Exception as e:
            self._log_attempt("header_fix", "standard", False, str(e))

        # Strategy 3: Try reading with minimal metadata
        try:
            df = pd.read_spss(self.file_path, convert_categoricals=False)
            meta = self._create_minimal_meta(df)
            self._log_attempt("minimal_metadata", "pandas", True)
            return df, meta, self.repair_attempts
        except Exception as e:
            self._log_attempt("minimal_metadata", "pandas", False, str(e))

        # Strategy 4: Try reading in binary mode and fix encoding markers
        try:
            df, meta = self._try_binary_fix()
            if df is not None:
                return df, meta, self.repair_attempts
        except Exception as e:
            self._log_attempt("binary_fix", "standard", False, str(e))

        return None, None, self.repair_attempts

    def _try_fix_header(self) -> Tuple[Optional[pd.DataFrame], Optional[object]]:
        """Attempts to fix the SPSS file header"""
        with open(self.file_path, 'rb') as f:
            content = f.read()
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as temp:
            # Write fixed header with CP1251 encoding
            temp.write(content[:72])  # Original header
            temp.write(b'CP1251\x00\x00')  # Fixed encoding marker
            temp.write(content[80:])  # Rest of the file
            temp_path = temp.name
            
        try:
            df, meta = pyreadstat.read_sav(temp_path, encoding='cp1251')
            self._log_attempt("header_fix", "cp1251", True)
            return df, meta
        except Exception as e:
            self._log_attempt("header_fix", "cp1251", False, str(e))
            return None, None
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def _try_binary_fix(self) -> Tuple[Optional[pd.DataFrame], Optional[object]]:
        """Attempts to fix the file by modifying binary content"""
        with open(self.file_path, 'rb') as f:
            content = f.read()
            
        # Look for common encoding markers and try to fix them
        markers = [
            (b'UTF-8\x00\x00\x00', b'CP1251\x00\x00'),
            (b'UTF-16\x00\x00', b'CP1251\x00\x00'),
            (b'ASCII\x00\x00\x00', b'CP1251\x00\x00')
        ]
        
        for old_marker, new_marker in markers:
            if old_marker in content:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as temp:
                    temp.write(content.replace(old_marker, new_marker))
                    temp_path = temp.name
                    
                try:
                    df, meta = pyreadstat.read_sav(temp_path, encoding='cp1251')
                    self._log_attempt("binary_fix", f"replace_{old_marker}", True)
                    return df, meta
                except Exception as e:
                    self._log_attempt("binary_fix", f"replace_{old_marker}", False, str(e))
                finally:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
        return None, None

    def _create_minimal_meta(self, df: pd.DataFrame) -> object:
        """Creates minimal metadata object for pandas DataFrame"""
        return type('Meta', (), {
            'column_names': df.columns.tolist(),
            'column_labels': dict(zip(df.columns, df.columns)),
            'variable_value_labels': {},
            'variable_measure': dict(zip(df.columns, ['unknown'] * len(df.columns)))
        })

    def _log_attempt(self, strategy: str, details: str, success: bool, error: str = None):
        """Log repair attempt details"""
        attempt = {
            'strategy': strategy,
            'details': details,
            'success': success
        }
        if error:
            attempt['error'] = error
        self.repair_attempts.append(attempt)
        
        if success:
            self.logger.info(f"Repair successful: {strategy} - {details}")
        else:
            self.logger.debug(f"Repair failed: {strategy} - {details} - {error}")
