"""
Base handler for SPSS file processing with support for multiple reading methods
and fallback strategies.
"""

import os
import logging
import tempfile
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import pyreadstat

logger = logging.getLogger(__name__)

class SPSSReadError(Exception):
    """Base exception for SPSS reading errors"""
    pass

class EncodingError(SPSSReadError):
    """Raised when file encoding cannot be determined or used"""
    pass

class FormatError(SPSSReadError):
    """Raised when file format is invalid or unsupported"""
    pass

class BaseFileHandler(ABC):
    """Base class for SPSS file handlers"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the handler"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def detect_format(self) -> Dict[str, Any]:
        """
        Detect file format and metadata
        
        Returns:
            Dict containing format information
        """
        pass
    
    def read_file(self) -> Tuple[pd.DataFrame, object]:
        """
        Read SPSS file using available methods
        
        Returns:
            Tuple[pd.DataFrame, object]: DataFrame and metadata
        
        Raises:
            SPSSReadError: If file cannot be read
        """
        errors = []
        
        # Try native pyreadstat first
        try:
            return self._read_with_pyreadstat()
        except Exception as e:
            self.logger.debug(f"pyreadstat read failed: {str(e)}")
            errors.append(("pyreadstat", str(e)))
        
        # Try pandas as fallback
        try:
            return self._read_with_pandas()
        except Exception as e:
            self.logger.debug(f"pandas read failed: {str(e)}")
            errors.append(("pandas", str(e)))
        
        # Try PSPP if available
        try:
            result = self._read_with_pspp()
            if result:
                return result
        except Exception as e:
            self.logger.debug(f"PSPP read failed: {str(e)}")
            errors.append(("pspp", str(e)))
        
        # If all methods fail, raise error with details
        raise SPSSReadError(
            f"Failed to read file with any method. Errors: {errors}"
        )
    
    def _read_with_pyreadstat(self) -> Tuple[pd.DataFrame, object]:
        """Read file using pyreadstat with encoding detection"""
        format_info = self.detect_format()
        
        # Try different encodings
        encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin1']
        if format_info.get('encoding'):
            encodings.insert(0, format_info['encoding'])
        
        for encoding in encodings:
            try:
                df, meta = pyreadstat.read_sav(
                    self.file_path,
                    encoding=encoding,
                    apply_value_formats=False
                )
                self.logger.info(f"Successfully read with pyreadstat ({encoding})")
                return df, meta
            except Exception as e:
                self.logger.debug(f"pyreadstat failed with {encoding}: {str(e)}")
                continue
        
        raise EncodingError("Failed to read with any encoding")
    
    def _read_with_pandas(self) -> Tuple[pd.DataFrame, object]:
        """Read file using pandas"""
        df = pd.read_spss(self.file_path, convert_categoricals=False)
        
        # Create minimal metadata
        meta = type('Meta', (), {
            'column_names': df.columns.tolist(),
            'column_labels': dict(zip(df.columns, df.columns)),
            'variable_value_labels': {},
            'variable_measure': dict(zip(df.columns, ['unknown'] * len(df.columns)))
        })
        
        self.logger.info("Successfully read with pandas")
        return df, meta
    
    def _read_with_pspp(self) -> Optional[Tuple[pd.DataFrame, object]]:
        """Read file using PSPP conversion"""
        pspp_convert = subprocess.which('pspp-convert')
        if not pspp_convert:
            self.logger.debug("pspp-convert not found")
            return None
            
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'converted.csv')
            cmd = [pspp_convert, '-O', 'csv', self.file_path, csv_path]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Try reading CSV with different encodings
                encodings = ['cp1251', 'windows-1251', 'utf-8', 'latin1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(csv_path, encoding=encoding)
                        
                        # Create minimal metadata
                        meta = type('Meta', (), {
                            'column_names': df.columns.tolist(),
                            'column_labels': dict(zip(df.columns, df.columns)),
                            'variable_value_labels': {},
                            'variable_measure': dict(zip(df.columns, ['unknown'] * len(df.columns)))
                        })
                        
                        self.logger.info(f"Successfully read with PSPP ({encoding})")
                        return df, meta
                    except Exception:
                        continue
                        
            except subprocess.CalledProcessError as e:
                self.logger.debug(f"PSPP conversion failed: {e.stderr}")
            
        return None
