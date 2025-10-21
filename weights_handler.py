import os
import struct
import tempfile
import time
from typing import List, Dict, Optional, Union, Tuple
from johnson_weights import calculate_johnson_weights
import pandas as pd
import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

class WeightsCalculationHandler:
    """
    Handler class for managing Johnson's weights calculations in the bot context.
    Provides an interface between the Telegram bot and the calculation engine.
    """
    
    def __init__(self, base_dir: str = None, max_age_days: int = 30):
        """
        Initialize the handler with base directory for file operations
        
        Args:
            base_dir (str): Base directory for file operations
            max_age_days (int): Maximum age in days for analysis folders before cleanup
        """
        self.base_dir = base_dir or os.getcwd()
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.max_age_days = max_age_days
        
        # Create necessary directories
        for directory in [self.temp_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Clean up old analysis folders
        self.cleanup_old_analyses()
    
    def validate_input_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate the input SPSS file
        
        Args:
            file_path (str): Path to the input file
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        if not file_path.lower().endswith('.sav'):
            return False, "File must be an SPSS .sav file"
            
        return True, ""
    
    def calculate_weights(
        self,
        input_file: str,
        dependent_vars: List[str],
        independent_vars: List[str],
        analysis_type: str = "total",
        subgroups: List[str] = None,
        min_sample_size: int = 100
    ) -> Dict[str, Union[str, Dict]]:
        """
        Calculate Johnson's weights with specified parameters
        
        Args:
            input_file (str): Path to SPSS file
            dependent_vars (List[str]): List of dependent variables
            independent_vars (List[str]): List of independent variables
            analysis_type (str): Type of analysis ('total' or 'group')
            subgroups (List[str], optional): List of subgroup variables (including brand variables if needed)
            min_sample_size (int): Minimum required sample size
            
        Returns:
            Dict containing:
                - status: 'success' or 'error'
                - message: Status message
                - results: Path to results file if successful
                - error: Error details if failed
        """
        try:
            # Validate input file
            is_valid, error_msg = self.validate_input_file(input_file)
            if not is_valid:
                return {
                    'status': 'error',
                    'message': f'Invalid input file: {error_msg}',
                    'error': error_msg
                }
            
            # Prepare analysis parameters based on type
            if analysis_type == "total":
                subgroups = None
            elif analysis_type == "group" and not subgroups:
                return {
                    'status': 'error',
                    'message': 'Group analysis requires subgroup variables',
                    'error': 'Missing subgroups'
                }
            
            # Calculate weights using both imputation methods
            results_file = calculate_johnson_weights(
                input_file=input_file,
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                subgroups=subgroups,
                min_sample_size=min_sample_size,
                output_dir=self.results_dir
            )
            
            if results_file:
                return {
                    'status': 'success',
                    'message': 'Calculation completed successfully using both MICE and Hybrid imputation methods',
                    'results': results_file,
                    'documentation': 'Multiple Imputations Readme.txt'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Calculation failed',
                    'error': 'No results generated'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during calculation: {str(e)}',
                'error': str(e)
            }
    
    def _detect_file_info(self, input_file: str) -> dict:
        """
        Detect SPSS file information from binary structure
        
        Args:
            input_file (str): Path to SPSS file
            
        Returns:
            dict: File information
        """
        with open(input_file, 'rb') as f:
            # Check magic number
            if f.read(4) != b'$FL2':
                return None
                
            # Skip product info
            f.seek(60, 1)  # 64 total
            
            # Read basic info
            compression = struct.unpack('i', f.read(4))[0]
            case_count = struct.unpack('i', f.read(4))[0]
            
            # Try to detect encoding
            f.seek(72)
            enc_info = f.read(8)
            
            # Detect encoding
            if b'UTF' in enc_info:
                encoding = 'utf-8'
            elif any(b in enc_info for b in [b'1251', b'866', b'KOI8']):
                encoding = 'windows-1251'
            else:
                encoding = 'cp1251'  # Default for Cyrillic
            
            # Try to detect from structure
            f.seek(84)  # Start of variable record
            count = 0
            while True:
                chunk = f.read(4)
                if not chunk or len(chunk) < 4:
                    break
                if chunk in [b'\x02\x00\x00\x00', b'\x03\x00\x00\x00']:
                    count += 1
            var_count = count if count > 0 else None
                
            return {
                'compression': compression,
                'case_count': case_count,
                'encoding': encoding,
                'var_count': var_count
            }
    
    def _try_encodings(self, input_file: str) -> Tuple[pd.DataFrame, object]:
        """
        Try different encodings to read SPSS file
        
        Args:
            input_file (str): Path to SPSS file
            
        Returns:
            Tuple[pd.DataFrame, object]: DataFrame and metadata
        """
        logger = logging.getLogger(__name__)
        
        # Defer heavy import to reduce cold start time
        import pyreadstat

        # First try to detect file info
        file_info = self._detect_file_info(input_file)
        if file_info:
            logger.info(f"Detected file info: {file_info}")
            
            if file_info['encoding'] in ['windows-1251', 'cp1251']:
                try:
                    with open(input_file, 'rb') as src:
                        content = src.read()
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as temp:
                        temp.write(content[:72])
                        temp.write(b'CP1251\x00\x00')
                        temp.write(content[80:])
                        temp_path = temp.name
                    try:
                        encodings_to_try = ['cp1251', 'windows-1251']
                        apply_formats_options = [True, False]
                        for enc in encodings_to_try:
                            for apply_formats in apply_formats_options:
                                try:
                                    read_args = {
                                        'encoding': enc,
                                        'apply_value_formats': apply_formats,
                                    }
                                    df, meta = pyreadstat.read_sav(temp_path, **read_args)
                                    logger.info(f"Successfully read file with encoding {enc}, apply_value_formats={apply_formats}")
                                    return df, meta
                                except Exception as e:
                                    last_temp_error = e
                                    continue
                        logger.debug(f"CP1251 header fix attempts failed: {str(last_temp_error)}")
                    finally:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                except Exception as e:
                    logger.debug(f"Failed to prepare CP1251 header fix: {str(e)}")
        
        # If Cyrillic approach fails or for other files, try standard approaches
        try:
            df, meta = pyreadstat.read_sav(input_file)
            logger.info("Successfully read file without encoding specification")
            return df, meta
        except Exception as first_error:
            logger.debug(f"Failed to read without encoding: {str(first_error)}")
            
            encodings = [
                'utf-8', 'cp1251', 'windows-1251', 'latin1', 'iso-8859-1',
                'cp866', 'koi8-r', 'mac-cyrillic', 'iso-8859-5'
            ]
            last_error = first_error
            for encoding in encodings:
                for apply_formats in [True, False]:
                    try:
                        logger.debug(f"Trying encoding={encoding}, apply_value_formats={apply_formats}")
                        df, meta = pyreadstat.read_sav(
                            input_file,
                            encoding=encoding,
                            apply_value_formats=apply_formats,
                        )
                        logger.info(f"Successfully read with encoding={encoding}, apply_value_formats={apply_formats}")
                        return df, meta
                    except Exception as e:
                        last_error = e
                        continue
            
            try:
                df = pd.read_spss(input_file, convert_categoricals=False)
                meta = type('Meta', (), {
                    'column_names': df.columns.tolist(),
                    'column_labels': dict(zip(df.columns, df.columns)),
                    'variable_value_labels': {},
                    'variable_measure': dict(zip(df.columns, ['unknown'] * len(df.columns)))
                })
                logger.info("Successfully read file with pandas read_spss")
                return df, meta
            except Exception as e:
                last_error = e
            
            # PSPP CLI fallback if available
            try:
                pspp_convert = shutil.which('pspp-convert')
                if pspp_convert:
                    logger.info("Attempting PSPP pspp-convert fallback to CSV")
                    with tempfile.TemporaryDirectory() as tmpdir:
                        csv_path = os.path.join(tmpdir, 'converted.csv')
                        cmd = [pspp_convert, '-O', 'csv', input_file, csv_path]
                        try:
                            subprocess.run(cmd, check=True, capture_output=True)
                            # Try reading CSV with common Cyrillic encodings
                            for enc in ['cp1251', 'windows-1251', 'utf-8', 'latin1']:
                                try:
                                    df = pd.read_csv(csv_path, encoding=enc)
                                    meta = type('Meta', (), {
                                        'column_names': df.columns.tolist(),
                                        'column_labels': dict(zip(df.columns, df.columns)),
                                        'variable_value_labels': {},
                                        'variable_measure': dict(zip(df.columns, ['unknown'] * len(df.columns)))
                                    })
                                    logger.info(f"Successfully read via PSPP with encoding {enc}")
                                    return df, meta
                                except Exception:
                                    continue
                        except subprocess.CalledProcessError as e2:
                            logger.debug(f"pspp-convert failed: {e2.stderr}")
            except Exception as e3:
                logger.debug(f"PSPP fallback attempt raised: {str(e3)}")
            
            error_msg = f"Failed to read file with any method. Last error: {str(last_error)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _filter_metadata_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Filter out common metadata columns
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            List[str]: List of columns to exclude
        """
        # Removed 'os' from metadata patterns since it might be a valid variable
        metadata_patterns = [
            'timestamp', 'created_at', 'updated_at', 'completed_at',
            'response_time', 'session_id', 'user_agent', 'ip_address',
            'referrer', 'source'
        ]
        
        # Only match exact words or words at the start/end of column names
        return [col for col in df.columns 
                if any(col.lower() == pattern or 
                      col.lower().startswith(pattern + '_') or 
                      col.lower().endswith('_' + pattern)
                      for pattern in metadata_patterns)]

    def get_available_variables(self, input_file: str) -> Dict[str, List[str]]:
        """
        Get list of available variables from the input file
        
        Args:
            input_file (str): Path to SPSS file
            
        Returns:
            Dict containing lists of numeric and categorical variables
        """
        try:
            # Try reading with different encodings
            df, meta = self._try_encodings(input_file)
            
            # Filter out metadata columns
            metadata_cols = self._filter_metadata_columns(df)
            
            # Get all variables except metadata
            all_vars = [col for col in df.columns if col not in metadata_cols]
            
            # Identify numeric variables
            numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_vars = [var for var in numeric_vars if var not in metadata_cols]
            
            # Filter out variables with special codes only
            for var in numeric_vars[:]:
                unique_vals = set(df[var].dropna().unique())
                if unique_vals.issubset({98, 99}):
                    numeric_vars.remove(var)
            
            # All non-numeric variables are considered categorical
            categorical_vars = [col for col in all_vars if col not in numeric_vars]
            
            return {
                'numeric': numeric_vars,
                'categorical': categorical_vars,
                'metadata': metadata_cols
            }
        except Exception as e:
            return {
                'numeric': [],
                'categorical': [],
                'metadata': [],
                'error': str(e)
            }
    
    def cleanup_old_analyses(self) -> None:
        """
        Clean up analysis folders older than max_age_days
        """
        try:
            current_time = time.time()
            for item in os.listdir(self.results_dir):
                item_path = os.path.join(self.results_dir, item)
                
                # Only process analysis folders
                if not os.path.isdir(item_path) or not item.startswith('analysis_'):
                    continue
                
                # Check folder age
                folder_time = os.path.getctime(item_path)
                age_in_days = (current_time - folder_time) / (24 * 3600)
                
                if age_in_days > self.max_age_days:
                    try:
                        shutil.rmtree(item_path)
                        logger.info(f"Cleaned up old analysis folder: {item}")
                    except Exception as e:
                        logger.error(f"Error cleaning up folder {item}: {str(e)}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def validate_analysis_parameters(
        self,
        input_file: str,
        dependent_vars: List[str],
        independent_vars: List[str],
        subgroups: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate analysis parameters against the input file
        
        Args:
            input_file (str): Path to SPSS file
            dependent_vars (List[str]): List of dependent variables
            independent_vars (List[str]): List of independent variables
            subgroups (List[str], optional): List of subgroup variables
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Get available variables
            available_vars = self.get_available_variables(input_file)
            
            # Get all available variables
            all_available_vars = available_vars['numeric'] + available_vars['categorical']
            
            # Create case-insensitive mapping for variable names
            vars_lower_map = {var.lower(): var for var in all_available_vars}
            
            # First check if all variables exist in the dataset (case-insensitive)
            all_requested_vars = dependent_vars + independent_vars + (subgroups or [])
            missing_vars = []
            for var in all_requested_vars:
                if var.lower() not in vars_lower_map:
                    missing_vars.append(var)
            if missing_vars:
                # Check if any of the missing variables were filtered as metadata by reading the raw file
                try:
                    df_check, _ = self._try_encodings(input_file)
                    metadata_cols = self._filter_metadata_columns(df_check)
                    metadata_vars = [var for var in missing_vars if var in metadata_cols]
                    if metadata_vars:
                        return False, (
                            f"Variables {', '.join(metadata_vars)} were filtered out as metadata variables.\n"
                            "If these are actually survey variables that you want to use, please rename them in your SPSS file. "
                            "For example, you could rename:\n" +
                            "\n".join(f"- {var} → survey_{var} or q_{var}" for var in metadata_vars)
                        )
                except:
                    # If we can't read the file, just report missing variables
                    pass
                return False, f"Missing variables in dataset: {', '.join(missing_vars)}"
            
            # For dependent and independent variables, ensure they can be converted to numeric
            try:
                df, _ = self._try_encodings(input_file)
                
                # Map requested variable names to correct case from dataframe
                df_columns_lower = {col.lower(): col for col in df.columns}
                analysis_vars = dependent_vars + independent_vars
                analysis_vars_mapped = []
                
                for var in analysis_vars:
                    var_lower = var.lower()
                    if var_lower in df_columns_lower:
                        analysis_vars_mapped.append(df_columns_lower[var_lower])
                    else:
                        # Variable not found even with case-insensitive matching
                        return False, f"Variable '{var}' not found in dataset"
                
                # Try converting to numeric, excluding special codes
                for var in analysis_vars_mapped:
                    # Replace special codes with NaN
                    values = df[var].replace([98, 99], pd.NA)
                    # Try converting to numeric
                    pd.to_numeric(values, errors='raise')
                
                return True, ""
                
            except ValueError as e:
                # If conversion fails, identify which variables failed
                non_numeric_vars = []
                for var in analysis_vars_mapped:
                    try:
                        values = df[var].replace([98, 99], pd.NA)
                        pd.to_numeric(values, errors='raise')
                    except:
                        non_numeric_vars.append(var)
                
                if non_numeric_vars:
                    return False, f"Variables must be numeric: {', '.join(non_numeric_vars)}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
