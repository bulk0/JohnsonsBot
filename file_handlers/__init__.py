"""
File handlers for different SPSS file formats.
"""

from .base_handler import BaseFileHandler, SPSSReadError, EncodingError, FormatError

def get_handler(file_path: str) -> BaseFileHandler:
    """
    Get appropriate handler for the file
    
    Args:
        file_path (str): Path to SPSS file
        
    Returns:
        BaseFileHandler: Appropriate handler instance
    """
    return BaseFileHandler(file_path)
