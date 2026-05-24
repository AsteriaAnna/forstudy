import re
from typing import Tuple, Optional


class IDGenerator:
    """Generator and parser for cutting IDs in PHM2010 dataset."""
    
    VALID_TOOL_IDS = ['c1', 'c4', 'c6']
    
    # Pattern 1: c_1_001.csv -> (c1, 1)
    # Pattern 2: c_c1_1.csv -> (c1, 1)
    PATTERN_1 = re.compile(r'^c_(\d)_\d{3}\.csv$')
    PATTERN_2 = re.compile(r'^c_c(\d)_\d+\.csv$')
    
    @classmethod
    def parse_filename(cls, filename: str) -> Tuple[str, int]:
        """Parse filename to extract tool_id and cut_num.
        
        Args:
            filename: CSV filename like c_1_001.csv or c_c1_1.csv
            
        Returns:
            Tuple of (tool_id, cut_num)
            
        Raises:
            ValueError: If filename doesn't match any known pattern
        """
        # Try Pattern 1: c_{tool}_{cut_num}.csv (tool is 1/4/6, cut_num is 3-digit zero-padded)
        match1 = cls.PATTERN_1.match(filename)
        if match1:
            tool_num = match1.group(1)
            cut_num = match1.group(2)
            return (f'c{tool_num}', int(cut_num))
        
        # Try Pattern 2: c_c{tool}_{cut_num}.csv (tool is 1/4/6, cut_num is not padded)
        match2 = cls.PATTERN_2.match(filename)
        if match2:
            tool_num = match2.group(1)
            cut_num = match2.group(2)
            return (f'c{tool_num}', int(cut_num))
        
        raise ValueError(f"Cannot parse filename: {filename}")
    
    @classmethod
    def generate_cut_unique_id(cls, tool_id: str, cut_num: int) -> str:
        """Generate cut unique ID.
        
        Args:
            tool_id: Tool identifier like 'c1', 'c4', 'c6'
            cut_num: Cut number (not zero-padded)
            
        Returns:
            Generated ID in format: {tool_id}_cut{cut_num}
        """
        return f'{tool_id}_cut{cut_num}'
    
    @classmethod
    def is_valid_tool_id(cls, tool_id: str) -> bool:
        """Check if tool_id is valid.
        
        Args:
            tool_id: Tool identifier to validate
            
        Returns:
            True if tool_id is one of c1/c4/c6, False otherwise
        """
        return tool_id in cls.VALID_TOOL_IDS
