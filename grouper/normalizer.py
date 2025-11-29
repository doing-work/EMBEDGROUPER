"""Text normalization and cleaning for company names."""

import re
from typing import Dict, List, Tuple


# Legal suffixes to remove
LEGAL_SUFFIXES = [
    r'\binc\b', r'\bllc\b', r'\bltd\b', r'\bcorp\b', r'\bcorporation\b',
    r'\bco\b', r'\bplc\b', r'\bgmbh\b', r'\bs\.a\.\b', r'\bsa\b',
    r'\bpty\b', r'\bbv\b', r'\bag\b', r'\blp\b', r'\bllp\b',
    r'\bllc\.\b', r'\binc\.\b', r'\bcorp\.\b', r'\bltd\.\b', r'\bco\.\b'
]

# Filler words to remove
FILLER_WORDS = [
    r'\bthe\b', r'\bcompany\b', r'\bcompanies\b', r'\bgroup\b',
    r'\bholdings\b', r'\bholding\b'
]

# Abbreviation expansions
ABBREVIATIONS = {
    r'\bintl\b': 'international',
    r'\binternatl\b': 'international',
    r'\bint\'l\b': 'international',
    r'\bmfg\b': 'manufacturing',
    r'\bmfgr\b': 'manufacturer',
    r'\bassoc\b': 'association',
    r'\bassn\b': 'association',
    r'\bcorp\b': 'corporation',
    r'\bco\b': 'company',
}


def normalize_company_name(name: str) -> str:
    """
    Normalize a company name by:
    1. Converting to lowercase
    2. Removing punctuation and accents
    3. Removing legal suffixes
    4. Removing filler words
    5. Expanding abbreviations
    6. Cleaning whitespace
    
    Args:
        name: Original company name
        
    Returns:
        Normalized company name
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    normalized = name.lower()
    
    # Remove accents (basic normalization)
    normalized = normalized.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
    normalized = normalized.replace('á', 'a').replace('à', 'a').replace('â', 'a')
    normalized = normalized.replace('í', 'i').replace('ì', 'i').replace('î', 'i')
    normalized = normalized.replace('ó', 'o').replace('ò', 'o').replace('ô', 'o')
    normalized = normalized.replace('ú', 'u').replace('ù', 'u').replace('û', 'u')
    normalized = normalized.replace('ñ', 'n').replace('ç', 'c')
    
    # Expand abbreviations
    for abbrev, expansion in ABBREVIATIONS.items():
        normalized = re.sub(abbrev, expansion, normalized, flags=re.IGNORECASE)
    
    # Remove legal suffixes
    for suffix in LEGAL_SUFFIXES:
        normalized = re.sub(suffix, '', normalized, flags=re.IGNORECASE)
    
    # Remove filler words
    for filler in FILLER_WORDS:
        normalized = re.sub(filler, '', normalized, flags=re.IGNORECASE)
    
    # Remove punctuation (keep alphanumeric and spaces)
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    
    # Clean up whitespace (multiple spaces to single, strip)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()
    
    return normalized


def normalize_company_names(names: List[str]) -> Tuple[List[str], Dict[int, str]]:
    """
    Normalize a list of company names and return mapping to originals.
    
    Args:
        names: List of original company names
        
    Returns:
        Tuple of (normalized_names, original_mapping)
        where original_mapping maps normalized index to original name
    """
    normalized = []
    original_mapping = {}
    
    for idx, name in enumerate(names):
        norm_name = normalize_company_name(name)
        normalized.append(norm_name)
        original_mapping[idx] = name
    
    return normalized, original_mapping

