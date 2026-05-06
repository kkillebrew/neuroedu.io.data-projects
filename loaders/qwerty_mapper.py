# =====================================================================
# TYPO TAXONOMY ENGINE: qwerty_mapper.py
# Architecture Note: Lives in Repo B ('loaders/').
# Purpose: Calculates mathematical Euclidean distances for fat-finger 
# detection and provides lookup tables for cognitive misfires.
# =====================================================================
import math

# 1. The QWERTY Cartesian Plane
# Maps every standard lowercase key to an (x, y) coordinate.
# Row offsets mimic the physical stagger of a standard keyboard.
QWERTY_MAP = {
    # Number Row (y = 0)
    '1': (0.0, 0), '2': (1.0, 0), '3': (2.0, 0), '4': (3.0, 0), '5': (4.0, 0),
    '6': (5.0, 0), '7': (6.0, 0), '8': (7.0, 0), '9': (8.0, 0), '0': (9.0, 0),
    '-': (10.0, 0), '=': (11.0, 0),
    
    # Top Letter Row (y = 1) - Baseline X offset
    'q': (0.0, 1), 'w': (1.0, 1), 'e': (2.0, 1), 'r': (3.0, 1), 't': (4.0, 1), 
    'y': (5.0, 1), 'u': (6.0, 1), 'i': (7.0, 1), 'o': (8.0, 1), 'p': (9.0, 1),
    '[': (10.0, 1), ']': (11.0, 1), '\\': (12.0, 1),
    
    # Home Row (y = 2) - Shifted right by 0.5 units
    'a': (0.5, 2), 's': (1.5, 2), 'd': (2.5, 2), 'f': (3.5, 2), 'g': (4.5, 2), 
    'h': (5.5, 2), 'j': (6.5, 2), 'k': (7.5, 2), 'l': (8.5, 2), ';': (9.5, 2), 
    "'": (10.5, 2),
    
    # Bottom Letter Row (y = 3) - Shifted right by 1.0 units
    'z': (1.0, 3), 'x': (2.0, 3), 'c': (3.0, 3), 'v': (4.0, 3), 'b': (5.0, 3), 
    'n': (6.0, 3), 'm': (7.0, 3), ',': (8.0, 3), '.': (9.0, 3), '/': (10.0, 3),
    
    # Spacebar (y = 4) - Approximating the center of mass
    ' ': (5.0, 4)
}

# 2. The Cognitive Substitution Dictionary
# Maps Intended letters to common Mental Misfires (Visual, Phonetic, Dyslexic)
COGNITIVE_MAP = {
    # Phonetic Swaps (Homophones)
    'c': ['k', 's'], 'k': ['c'], 's': ['c', 'z'], 'z': ['s'],
    'f': ['v', 'ph'], 'v': ['f'],
    
    # Visual Homoglyphs
    'o': ['0', 'O', 'p'], '0': ['o', 'O'],
    'l': ['1', 'I', 'i'], '1': ['l', 'I'], 'i': ['l', '1'],
    
    # Dyslexic / Symmetric Inversions
    'b': ['d', 'p'], 'd': ['b', 'q'], 
    'p': ['q', 'b'], 'q': ['p', 'd'],
    'm': ['n', 'w'], 'n': ['m', 'h'], 'w': ['m', 'v'],
    
    # Vowel Confusion (Mental approximations)
    'a': ['e', 'o'], 'e': ['a', 'i'], 'i': ['e', 'y'], 
    'o': ['a', 'u'], 'u': ['o']
}

# --- Core Engine Functions ---

def get_euclidean_distance(char1, char2):
    """
    Calculates the physical distance between two keys on the keyboard.
    Returns: Float distance, or None if characters aren't in the map.
    """
    # Lowercase everything to handle Shift errors easily
    c1, c2 = str(char1).lower(), str(char2).lower()
    
    if c1 not in QWERTY_MAP or c2 not in QWERTY_MAP:
        return None
        
    x1, y1 = QWERTY_MAP[c1]
    x2, y2 = QWERTY_MAP[c2]
    
    # Standard Pythagorean theorem: a^2 + b^2 = c^2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def classify_typo(intended, typed):
    """
    Ingests an error pair and applies the Phase 1B Taxonomy.
    Returns: 'Spatial', 'Cognitive', or 'Unknown'
    """
    # 1. Check for Category A: Spatial Error (Fat-Fingering)
    # If the Euclidean distance is <= sqrt(2) (approx 1.414), it's an adjacent key.
    dist = get_euclidean_distance(intended, typed)
    
    if dist is not None and dist <= 1.45:
        return 'Category A: Spatial (Proximity)'
        
    # 2. Check for Category B: Cognitive Error (Mental Misfire)
    i_lower = str(intended).lower()
    t_lower = str(typed).lower()
    
    if i_lower in COGNITIVE_MAP and t_lower in COGNITIVE_MAP[i_lower]:
        return 'Category B: Cognitive (Substitution)'
        
    # If it fails both, it's either an unmapped edge case or potentially 
    # a Category C: Motor Transposition error (which requires N-gram context to detect, 
    # so we return 'Unknown' here at the single-character level).
    return 'Unknown / Requires Sequence Context'

# Example usage/testing block
if __name__ == "__main__":
    print(f"Distance 'a' to 's': {get_euclidean_distance('a', 's'):.2f}")
    print(f"Classification 'a' -> 's': {classify_typo('a', 's')}")
    
    print(f"\nDistance 'c' to 'k': {get_euclidean_distance('c', 'k'):.2f}")
    print(f"Classification 'c' -> 'k': {classify_typo('c', 'k')}")