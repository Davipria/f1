# --- DEFAULT PIT STOP LOSS ---
DEFAULT_PIT_LOSS = 23.0

# --- 1. STRUCTURAL LIMITS (Max tire life) ---
MAX_LIFE = {
    'SOFT': 18,    
    'MEDIUM': 28,
    'HARD': 45     
}

# --- 2. NON-LINEAR PHYSICAL WEAR (The "Cliff") ---
NON_LINEAR_WEAR = {
    'SOFT': 0.005,    
    'MEDIUM': 0.002,  
    'HARD': 0.001     
}

# --- 3. TERMIC WARM-UP ---
WARMUP_PENALTY = {
    'SOFT': 0.5,   
    'MEDIUM': 1.5, 
    'HARD': 4.5    
}

# --- GENETIC ALGORITHMN CONFIGURATION ---
GA_SETTINGS = {
    'POP_SIZE': 80,
    'GENERATIONS': 60,
    'MUTATION_RATE': 0.25
}

RANDOM_SEED = 42
