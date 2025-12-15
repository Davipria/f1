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

# --- GENETIC ALGORITHM CONFIGURATION ---
GA_SETTINGS = {
    'POP_SIZE': 100,
    'GENERATIONS': 100,
    'MUTATION_RATE': 0.3, 
    'CROSSOVER_TYPE': 'mixed',
    'MAX_STOPS': 3  
}

# --- GREEDY ALGORITHM CONFIGURATION ---
GREEDY_SETTINGS = {
    'PIT_THRESHOLD': 2.5,     # Seconds of degradation before considering pit stop
    'TRAFFIC_FEAR': 1.5,      # Penalty per lap in traffic (dirty air)
    'PREDICTION_HORIZON': 20, # How many laps to simulate when choosing compound
    'MAX_STOPS': 3          
}

RANDOM_SEED = 42