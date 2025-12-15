# --- DEFAULT PIT STOP LOSS ---
DEFAULT_PIT_LOSS = 23.0

# --- 1. STRUCTURAL LIMITS (Max tire life) ---
MAX_LIFE = {
    'SOFT': 18,    
    'MEDIUM': 28,
    'HARD': 45     
}

# --- 2. NON-LINEAR PHYSICAL WEAR (The "Cliff") ---
# NOTE: These are now FALLBACK values only.
# Real coefficients are extracted from telemetry via polynomial regression.
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
    'POP_SIZE': 80,
    'GENERATIONS': 60,
    'MUTATION_RATE': 0.25,  # Base rate - will adapt during evolution
    'CROSSOVER_TYPE': 'one_point'  # Options: 'one_point' or 'uniform'
}

# --- GREEDY ALGORITHM CONFIGURATION ---
GREEDY_SETTINGS = {
    'PIT_THRESHOLD': 2.5,     # Seconds of degradation before considering pit stop
    'TRAFFIC_FEAR': 1.5,      # Penalty per lap in traffic (dirty air)
    'PREDICTION_HORIZON': 20  # How many laps to simulate when choosing compound
}

RANDOM_SEED = 42