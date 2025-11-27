import random
import config  # Importiamo il file di configurazione

class StrategyIndividual:
    """
    GENETIC ALGORITHM CHROMOSOME
    """
    def __init__(self, tyre_models, total_laps, stints=None, pit_loss=config.DEFAULT_PIT_LOSS):
        self.tyre_models = tyre_models
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        if stints:
            self.genes = stints
        else:
            self.genes = self._random_init()
        self.fitness = 0.0

    def _random_init(self):
        n_stops = random.randint(1, 3) 
        possible_cuts = list(range(1, self.total_laps))
        
        if possible_cuts:
            n_stops = min(n_stops, len(possible_cuts))
            cuts = sorted(random.sample(possible_cuts, n_stops))
        else:
            cuts = []
        
        stints = []
        prev_cut = 0
        for cut in cuts:
            comp = random.choice(list(self.tyre_models.keys()))
            stints.append([comp, cut - prev_cut]) 
            prev_cut = cut
        
        comp = random.choice(list(self.tyre_models.keys()))
        stints.append([comp, self.total_laps - prev_cut])
        return stints

    def calculate_fitness(self):
        """ OBJECTIVE FUNCTION """
        total_time = 0.0
        compounds_used = set(s[0] for s in self.genes)
        penalty = 0
        if len(compounds_used) < 2:
            penalty = 1000.0 

        for i, (comp, laps) in enumerate(self.genes):
            model = self.tyre_models[comp]
            
            # 1. Linear Component + Non-linear Component
            linear_time = (model['base_pace'] * laps) + (model['degradation'] * (laps * (laps - 1) / 2))
            
            wear_factor = config.NON_LINEAR_WEAR.get(comp, 0.002)
            n = laps
            sum_squares = ((n - 1) * n * (2 * n - 1)) / 6
            nonlinear_time = wear_factor * sum_squares
            
            stint_time = linear_time + nonlinear_time
            
            # 2. Logistic and physics penalties
            if i > 0: 
                traffic_laps = min(3, laps)
                stint_time += traffic_laps * 1.5 
                
                # Warm-up da config
                w_pen = config.WARMUP_PENALTY.get(comp, 3.0)
                stint_time += w_pen 

                if laps < 10:
                    stint_time += (10 - laps) * 4.0 

            # Limite massimo da config
            limit = config.MAX_LIFE.get(comp, 40)
            if laps > limit:
                over_limit = laps - limit
                stint_time += over_limit * 20.0 

            total_time += stint_time
            if i < len(self.genes) - 1:
                total_time += self.pit_loss

        self.fitness = total_time + penalty
        return self.fitness

class GeneticOptimizer:
    # Usiamo i default da GA_SETTINGS se non specificati
    def __init__(self, tyre_models, total_laps, 
                 pop_size=config.GA_SETTINGS['POP_SIZE'], 
                 generations=config.GA_SETTINGS['GENERATIONS'], 
                 mutation_rate=config.GA_SETTINGS['MUTATION_RATE'], 
                 pit_loss=config.DEFAULT_PIT_LOSS):
        self.tyre_models = tyre_models
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_history = []

    def run(self):
        self.population = [StrategyIndividual(self.tyre_models, self.total_laps, pit_loss=self.pit_loss) for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            for ind in self.population:
                ind.calculate_fitness()
            
            self.population.sort(key=lambda x: x.fitness)
            self.best_history.append(self.population[0].fitness)
            
            next_gen = self.population[:2]
            while len(next_gen) < self.pop_size:
                p1 = self._tournament()
                p2 = self._tournament()
                child = self._crossover(p1, p2)
                self._mutate(child)
                next_gen.append(child)
            self.population = next_gen
            
        return self.population[0]

    def _tournament(self):
        return min(random.sample(self.population, 3), key=lambda x: x.fitness)

    def _crossover(self, p1, p2):
        new_genes = []
        for i, (comp, laps) in enumerate(p1.genes):
            new_comp = p2.genes[i % len(p2.genes)][0] if random.random() > 0.5 else comp
            new_genes.append([new_comp, laps])
        return StrategyIndividual(self.tyre_models, self.total_laps, stints=new_genes, pit_loss=self.pit_loss)

    def _mutate(self, ind):
        if random.random() < self.mutation_rate:
            if random.random() < 0.5:
                idx = random.randint(0, len(ind.genes)-1)
                ind.genes[idx][0] = random.choice(list(self.tyre_models.keys()))
            elif len(ind.genes) > 1:
                idx = random.randint(0, len(ind.genes)-2)
                transfer = random.randint(-2, 2)
                if ind.genes[idx][1] + transfer > 1 and ind.genes[idx+1][1] - transfer > 1:
                    ind.genes[idx][1] += transfer
                    ind.genes[idx+1][1] -= transfer

class GreedySolver:
    """
    SMART GREEDY (EVALUATIVE)
    """
    def __init__(self, tyre_models, total_laps, pit_loss=config.DEFAULT_PIT_LOSS):
        self.tyre_models = tyre_models
        self.total_laps = total_laps
        self.pit_loss = pit_loss

    def solve(self):
        current_compound = min(self.tyre_models, key=lambda k: self.tyre_models[k]['base_pace'])
        current_tyre_age = 0
        stints = []
        total_time = 0
        compounds_used = {current_compound} 
        
        pit_threshold_loss = 2.5     
        traffic_fear_factor = 1.5    
        stint_start_lap = 0
        
        for lap in range(1, self.total_laps + 1):
            model = self.tyre_models[current_compound]
            
            # Lap Time Calculation
            wear_factor = config.NON_LINEAR_WEAR.get(current_compound, 0.002)
            linear_deg = model['degradation'] * current_tyre_age
            nonlinear_deg = wear_factor * (current_tyre_age ** 2)
            lap_time = model['base_pace'] + linear_deg + nonlinear_deg
            
            current_tyre_age += 1
            
            # Pit Stop Conditions (Limits from config)
            limit = config.MAX_LIFE.get(current_compound, 40)
            is_unsafe = current_tyre_age >= limit 
            is_slow = (lap_time > model['base_pace'] + pit_threshold_loss + traffic_fear_factor)
            
            laps_remaining = self.total_laps - lap
            must_change = (laps_remaining <= 2) and (len(compounds_used) < 2)
            
            if (is_unsafe or is_slow or must_change) and (laps_remaining > 0):
                stints.append([current_compound, lap - stint_start_lap])
                total_time += self.pit_loss
                
                # --- SMARTEST CHOICE FOR NEXT TYRES ---
                candidates = list(self.tyre_models.keys())
                if must_change:
                    candidates = [c for c in candidates if c not in compounds_used]
                    if not candidates: candidates = list(self.tyre_models.keys())

                best_candidate = None
                best_predicted_time = float('inf')
                
                prediction_horizon = min(20, laps_remaining)
                
                for cand in candidates:
                    cand_model = self.tyre_models[cand]
                    cand_wear = config.NON_LINEAR_WEAR.get(cand, 0.002)
                    
                    # Initial Warm-up cost (From config)
                    w_pen = config.WARMUP_PENALTY.get(cand, 3.0)
                    predicted_time = w_pen 
                    
                    # Simulate next stint
                    for t in range(prediction_horizon):
                        lin = cand_model['degradation'] * t
                        non_lin = cand_wear * (t**2)
                        predicted_time += cand_model['base_pace'] + lin + non_lin
                    
                    if predicted_time < best_predicted_time:
                        best_predicted_time = predicted_time
                        best_candidate = cand
                
                current_compound = best_candidate
                compounds_used.add(current_compound)
                current_tyre_age = 0
                stint_start_lap = lap

                # --- WARM UP PENALTIES ---
                total_time += config.WARMUP_PENALTY.get(current_compound, 3.0)
                traffic_laps = min(3, laps_remaining)
                total_time += traffic_laps * 1.5
                
            total_time += lap_time
            
        stints.append([current_compound, self.total_laps - stint_start_lap])
        return total_time, stints