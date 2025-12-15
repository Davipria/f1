import random
import config

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
        # MODIFICA: Rispetta il limite massimo di soste del GA
        max_stops = config.GA_SETTINGS['MAX_STOPS']
        n_stops = random.randint(1, max_stops) 
        
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
        total_time = 0.0
        compounds_used = set(s[0] for s in self.genes)
        penalty = 0
        if len(compounds_used) < 2:
            penalty = 1000.0 

        for i, (comp, laps) in enumerate(self.genes):
            model = self.tyre_models[comp]
            
            # --- PHYSICAL MODEL ---
            n = laps
            sum_t = (n - 1) * n / 2
            sum_t_squared = ((n - 1) * n * (2 * n - 1)) / 6
            
            base_time = model['base_pace'] * n
            linear_time = model['linear_degradation'] * sum_t
            quadratic_time = model['quadratic_degradation'] * sum_t_squared
            
            stint_time = base_time + linear_time + quadratic_time
            
            # --- PENALTIES ---
            if i > 0:
                traffic_laps = min(3, laps)
                stint_time += traffic_laps * 1.5 
                
                w_pen = config.WARMUP_PENALTY.get(comp, 3.0)
                stint_time += w_pen 

                if laps < 10:
                    stint_time += (10 - laps) * 4.0 

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
    def __init__(self, tyre_models, total_laps, 
                 pop_size=config.GA_SETTINGS['POP_SIZE'], 
                 generations=config.GA_SETTINGS['GENERATIONS'], 
                 mutation_rate=config.GA_SETTINGS['MUTATION_RATE'], 
                 pit_loss=config.DEFAULT_PIT_LOSS,
                 crossover_type='mixed'): 
        self.tyre_models = tyre_models
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.pop_size = pop_size
        self.generations = generations
        self.base_mutation_rate = mutation_rate
        self.mutation_rate = mutation_rate
        self.crossover_type = crossover_type
        self.population = []
        self.best_history = []
        self.current_generation = 0
        
        # Carica il limite dal config (Max Stints = Max Stops + 1)
        self.max_stints = config.GA_SETTINGS['MAX_STOPS'] + 1

    def run(self):
        self.population = [
            StrategyIndividual(self.tyre_models, self.total_laps, pit_loss=self.pit_loss) 
            for _ in range(self.pop_size)
        ]
        
        for gen in range(self.generations):
            self.current_generation = gen
            
            for ind in self.population:
                ind.calculate_fitness()
            
            self.population.sort(key=lambda x: x.fitness)
            self.best_history.append(self.population[0].fitness)
            
            progress = gen / self.generations
            self.mutation_rate = self.base_mutation_rate * (1 - progress ** 2)
            
            next_gen = self.population[:3]
            
            while len(next_gen) < self.pop_size:
                p1 = self._tournament()
                p2 = self._tournament()
                
                if self.crossover_type == 'one_point':
                    child = self._one_point_crossover(p1, p2)
                elif self.crossover_type == 'uniform':
                    child = self._uniform_crossover(p1, p2)
                else:
                    if random.random() < 0.7:
                        child = self._one_point_crossover(p1, p2)
                    else:
                        child = self._uniform_crossover(p1, p2)
                
                self._repair(child)
                self._mutate(child)
                self._repair(child)
                
                next_gen.append(child)
            
            self.population = next_gen
            
        return self.population[0]

    def _tournament(self, tournament_size=3):
        contestants = random.sample(self.population, tournament_size)
        return min(contestants, key=lambda x: x.fitness)

    def _one_point_crossover(self, p1, p2):
        cut_lap = random.randint(5, self.total_laps - 5)
        child_stints = []
        
        current_lap = 0
        for comp, laps in p1.genes:
            if current_lap + laps <= cut_lap:
                child_stints.append([comp, laps])
                current_lap += laps
            else:
                remaining_before_cut = cut_lap - current_lap
                if remaining_before_cut > 0:
                    child_stints.append([comp, remaining_before_cut])
                break
        
        p2_lap_counter = 0
        for comp, laps in p2.genes:
            p2_lap_counter += laps
            if p2_lap_counter > cut_lap:
                laps_after_cut = p2_lap_counter - cut_lap
                remaining_race_laps = self.total_laps - cut_lap
                laps_to_add = min(laps_after_cut, remaining_race_laps)
                
                if laps_to_add > 0:
                    child_stints.append([comp, laps_to_add])
                    cut_lap += laps_to_add
                
                for comp2, laps2 in p2.genes[p2.genes.index([comp, laps]) + 1:]:
                    remaining_race_laps = self.total_laps - cut_lap
                    if remaining_race_laps <= 0:
                        break
                    laps_to_add = min(laps2, remaining_race_laps)
                    if laps_to_add > 0:
                        child_stints.append([comp2, laps_to_add])
                        cut_lap += laps_to_add
                break
        
        return StrategyIndividual(self.tyre_models, self.total_laps, 
                                  stints=child_stints, pit_loss=self.pit_loss)

    def _uniform_crossover(self, p1, p2):
        child_stints = []
        remaining_laps = self.total_laps
        all_stints = p1.genes + p2.genes
        compounds = list(self.tyre_models.keys())
        
        while remaining_laps > 0:
            if random.random() < 0.7 and all_stints:
                parent = random.choice([p1, p2])
                if parent.genes:
                    reference_stint = random.choice(parent.genes)
                    target_length = reference_stint[1]
                else:
                    target_length = random.randint(1, max(1, remaining_laps))
            else:
                min_length = min(10, remaining_laps)
                max_length = min(30, remaining_laps)
                if min_length <= max_length:
                    target_length = random.randint(min_length, max_length)
                else:
                    target_length = remaining_laps
            
            stint_length = min(target_length, remaining_laps)
            if stint_length < 1: stint_length = remaining_laps
            
            if random.random() < 0.8:
                parent_compounds = [s[0] for s in p1.genes + p2.genes]
                compound = random.choice(parent_compounds)
            else:
                compound = random.choice(compounds)
            
            child_stints.append([compound, stint_length])
            remaining_laps -= stint_length
            
            # MODIFICA: Controllo sul numero massimo di stint anche qui
            if len(child_stints) >= self.max_stints:
                if remaining_laps > 0:
                    # Se abbiamo raggiunto il limite, forziamo l'ultimo stint
                    child_stints[-1][1] += remaining_laps
                break
        
        return StrategyIndividual(self.tyre_models, self.total_laps, 
                                  stints=child_stints, pit_loss=self.pit_loss)

    def _repair(self, ind):
        ind.genes = [[c, l] for c, l in ind.genes if l > 0]
        
        if len(ind.genes) == 0:
            comp = random.choice(list(self.tyre_models.keys()))
            ind.genes = [[comp, self.total_laps]]
            return
        
        merged = []
        for comp, laps in ind.genes:
            if merged and merged[-1][0] == comp:
                merged[-1][1] += laps
            else:
                merged.append([comp, laps])
        ind.genes = merged
        
        current_total = sum(s[1] for s in ind.genes)
        diff = self.total_laps - current_total
        
        if diff != 0:
            if len(ind.genes) == 1:
                ind.genes[0][1] += diff
            else:
                longest_idx = max(range(len(ind.genes)), key=lambda i: ind.genes[i][1])
                ind.genes[longest_idx][1] += diff
                if ind.genes[longest_idx][1] <= 0:
                    ind.genes[longest_idx][1] = 1
                    remaining_diff = self.total_laps - sum(s[1] for s in ind.genes)
                    ind.genes[-1][1] += remaining_diff
        
        ind.genes = [[c, max(1, l)] for c, l in ind.genes]

    def _mutate(self, ind):
        if random.random() < self.mutation_rate:
            mutation_type = random.random()
            
            if mutation_type < 0.5:
                # Swap
                if len(ind.genes) > 0:
                    idx = random.randint(0, len(ind.genes) - 1)
                    ind.genes[idx][0] = random.choice(list(self.tyre_models.keys()))
            
            elif mutation_type < 0.8:
                # Transfer
                if len(ind.genes) > 1:
                    idx = random.randint(0, len(ind.genes) - 2)
                    transfer = random.randint(-3, 3)
                    if ind.genes[idx][1] + transfer >= 1 and ind.genes[idx + 1][1] - transfer >= 1:
                        ind.genes[idx][1] += transfer
                        ind.genes[idx + 1][1] -= transfer
            
            else:
                # MODIFICA: Split Stint controlla il limite max_stints
                if len(ind.genes) > 0 and len(ind.genes) < self.max_stints: 
                    idx = random.randint(0, len(ind.genes) - 1)
                    original_length = ind.genes[idx][1]
                    
                    if original_length > 10:
                        split_point = random.randint(5, original_length - 5)
                        old_comp = ind.genes[idx][0]
                        new_comp = random.choice(list(self.tyre_models.keys()))
                        
                        remaining_laps = original_length - split_point
                        ind.genes[idx] = [old_comp, split_point]
                        ind.genes.insert(idx + 1, [new_comp, remaining_laps])

class GreedySolver:
    """
    SMART GREEDY (EVALUATIVE)
    """
    def __init__(self, tyre_models, total_laps, pit_loss=config.DEFAULT_PIT_LOSS):
        self.tyre_models = tyre_models
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        # Carica il limite massimo di soste dal config
        self.max_stops = config.GREEDY_SETTINGS.get('MAX_STOPS', 3)

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
            
            t = current_tyre_age
            lap_time = (model['base_pace'] + 
                       model['linear_degradation'] * t + 
                       model['quadratic_degradation'] * (t ** 2))
            
            current_tyre_age += 1
            
            limit = config.MAX_LIFE.get(current_compound, 40)
            is_unsafe = current_tyre_age >= limit 
            is_slow = (lap_time > model['base_pace'] + pit_threshold_loss + traffic_fear_factor)
            
            laps_remaining = self.total_laps - lap
            must_change = (laps_remaining <= 2) and (len(compounds_used) < 2)
            
            # Conta quante soste sono state fatte finora
            stops_made = len(stints)
            can_pit_for_pace = stops_made < self.max_stops
            
            should_pit = (is_unsafe or must_change or (is_slow and can_pit_for_pace))
            
            if should_pit and (laps_remaining > 0):
                stints.append([current_compound, lap - stint_start_lap])
                total_time += self.pit_loss
                
                candidates = list(self.tyre_models.keys())
                if must_change:
                    candidates = [c for c in candidates if c not in compounds_used]
                    if not candidates: candidates = list(self.tyre_models.keys())

                best_candidate = None
                best_predicted_time = float('inf')
                
                prediction_horizon = min(20, laps_remaining)
                
                for cand in candidates:
                    cand_model = self.tyre_models[cand]
                    w_pen = config.WARMUP_PENALTY.get(cand, 3.0)
                    predicted_time = w_pen 
                    
                    for t in range(prediction_horizon):
                        lap_cost = (cand_model['base_pace'] + 
                                   cand_model['linear_degradation'] * t + 
                                   cand_model['quadratic_degradation'] * (t ** 2))
                        predicted_time += lap_cost
                    
                    if predicted_time < best_predicted_time:
                        best_predicted_time = predicted_time
                        best_candidate = cand
                
                current_compound = best_candidate
                compounds_used.add(current_compound)
                current_tyre_age = 0
                stint_start_lap = lap

                total_time += config.WARMUP_PENALTY.get(current_compound, 3.0)
                traffic_laps = min(3, laps_remaining)
                total_time += traffic_laps * 1.5
                
            total_time += lap_time
            
        stints.append([current_compound, self.total_laps - stint_start_lap])
        return total_time, stints