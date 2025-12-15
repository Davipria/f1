import random
import config

class StrategyIndividual:
    def __init__(self, tyre_models, total_laps, stints=None, pit_loss=config.DEFAULT_PIT_LOSS):
        self.tyre_models = tyre_models
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.genes = stints or self._random_init()
        self.fitness = 0.0

    def _random_init(self):
        max_stops = config.GA_SETTINGS['MAX_STOPS']
        possible_cuts = list(range(1, self.total_laps))
        n_stops = min(random.randint(1, max_stops), len(possible_cuts)) if possible_cuts else 0
        cuts = sorted(random.sample(possible_cuts, n_stops)) if n_stops else []
        
        stints, prev = [], 0
        for cut in cuts:
            stints.append([random.choice(list(self.tyre_models.keys())), cut - prev])
            prev = cut
        stints.append([random.choice(list(self.tyre_models.keys())), self.total_laps - prev])
        return stints

    def calculate_fitness(self):
        total_time = 0.0
        compounds_used = {s[0] for s in self.genes}
        penalty = 1000.0 if len(compounds_used) < 2 else 0

        for i, (comp, laps) in enumerate(self.genes):
            model = self.tyre_models[comp]
            n = laps
            stint_time = (model['base_pace'] * n + 
                         model['linear_degradation'] * (n - 1) * n / 2 + 
                         model['quadratic_degradation'] * (n - 1) * n * (2 * n - 1) / 6)
            
            if i > 0:
                stint_time += min(3, laps) * 1.5 + config.WARMUP_PENALTY.get(comp, 3.0)
                if laps < 10:
                    stint_time += (10 - laps) * 4.0
            
            limit = config.MAX_LIFE.get(comp, 40)
            if laps > limit:
                stint_time += (laps - limit) * 20.0
            
            total_time += stint_time
            if i < len(self.genes) - 1:
                total_time += self.pit_loss

        self.fitness = total_time + penalty
        return self.fitness

class GeneticOptimizer:
    def __init__(self, tyre_models, total_laps, pop_size=config.GA_SETTINGS['POP_SIZE'], 
                 generations=config.GA_SETTINGS['GENERATIONS'], mutation_rate=config.GA_SETTINGS['MUTATION_RATE'], 
                 pit_loss=config.DEFAULT_PIT_LOSS, crossover_type='mixed'):
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
        self.max_stints = config.GA_SETTINGS['MAX_STOPS'] + 1

    def run(self):
        self.population = [StrategyIndividual(self.tyre_models, self.total_laps, pit_loss=self.pit_loss) 
                          for _ in range(self.pop_size)]
        
        for gen in range(self.generations):
            self.current_generation = gen
            for ind in self.population:
                ind.calculate_fitness()
            
            self.population.sort(key=lambda x: x.fitness)
            self.best_history.append(self.population[0].fitness)
            self.mutation_rate = self.base_mutation_rate * (1 - (gen / self.generations) ** 2)
            
            next_gen = self.population[:3]
            while len(next_gen) < self.pop_size:
                p1, p2 = self._tournament(), self._tournament()
                child = (self._one_point_crossover(p1, p2) if self.crossover_type == 'one_point' or 
                        (self.crossover_type == 'mixed' and random.random() < 0.7) else 
                        self._uniform_crossover(p1, p2)) if self.crossover_type != 'uniform' else self._uniform_crossover(p1, p2)
                
                self._repair(child)
                self._mutate(child)
                self._repair(child)
                next_gen.append(child)
            
            self.population = next_gen
        return self.population[0]

    def _tournament(self, tournament_size=3):
        return min(random.sample(self.population, tournament_size), key=lambda x: x.fitness)

    def _one_point_crossover(self, p1, p2):
        cut_lap = random.randint(5, self.total_laps - 5)
        child_stints, current_lap = [], 0
        
        for comp, laps in p1.genes:
            if current_lap + laps <= cut_lap:
                child_stints.append([comp, laps])
                current_lap += laps
            else:
                if (remaining := cut_lap - current_lap) > 0:
                    child_stints.append([comp, remaining])
                break
        
        p2_lap_counter = 0
        for idx, (comp, laps) in enumerate(p2.genes):
            p2_lap_counter += laps
            if p2_lap_counter > cut_lap:
                if (laps_to_add := min(p2_lap_counter - cut_lap, self.total_laps - cut_lap)) > 0:
                    child_stints.append([comp, laps_to_add])
                    cut_lap += laps_to_add
                
                for comp2, laps2 in p2.genes[idx + 1:]:
                    if (remaining := self.total_laps - cut_lap) <= 0:
                        break
                    if (laps_to_add := min(laps2, remaining)) > 0:
                        child_stints.append([comp2, laps_to_add])
                        cut_lap += laps_to_add
                break
        
        return StrategyIndividual(self.tyre_models, self.total_laps, stints=child_stints, pit_loss=self.pit_loss)

    def _uniform_crossover(self, p1, p2):
        child_stints, remaining_laps = [], self.total_laps
        compounds = list(self.tyre_models.keys())
        
        while remaining_laps > 0:
            if random.random() < 0.7 and (p1.genes or p2.genes):
                parent = random.choice([p1, p2])
                target_length = random.choice(parent.genes)[1] if parent.genes else random.randint(1, remaining_laps)
            else:
                min_len, max_len = min(10, remaining_laps), min(30, remaining_laps)
                target_length = random.randint(min_len, max_len) if min_len <= max_len else remaining_laps
            
            stint_length = max(1, min(target_length, remaining_laps))
            compound = (random.choice([s[0] for s in p1.genes + p2.genes]) if random.random() < 0.8 
                       else random.choice(compounds))
            
            child_stints.append([compound, stint_length])
            remaining_laps -= stint_length
            
            if len(child_stints) >= self.max_stints:
                if remaining_laps > 0:
                    child_stints[-1][1] += remaining_laps
                break
        
        return StrategyIndividual(self.tyre_models, self.total_laps, stints=child_stints, pit_loss=self.pit_loss)

    def _repair(self, ind):
        ind.genes = [[c, l] for c, l in ind.genes if l > 0]
        if not ind.genes:
            ind.genes = [[random.choice(list(self.tyre_models.keys())), self.total_laps]]
            return
        
        merged = []
        for comp, laps in ind.genes:
            if merged and merged[-1][0] == comp:
                merged[-1][1] += laps
            else:
                merged.append([comp, laps])
        ind.genes = merged
        
        diff = self.total_laps - sum(s[1] for s in ind.genes)
        if diff != 0:
            idx = 0 if len(ind.genes) == 1 else max(range(len(ind.genes)), key=lambda i: ind.genes[i][1])
            ind.genes[idx][1] += diff
            if ind.genes[idx][1] <= 0:
                ind.genes[idx][1] = 1
                ind.genes[-1][1] += self.total_laps - sum(s[1] for s in ind.genes)
        
        ind.genes = [[c, max(1, l)] for c, l in ind.genes]

    def _mutate(self, ind):
        if random.random() >= self.mutation_rate:
            return
        
        mt = random.random()
        if mt < 0.5 and ind.genes:
            ind.genes[random.randint(0, len(ind.genes) - 1)][0] = random.choice(list(self.tyre_models.keys()))
        elif mt < 0.8 and len(ind.genes) > 1:
            idx, transfer = random.randint(0, len(ind.genes) - 2), random.randint(-3, 3)
            if ind.genes[idx][1] + transfer >= 1 and ind.genes[idx + 1][1] - transfer >= 1:
                ind.genes[idx][1] += transfer
                ind.genes[idx + 1][1] -= transfer
        elif ind.genes and len(ind.genes) < self.max_stints:
            idx = random.randint(0, len(ind.genes) - 1)
            if ind.genes[idx][1] > 10:
                original_length = ind.genes[idx][1]
                split = random.randint(5, original_length - 5)
                old_comp, new_comp = ind.genes[idx][0], random.choice(list(self.tyre_models.keys()))
                ind.genes[idx] = [old_comp, split]
                ind.genes.insert(idx + 1, [new_comp, original_length - split])

class GreedySolver:
    def __init__(self, tyre_models, total_laps, pit_loss=config.DEFAULT_PIT_LOSS):
        self.tyre_models = tyre_models
        self.total_laps = total_laps
        self.pit_loss = pit_loss
        self.max_stops = config.GREEDY_SETTINGS.get('MAX_STOPS', 3)

    def _lap_time(self, model, age):
        return model['base_pace'] + model['linear_degradation'] * age + model['quadratic_degradation'] * (age ** 2)

    def solve(self):
        current_compound = min(self.tyre_models, key=lambda k: self.tyre_models[k]['base_pace'])
        current_tyre_age, stints, total_time, stint_start_lap = 0, [], 0, 0
        compounds_used = {current_compound}
        
        for lap in range(1, self.total_laps + 1):
            model = self.tyre_models[current_compound]
            lap_time = self._lap_time(model, current_tyre_age)
            current_tyre_age += 1
            
            laps_remaining = self.total_laps - lap
            is_unsafe = current_tyre_age >= config.MAX_LIFE.get(current_compound, 40)
            is_slow = lap_time > model['base_pace'] + 4.0
            must_change = laps_remaining <= 2 and len(compounds_used) < 2
            can_pit = len(stints) < self.max_stops
            
            if (is_unsafe or must_change or (is_slow and can_pit)) and laps_remaining > 0:
                stints.append([current_compound, lap - stint_start_lap])
                total_time += self.pit_loss
                
                candidates = ([c for c in self.tyre_models if c not in compounds_used] if must_change 
                             else list(self.tyre_models.keys()))
                if not candidates:
                    candidates = list(self.tyre_models.keys())
                
                horizon = min(20, laps_remaining)
                best_candidate = min(candidates, key=lambda c: config.WARMUP_PENALTY.get(c, 3.0) + 
                                    sum(self._lap_time(self.tyre_models[c], t) for t in range(horizon)))
                
                current_compound = best_candidate
                compounds_used.add(current_compound)
                current_tyre_age, stint_start_lap = 0, lap
                total_time += config.WARMUP_PENALTY.get(current_compound, 3.0) + min(3, laps_remaining) * 1.5
            
            total_time += lap_time
        
        stints.append([current_compound, self.total_laps - stint_start_lap])
        return total_time, stints