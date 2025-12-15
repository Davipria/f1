import random
import numpy as np
import fastf1
import config
from data_model import TyreDataModeler
from optimizers import GreedySolver, GeneticOptimizer
from visualization import plot_results

def check_legality(strategy):
    """Checks if strategy respects 2-compound rule."""
    return "VALID" if len(set(s[0] for s in strategy)) >= 2 else "ILLEGAL (DSQ: < 2 compounds)"

def get_user_input():
    """Handles interactive input for year and race selection."""
    print("\n" + "="*42)
    print("      F1 SIMULATION CONFIGURATION")
    print("="*42)
    
    # Year selection
    while True:
        try:
            year = int(input("\nEnter Year: ").strip())
            if year < 2018 or year > 2025:
                if input(f"Warning: Year {year} may lack data. Continue? (y/n): ").lower() != 'y':
                    continue
            break
        except ValueError:
            print("Error: Enter a valid year (e.g. 2023).")

    # Get calendar
    print(f"\nDownloading {year} calendar...")
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = schedule[schedule['RoundNumber'] > 0][['EventName', 'Location']].reset_index(drop=True)
        if races.empty:
            print(f"No races found for {year}.")
            return get_user_input()
    except Exception as e:
        print(f"Error downloading calendar: {e}")
        exit()

    # Race selection
    print(f"\nAvailable races for {year}:")
    print("-" * 50)
    for idx, row in races.iterrows():
        print(f"{idx + 1:2}. {row['EventName']} ({row['Location']})")
    print("-" * 50)

    while True:
        try:
            sel = int(input(f"\nChoose race (1-{len(races)}): ").strip())
            if 1 <= sel <= len(races):
                gp = races.iloc[sel - 1]['EventName']
                print(f"Selected: {gp.upper()}")
                return year, gp
            print(f"Error: Enter 1-{len(races)}.")
        except ValueError:
            print("Error: Enter a valid number.")

def main():
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    year, gp = get_user_input()
    
    print(f"\n{'='*42}")
    print(f"   STARTING: {gp.upper()} {year}")
    print("="*42)

    # Phase 1: Data extraction
    print("\n[1/3] Extracting Telemetry...")
    try:
        engine = TyreDataModeler(year, gp, visualize_fits=False)
        engine.load_and_clean_data()
        engine.analyze_degradation()
        engine.print_model_summary()
        
        models, total_laps, pit_loss = engine.get_simulation_data()
    except Exception as e:
        print(f"\n[ERROR] Could not load data: {e}")
        return

    print(f"\nParameters for {gp}:")
    print(f"Total Laps: {total_laps}, Pit Loss: {pit_loss:.2f}s")
    print("\nTyre Models:")
    print("-" * 60)
    for k, v in models.items():
        print(f"{k:<10} Base:{v['base_pace']:>6.2f}s Lin:{v['linear_degradation']:>7.4f} Quad:{v['quadratic_degradation']:>9.6f}")
    print("-" * 60)

    # Phase 2: Greedy
    print("\n[2/3] Running Greedy...")
    greedy = GreedySolver(models, total_laps, pit_loss=pit_loss)
    g_time, g_strat = greedy.solve()
    print(f"Greedy: {g_time:.2f}s | {g_strat} -> {check_legality(g_strat)}")

    # Phase 3: Genetic Algorithm
    print("\n[3/3] Running Genetic Algorithm...")
    ga = GeneticOptimizer(
        tyre_models=models, 
        total_laps=total_laps,
        pop_size=config.GA_SETTINGS['POP_SIZE'],
        generations=config.GA_SETTINGS['GENERATIONS'],
        mutation_rate=config.GA_SETTINGS['MUTATION_RATE'],
        pit_loss=pit_loss,
        crossover_type=config.GA_SETTINGS['CROSSOVER_TYPE']
    )
    
    best = ga.run()
    print(f"GA Best: {best.fitness:.2f}s | {best.genes} -> {check_legality(best.genes)}")
    
    gain = g_time - best.fitness
    print(f"\n{'='*60}")
    print(f">>> GAIN: {gain:.2f}s ({(gain/g_time)*100:.2f}%) <<<")
    print("="*60)

    # Phase 4: Visualization
    print("\nGenerating chart...")
    plot_results(ga.best_history, g_time, g_strat, best.genes, gp, year)
    print("Complete.")

if __name__ == "__main__":
    main()