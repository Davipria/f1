import matplotlib.pyplot as plt
import fastf1
import pandas as pd
from data_model import TyreDataModeler
from optimizers import GreedySolver, GeneticOptimizer

def check_legality(strategy):
    """Checks if the strategy respects the 2-compound rule."""
    compounds = set(s[0] for s in strategy)
    if len(compounds) < 2:
        return "ILLEGAL (DSQ: < 2 compounds)"
    return "VALID"

def get_user_input():
    """
    Handles mandatory interactive input.
    No defaults allowed: the user MUST choose.
    """
    print("\n==========================================")
    print("      F1 SIMULATION CONFIGURATION         ")
    print("==========================================")
    
    # --- 1. YEAR SELECTION ---
    while True:
        try:
            str_year = input("\nEnter Year: ").strip()
            
            if str_year == "":
                print("Error: You must enter a year.")
                continue
                
            year = int(str_year)
            
            # Validity check (FastF1 has good data from 2018 onwards)
            if year < 2018 or year > 2025:
                print(f"Warning: Year {year} might not have complete data.")
                confirm = input("Do you want to continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            break
        except ValueError:
            print("Error: Please enter a valid numeric year (e.g. 2023).")

    # --- 2. RETRIEVING CALENDAR ---
    print(f"\nDownloading {year} calendar from FastF1...")
    try:
        # Download calendar 
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = schedule[schedule['RoundNumber'] > 0][['EventName', 'Location', 'RoundNumber']].reset_index(drop=True)
        
        if races.empty:
            print(f"No races found for year {year}.")
            return get_user_input()

    except Exception as e:
        print(f"Critical error downloading calendar: {e}")
        exit()

    # --- 3. RACE SELECTION MENU ---
    print(f"\nAvailable races for {year} season:")
    print("-" * 50)
    for idx, row in races.iterrows():
        print(f"{idx + 1:2}. {row['EventName']} ({row['Location']})")
    print("-" * 50)

    # --- 4. USER CHOICE ---
    while True:
        try:
            sel_str = input(f"\nChoose race number (1-{len(races)}): ").strip()
            
            if sel_str == "":
                print("Error: You must enter a number.")
                continue
                
            selection = int(sel_str)
            
            if 1 <= selection <= len(races):
                selected_gp = races.iloc[selection - 1]['EventName']
                print(f"--> You selected: {selected_gp.upper()}")
                return year, selected_gp
            else:
                print(f"Error: Number must be between 1 and {len(races)}.")
        except ValueError:
            print("Error: Please enter a valid number.")

def main():
    # PHASE 0: INTERACTIVE INPUT
    race_year, race_gp = get_user_input()

    print("\n==========================================")
    print(f"   STARTING SIMULATION: {race_gp.upper()} {race_year}   ")
    print("==========================================")

    # PHASE 1: DATA PROCESS
    print(f"\n[1/3] Extracting Telemetry Data...")
    
    try:
        data_engine = TyreDataModeler(race_year, race_gp)
        data_engine.load_and_clean_data()
        data_engine.analyze_degradation()
        
        real_tyre_models, total_laps, dynamic_pit_loss = data_engine.get_simulation_data()
        
    except Exception as e:
        print(f"\n[ERROR] Could not load race data: {e}")
        print("Tip: Check internet connection or try another race.")
        return

    print(f"\nExtracted Parameters for {race_gp}:")
    print(f"Total Laps: {total_laps}")
    print(f"Dynamic Pit Loss: {dynamic_pit_loss:.2f}s")
    for k, v in real_tyre_models.items():
        print(f"{k}: Base={v['base_pace']:.2f}s, Deg={v['degradation']:.3f}s/lap")

    # PHASE 2: GREEDY ALGORITHM
    print("\n[2/3] Running Greedy Algorithm...")
    greedy = GreedySolver(real_tyre_models, total_laps, pit_loss=dynamic_pit_loss)
    greedy_time, greedy_stints = greedy.solve()
    
    print(f"Greedy Time: {greedy_time:.2f}s")
    print(f"Greedy Strategy: {greedy_stints} -> {check_legality(greedy_stints)}")

    # PHASE 3: GENETIC ALGORITHM
    print("\n[3/3] Running Genetic Algorithm...")
    ga = GeneticOptimizer(
        tyre_models=real_tyre_models, 
        total_laps=total_laps,
        pop_size=80,       
        generations=60,    
        mutation_rate=0.25,
        pit_loss=dynamic_pit_loss
    )
    
    best_solution = ga.run()
    
    print(f"Best GA Time: {best_solution.fitness:.2f}s")
    print(f"GA Strategy: {best_solution.genes} -> {check_legality(best_solution.genes)}")
    
    improvement = greedy_time - best_solution.fitness
    print(f"\n>>> STRATEGIC GAIN: {improvement:.2f} seconds <<<")

    # PHASE 4: VISUALIZATION
    plt.figure(figsize=(10, 6))
    plt.plot(ga.best_history, label='Genetic Evolution', color='blue', linewidth=2)
    plt.axhline(y=greedy_time, color='red', linestyle='--', label='Enhanced Greedy', linewidth=2)
    plt.title(f"Optimization Analysis: {race_gp} {race_year}")
    plt.xlabel("Generations")
    plt.ylabel("Total Race Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("\nChart generation completed.")
    plt.show()

if __name__ == "__main__":
    main()