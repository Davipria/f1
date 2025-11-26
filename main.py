import matplotlib.pyplot as plt
import fastf1
import pandas as pd
from data_model import TyreDataModeler
from optimizers import GreedySolver, GeneticOptimizer

def check_legality(strategy):
    compounds = set(s[0] for s in strategy)
    if len(compounds) < 2:
        return "ILLEGAL (DSQ: < 2 compounds)"
    return "VALID"

def get_user_input():
    print("\n==========================================")
    print("      CONFIGURAZIONE SIMULAZIONE F1       ")
    print("==========================================")
    
    # 1. Anno
    while True:
        try:
            str_year = input("\nInserisci Anno (es. 2024): ").strip()
            if str_year == "": continue
            year = int(str_year)
            if year < 2018 or year > 2025:
                if input("Anno fuori standard. Proseguire? (s/n): ").lower() != 's': continue
            break
        except ValueError: pass

    # 2. Calendario
    print(f"\nSto scaricando il calendario {year}...")
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = schedule[schedule['RoundNumber'] > 0][['EventName', 'Location']].reset_index(drop=True)
        if races.empty: return get_user_input()
    except Exception as e:
        print(e); exit()

    # 3. Scelta Gara
    print(f"\nGare {year}:")
    print("-" * 40)
    for idx, row in races.iterrows():
        print(f"{idx + 1:2}. {row['EventName']}")
    print("-" * 40)

    while True:
        try:
            s = int(input(f"\nNumero gara (1-{len(races)}): "))
            if 1 <= s <= len(races):
                return year, races.iloc[s - 1]['EventName']
        except ValueError: pass

def main():
    race_year, race_gp = get_user_input()

    print("\n==========================================")
    print(f"   STARTING SIMULATION: {race_gp.upper()} {race_year}   ")
    print("==========================================")

    # FASE 1: DATI
    try:
        data_engine = TyreDataModeler(race_year, race_gp)
        data_engine.load_and_clean_data()
        data_engine.analyze_degradation()
        
        # RECUPERIAMO ANCHE LA PIT LOSS DINAMICA
        real_tyre_models, total_laps, dynamic_pit_loss = data_engine.get_simulation_data()
        
    except Exception as e:
        print(f"\n[ERRORE] {e}"); return

    print(f"\nParametri Estratti:")
    print(f"Giri Totali: {total_laps}")
    print(f"Pit Loss Dinamica: {dynamic_pit_loss:.2f}s")
    for k, v in real_tyre_models.items():
        print(f"{k}: Base={v['base_pace']:.2f}s, Deg={v['degradation']:.3f}s/lap")

    # FASE 2: GREEDY (Passiamo la pit_loss dinamica)
    print("\n[2/3] Greedy Algorithm...")
    greedy = GreedySolver(real_tyre_models, total_laps, pit_loss=dynamic_pit_loss)
    gt, gs = greedy.solve()
    print(f"Greedy: {gt:.2f}s | {gs} -> {check_legality(gs)}")

    # FASE 3: GENETIC (Passiamo la pit_loss dinamica)
    print("\n[3/3] Genetic Algorithm...")
    ga = GeneticOptimizer(
        tyre_models=real_tyre_models, 
        total_laps=total_laps,
        pop_size=80, generations=60, mutation_rate=0.25,
        pit_loss=dynamic_pit_loss # <--- Qui passiamo il valore calcolato
    )
    best = ga.run()
    print(f"Genetic: {best.fitness:.2f}s | {best.genes} -> {check_legality(best.genes)}")
    
    print(f"\n>>> GUADAGNO: {gt - best.fitness:.2f}s <<<")

    # FASE 4: PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(ga.best_history, label='Genetic', color='blue')
    plt.axhline(y=gt, color='red', linestyle='--', label='Greedy')
    plt.title(f"Optimization: {race_gp} {race_year} (PitLoss: {dynamic_pit_loss:.1f}s)")
    plt.xlabel("Generations"); plt.ylabel("Time (s)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()