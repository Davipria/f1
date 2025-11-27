import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_results(history, greedy_time, greedy_stints, ga_stints, race_gp, year):
    """
    Generates a dashboard with two panels:
    1. Convergence Evolution (Genetic Algorithm improvement over generations).
    2. Strategy Comparison (Gantt chart showing tyre usage).
    """
    # Setup the figure with 2 subplots (Convergence + Strategy)
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # --- PANEL 1: CONVERGENCE EVOLUTION ---
    ax1 = fig.add_subplot(gs[0])
    
    # Dynamic Y-Limit: Focus on the relevant range
    best_time = history[-1]
    max_view = max(greedy_time, best_time) + 60 
    
    # Filter data for plotting to avoid scaling issues
    plot_history = [h if h < max_view + 100 else max_view + 20 for h in history]
    
    ax1.plot(plot_history, label='Genetic Evolution', color='#1f77b4', linewidth=2)
    ax1.axhline(y=greedy_time, color='#d62728', linestyle='--', label='Greedy Baseline', linewidth=2)
    
    ax1.set_title(f"Optimization Analysis: {race_gp} {year}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Race Time (s)", fontsize=12)
    ax1.set_xlabel("Generations", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add Text Annotation for the Gain
    gain = greedy_time - best_time
    color = 'green' if gain > 0 else 'red'
    ax1.annotate(f"Strategic Gain: {gain:.2f}s", 
                 xy=(len(history)-1, best_time), 
                 xytext=(len(history)-15, best_time - (gain/2 if gain > 0 else -10)),
                 arrowprops=dict(facecolor=color, shrink=0.05),
                 fontsize=12, color=color, fontweight='bold')

    # --- PANEL 2: STRATEGY VISUALIZATION (GANTT) ---
    ax2 = fig.add_subplot(gs[1])
    
    # Tyre Colors (Standard F1 Colors)
    tyre_colors = {
        'SOFT': '#ff3b3b',   # Red
        'MEDIUM': '#ffd700', # Yellow
        'HARD': '#f0f0f0',   # White
        'INTER': '#32cd32',  # Green
        'WET': '#1e90ff'     # Blue
    }
    
    def draw_strategy_bar(ax, y_pos, stints, label):
        current_lap = 0
        for compound, laps in stints:
            c = tyre_colors.get(compound, 'gray')
            # Draw the bar
            ax.broken_barh([(current_lap, laps)], (y_pos, 0.8), facecolors=c, edgecolor='black')
            # Add label in the center of the bar
            ax.text(current_lap + laps/2, y_pos + 0.4, f"{compound}\n{laps} Laps", 
                    ha='center', va='center', color='black', fontsize=9, fontweight='bold')
            current_lap += laps
        return current_lap

    # Plot both strategies
    max_laps_g = draw_strategy_bar(ax2, 10, greedy_stints, "Greedy")
    max_laps_ga = draw_strategy_bar(ax2, 24, ga_stints, "Genetic")
    
    ax2.set_yticks([14, 28])
    ax2.set_yticklabels(['Greedy (Local Opt)', 'Genetic (Global Opt)'], fontsize=12, fontweight='bold')
    ax2.set_xlabel("Lap Number", fontsize=12)
    ax2.set_title("Strategy Comparison: Stints & Compounds", fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max(max_laps_g, max_laps_ga) + 2)
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()