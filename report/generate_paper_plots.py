"""
Thesis-Ready Plot Generator for F1 Strategy Optimizer

Creates publication-quality visualizations for academic thesis.

Usage:
    python generate_thesis_plots.py results/batch_results_2024_*.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class ThesisPlotGenerator:
    def __init__(self, results_file, output_dir="plots"):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data = None
        self.df = None
        
    def load_results(self):
        """Load results from JSON file."""
        with open(self.results_file, 'r') as f:
            self.data = json.load(f)
        
        successful = [r for r in self.data['results'] if r['status'] == 'SUCCESS']
        
        self.df = pd.DataFrame([{
            'race': r['race'],
            'greedy_time': r['greedy_time'],
            'ga_mean': r['ga_mean'],
            'ga_std': r['ga_std'],
            'ga_best': r['ga_best'],
            'improvement_mean': r['improvement_mean'],
            'improvement_std': r['improvement_std']
        } for r in successful])
        
        # Shorten race names for better visualization
        self.df['race_short'] = self.df['race'].str.replace(' Grand Prix', '').str.replace('Grand Prix', '')
        
        print(f"Loaded {len(self.df)} circuits from {self.data['year']}")
    
    def plot_1_comparison_bar(self):
        """
        Figure 1: Bar chart comparing Greedy vs GA for each circuit.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(self.df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, self.df['greedy_time'], width, 
                      label='Greedy', color='#d62728', alpha=0.8)
        bars2 = ax.bar(x + width/2, self.df['ga_mean'], width, 
                      label='Genetic Algorithm', color='#2ca02c', alpha=0.8,
                      yerr=self.df['ga_std'], capsize=3)
        
        ax.set_xlabel('Circuit', fontsize=12, fontweight='bold')
        ax.set_ylabel('Race Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Algorithm Comparison: {self.data["year"]} F1 Season', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.df['race_short'], rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig1_comparison_bar.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 1 saved: {output_path}")
    
    def plot_2_improvement_distribution(self):
        """
        Figure 2: Distribution of improvements with histogram and box plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        improvements = self.df['improvement_mean'].values
        
        # Histogram
        ax1.hist(improvements, bins=12, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(improvements), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(improvements):.2f}s')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Improvement (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of GA Improvements', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        bp = ax2.boxplot([improvements], vert=True, patch_artist=True, 
                         widths=0.5, showmeans=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][0].set_alpha(0.7)
        bp['means'][0].set_marker('D')
        bp['means'][0].set_markerfacecolor('red')
        bp['means'][0].set_markersize(8)
        
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Improvement (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Improvement Statistics', fontsize=13, fontweight='bold')
        ax2.set_xticklabels(['GA vs Greedy'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add text annotations
        q1, median, q3 = np.percentile(improvements, [25, 50, 75])
        ax2.text(1.25, q1, f'Q1: {q1:.2f}s', fontsize=9)
        ax2.text(1.25, median, f'Median: {median:.2f}s', fontsize=9, fontweight='bold')
        ax2.text(1.25, q3, f'Q3: {q3:.2f}s', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig2_improvement_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 2 saved: {output_path}")
    
    def plot_3_scatter_correlation(self):
        """
        Figure 3: Scatter plot showing Greedy vs GA times.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        greedy = self.df['greedy_time'].values
        ga = self.df['ga_mean'].values
        
        # Scatter plot
        ax.scatter(greedy, ga, s=100, alpha=0.6, edgecolors='black', linewidths=1)
        
        # Add diagonal line (y=x, representing no improvement)
        min_val = min(greedy.min(), ga.min())
        max_val = max(greedy.max(), ga.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               linewidth=2, alpha=0.7, label='No Improvement (y=x)')
        
        # Add circuit labels for outliers
        for idx, row in self.df.iterrows():
            if abs(row['improvement_mean']) > self.df['improvement_mean'].std():
                ax.annotate(row['race_short'], 
                          (row['greedy_time'], row['ga_mean']),
                          fontsize=8, alpha=0.7,
                          xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Greedy Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('GA Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Greedy vs Genetic Algorithm Performance', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add annotation
        correlation = np.corrcoef(greedy, ga)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig3_scatter_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 3 saved: {output_path}")
    
    def plot_4_ranking_improvement(self):
        """
        Figure 4: Circuits ranked by improvement.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by improvement
        df_sorted = self.df.sort_values('improvement_mean', ascending=True)
        
        # Color code: green for positive, red for negative
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in df_sorted['improvement_mean']]
        
        y_pos = np.arange(len(df_sorted))
        ax.barh(y_pos, df_sorted['improvement_mean'], 
               xerr=df_sorted['improvement_std'],
               color=colors, alpha=0.7, capsize=3, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['race_short'], fontsize=10)
        ax.set_xlabel('Improvement (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Circuits Ranked by GA Improvement', 
                    fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add mean line
        mean_improvement = df_sorted['improvement_mean'].mean()
        ax.axvline(mean_improvement, color='blue', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Mean: {mean_improvement:.2f}s')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig4_ranking_improvement.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 4 saved: {output_path}")
    
    def plot_5_algorithm_comparison_table(self):
        """
        Figure 5: Summary table as an image.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        table_data.append(['Circuit', 'Greedy (s)', 'GA Mean (s)', 'GA Std (s)', 'Improvement (s)', 'Win'])
        
        for _, row in self.df.iterrows():
            win_symbol = '✓' if row['improvement_mean'] > 0 else '✗'
            table_data.append([
                row['race_short'],
                f"{row['greedy_time']:.2f}",
                f"{row['ga_mean']:.2f}",
                f"±{row['ga_std']:.2f}",
                f"{row['improvement_mean']:+.2f}",
                win_symbol
            ])
        
        # Summary row
        greedy_mean = self.df['greedy_time'].mean()
        ga_mean_mean = self.df['ga_mean'].mean()
        improvement_mean = self.df['improvement_mean'].mean()
        wins = (self.df['improvement_mean'] > 0).sum()
        
        table_data.append(['---', '---', '---', '---', '---', '---'])
        table_data.append([
            'MEAN',
            f"{greedy_mean:.2f}",
            f"{ga_mean_mean:.2f}",
            '-',
            f"{improvement_mean:+.2f}",
            f"{wins}/{len(self.df)}"
        ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.12, 0.12, 0.12, 0.15, 0.08])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style summary row
        for i in range(6):
            table[(len(table_data)-1, i)].set_facecolor('#FFC107')
            table[(len(table_data)-1, i)].set_text_props(weight='bold')
        
        plt.title(f'Complete Results: {self.data["year"]} F1 Season', 
                 fontsize=14, fontweight='bold', pad=20)
        
        output_path = self.output_dir / 'fig5_results_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Figure 5 saved: {output_path}")
    
    def generate_latex_table(self):
        """
        Generate LaTeX code for thesis table.
        """
        latex_file = self.output_dir / 'results_table.tex'
        
        with open(latex_file, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparison of Greedy and Genetic Algorithm Performance}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\begin{tabular}{lrrrr}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Circuit} & \\textbf{Greedy (s)} & \\textbf{GA Mean (s)} & \\textbf{GA Std (s)} & \\textbf{Improvement (s)} \\\\\n")
            f.write("\\hline\n")
            
            for _, row in self.df.iterrows():
                f.write(f"{row['race_short']} & {row['greedy_time']:.2f} & {row['ga_mean']:.2f} & {row['ga_std']:.2f} & {row['improvement_mean']:+.2f} \\\\\n")
            
            f.write("\\hline\n")
            
            # Summary
            greedy_mean = self.df['greedy_time'].mean()
            ga_mean_mean = self.df['ga_mean'].mean()
            improvement_mean = self.df['improvement_mean'].mean()
            improvement_std = self.df['improvement_mean'].std()
            
            f.write(f"\\textbf{{Mean}} & {greedy_mean:.2f} & {ga_mean_mean:.2f} & - & {improvement_mean:.2f} $\\pm$ {improvement_std:.2f} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"✓ LaTeX table saved: {latex_file}")
    
    def generate_all_plots(self):
        """
        Generate all thesis plots.
        """
        print(f"\n{'='*70}")
        print(f"GENERATING THESIS PLOTS")
        print(f"{'='*70}\n")
        
        self.load_results()
        
        self.plot_1_comparison_bar()
        self.plot_2_improvement_distribution()
        self.plot_3_scatter_correlation()
        self.plot_4_ranking_improvement()
        self.plot_5_algorithm_comparison_table()
        self.generate_latex_table()
        
        print(f"\n{'='*70}")
        print(f"ALL PLOTS GENERATED SUCCESSFULLY")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Generate thesis plots from F1 optimizer results')
    parser.add_argument('results_file', type=str, help='Path to JSON results file')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    generator = ThesisPlotGenerator(args.results_file, output_dir=args.output)
    generator.generate_all_plots()

if __name__ == "__main__":
    main()