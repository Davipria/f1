import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse

"""
Statistical Analysis for F1 Strategy Optimizer Results

Performs comprehensive statistical tests to validate the superiority
of the Genetic Algorithm over the Greedy baseline.

"""

class StatisticalAnalyzer:
    def __init__(self, results_file):
        self.results_file = Path(results_file)
        self.data = None
        self.df = None
        
    def load_results(self):
        """Load results from JSON file."""
        print(f"\n{'='*70}")
        print(f"LOADING RESULTS")
        print(f"{'='*70}")
        
        with open(self.results_file, 'r') as f:
            self.data = json.load(f)
        
        # Convert to DataFrame for analysis
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
        
        print(f"Loaded {len(self.df)} successful circuits from {self.data['year']}")
        print(f"File: {self.results_file}")
        
    def descriptive_statistics(self):
        """Calculate descriptive statistics."""
        print(f"\n{'='*70}")
        print(f"DESCRIPTIVE STATISTICS")
        print(f"{'='*70}")
        
        improvements = self.df['improvement_mean'].values
        
        stats_dict = {
            'Mean': np.mean(improvements),
            'Median': np.median(improvements),
            'Std Dev': np.std(improvements, ddof=1),
            'Min': np.min(improvements),
            'Max': np.max(improvements),
            'Q1 (25%)': np.percentile(improvements, 25),
            'Q3 (75%)': np.percentile(improvements, 75),
            'IQR': np.percentile(improvements, 75) - np.percentile(improvements, 25)
        }
        
        print(f"\nImprovement Statistics (GA vs Greedy):")
        print(f"{'-'*40}")
        for key, value in stats_dict.items():
            print(f"{key:<15} {value:>10.3f}s")
        
        return stats_dict
    
    def one_sample_ttest(self):
        """
        One-sample t-test: H0: mean improvement = 0
        
        Tests if the GA improvement is significantly different from zero.
        """
        print(f"\n{'='*70}")
        print(f"ONE-SAMPLE T-TEST")
        print(f"{'='*70}")
        
        improvements = self.df['improvement_mean'].values
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(improvements, 0)
        
        # Calculate confidence interval
        n = len(improvements)
        mean = np.mean(improvements)
        std_err = np.std(improvements, ddof=1) / np.sqrt(n)
        confidence = 0.95
        df_freedom = n - 1
        t_critical = stats.t.ppf((1 + confidence) / 2, df_freedom)
        ci_lower = mean - t_critical * std_err
        ci_upper = mean + t_critical * std_err
        
        print(f"\nHypothesis Test:")
        print(f"  H0: Mean improvement = 0 (no difference)")
        print(f"  H1: Mean improvement ≠ 0 (GA is different)")
        print(f"\nResults:")
        print(f"  Sample size (n):     {n}")
        print(f"  Mean improvement:    {mean:.3f}s")
        print(f"  Standard error:      {std_err:.3f}s")
        print(f"  t-statistic:         {t_stat:.3f}")
        print(f"  p-value:             {p_value:.6f}")
        print(f"  95% CI:              [{ci_lower:.3f}s, {ci_upper:.3f}s]")
        
        # Interpretation
        alpha = 0.05
        print(f"\nInterpretation (α = {alpha}):")
        if p_value < 0.001:
            print(f"  ✓ HIGHLY SIGNIFICANT (p < 0.001)")
            print(f"    The GA provides a statistically significant improvement.")
        elif p_value < alpha:
            print(f"  ✓ SIGNIFICANT (p < {alpha})")
            print(f"    The GA provides a statistically significant improvement.")
        else:
            print(f"  ✗ NOT SIGNIFICANT (p ≥ {alpha})")
            print(f"    Cannot conclude that GA is better than Greedy.")
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < alpha
        }
    
    def effect_size(self):
        """
        Calculate Cohen's d effect size.
        
        Cohen's d measures the standardized difference between two means.
        Interpretation:
            |d| < 0.2  : negligible
            |d| < 0.5  : small
            |d| < 0.8  : medium
            |d| ≥ 0.8  : large
        """
        print(f"\n{'='*70}")
        print(f"EFFECT SIZE (COHEN'S d)")
        print(f"{'='*70}")
        
        improvements = self.df['improvement_mean'].values
        
        # Cohen's d for one-sample: d = mean / std
        mean = np.mean(improvements)
        std = np.std(improvements, ddof=1)
        cohens_d = mean / std
        
        print(f"\nCalculation:")
        print(f"  Mean improvement: {mean:.3f}s")
        print(f"  Std deviation:    {std:.3f}s")
        print(f"  Cohen's d:        {cohens_d:.3f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if abs(cohens_d) < 0.2:
            magnitude = "NEGLIGIBLE"
            interpretation = "The effect is very small and may not be practically significant."
        elif abs(cohens_d) < 0.5:
            magnitude = "SMALL"
            interpretation = "The effect is noticeable but modest."
        elif abs(cohens_d) < 0.8:
            magnitude = "MEDIUM"
            interpretation = "The effect is substantial and practically important."
        else:
            magnitude = "LARGE"
            interpretation = "The effect is very strong and highly significant."
        
        print(f"  Magnitude: {magnitude} (|d| = {abs(cohens_d):.3f})")
        print(f"  {interpretation}")
        
        return cohens_d
    
    def paired_comparison(self):
        """
        Paired t-test: Greedy vs GA on the same circuits.
        """
        print(f"\n{'='*70}")
        print(f"PAIRED T-TEST (Greedy vs GA)")
        print(f"{'='*70}")
        
        greedy_times = self.df['greedy_time'].values
        ga_times = self.df['ga_mean'].values
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(greedy_times, ga_times)
        
        print(f"\nHypothesis Test:")
        print(f"  H0: Mean(Greedy) = Mean(GA)")
        print(f"  H1: Mean(Greedy) ≠ Mean(GA)")
        print(f"\nResults:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value:     {p_value:.6f}")
        
        if p_value < 0.001:
            print(f"  ✓ HIGHLY SIGNIFICANT (p < 0.001)")
        elif p_value < 0.05:
            print(f"  ✓ SIGNIFICANT (p < 0.05)")
        else:
            print(f"  ✗ NOT SIGNIFICANT")
        
        return {'t_statistic': t_stat, 'p_value': p_value}
    
    def win_rate_analysis(self):
        """
        Analyze how often GA beats Greedy.
        """
        print(f"\n{'='*70}")
        print(f"WIN RATE ANALYSIS")
        print(f"{'='*70}")
        
        improvements = self.df['improvement_mean'].values
        
        wins = np.sum(improvements > 0)
        losses = np.sum(improvements < 0)
        ties = np.sum(improvements == 0)
        total = len(improvements)
        
        win_rate = wins / total * 100
        
        print(f"\nResults:")
        print(f"  GA Wins:    {wins}/{total} ({win_rate:.1f}%)")
        print(f"  GA Losses:  {losses}/{total} ({losses/total*100:.1f}%)")
        print(f"  Ties:       {ties}/{total}")
        
        # Binomial test: Is win rate significantly > 50%?
        p_value = stats.binomtest(wins, total, 0.5, alternative='greater').pvalue        
        print(f"\nBinomial Test (H0: win rate = 50%):")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  ✓ GA wins significantly more than expected by chance")
        else:
            print(f"  ✗ Win rate not significantly different from 50%")
        
        return {
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_rate': win_rate,
            'p_value': p_value
        }
    
    def generate_report(self, output_file="statistical_report.txt"):
        """
        Generate a comprehensive text report.
        """
        print(f"\n{'='*70}")
        print(f"GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*70}")
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("STATISTICAL VALIDATION REPORT\n")
            f.write(f"F1 Strategy Optimizer: Genetic Algorithm vs Greedy\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: {self.data['year']} F1 Season\n")
            f.write(f"Circuits tested: {len(self.df)}\n")
            f.write(f"Analysis date: {self.data['timestamp']}\n\n")
            
            # Descriptive stats
            desc_stats = self.descriptive_statistics()
            f.write("\n1. DESCRIPTIVE STATISTICS\n")
            f.write("-"*70 + "\n")
            for key, value in desc_stats.items():
                f.write(f"{key:<15} {value:>10.3f}s\n")
            
            # T-test
            ttest_results = self.one_sample_ttest()
            f.write("\n2. ONE-SAMPLE T-TEST\n")
            f.write("-"*70 + "\n")
            f.write(f"t-statistic:    {ttest_results['t_statistic']:.3f}\n")
            f.write(f"p-value:        {ttest_results['p_value']:.6f}\n")
            f.write(f"95% CI:         [{ttest_results['ci_lower']:.3f}s, {ttest_results['ci_upper']:.3f}s]\n")
            f.write(f"Significant:    {'YES' if ttest_results['significant'] else 'NO'}\n")
            
            # Effect size
            cohens_d = self.effect_size()
            f.write("\n3. EFFECT SIZE\n")
            f.write("-"*70 + "\n")
            f.write(f"Cohen's d:      {cohens_d:.3f}\n")
            
            # Win rate
            win_results = self.win_rate_analysis()
            f.write("\n4. WIN RATE\n")
            f.write("-"*70 + "\n")
            f.write(f"GA Wins:        {win_results['wins']}/{len(self.df)} ({win_results['win_rate']:.1f}%)\n")
            f.write(f"Binomial test:  p = {win_results['p_value']:.6f}\n")
            
            # Conclusion
            f.write("\n5. CONCLUSION\n")
            f.write("-"*70 + "\n")
            
            if ttest_results['significant'] and cohens_d > 0.5:
                f.write("The Genetic Algorithm demonstrates a STATISTICALLY SIGNIFICANT\n")
                f.write("and PRACTICALLY MEANINGFUL improvement over the Greedy baseline.\n")
                f.write(f"Mean improvement: {desc_stats['Mean']:.2f}s ± {desc_stats['Std Dev']:.2f}s\n")
                f.write(f"Effect size: {cohens_d:.2f} (Medium to Large)\n")
            elif ttest_results['significant']:
                f.write("The Genetic Algorithm shows a STATISTICALLY SIGNIFICANT improvement,\n")
                f.write("but the effect size is relatively small.\n")
            else:
                f.write("No statistically significant difference was found between GA and Greedy.\n")
        
        print(f"✓ Report saved to: {output_file}")
    
    def run_full_analysis(self):
        """
        Run all statistical tests and generate report.
        """
        self.load_results()
        self.descriptive_statistics()
        self.one_sample_ttest()
        self.effect_size()
        self.paired_comparison()
        self.win_rate_analysis()
        self.generate_report()

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis of F1 optimizer results')
    parser.add_argument('results_file', type=str, help='Path to JSON results file')
    parser.add_argument('--output', type=str, default='statistical_report.txt', help='Output report filename')
    
    args = parser.parse_args()
    
    analyzer = StatisticalAnalyzer(args.results_file)
    analyzer.run_full_analysis()
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()