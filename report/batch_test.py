import random
import numpy as np
import pandas as pd
import fastf1
import argparse
import json
from datetime import datetime
from pathlib import Path
import config
from data_model import TyreDataModeler
from optimizers import GeneticOptimizer, GreedySolver

"""
Batch Testing Script for F1 Strategy Optimizer

Automatically tests the Genetic Algorithm vs Greedy on multiple Grand Prix
and saves results for statistical analysis.

"""


class BatchTester:
    def __init__(self, year, output_dir="results"):
        self.year = year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def get_available_races(self):
        """
        Fetch all races from the specified season.
        Returns a list of race names that have data available.
        """
        print(f"\n{'='*70}")
        print(f"FETCHING {self.year} CALENDAR")
        print(f"{'='*70}")
        
        try:
            schedule = fastf1.get_event_schedule(self.year, include_testing=False)
            races = schedule[schedule['RoundNumber'] > 0][['EventName', 'Location', 'RoundNumber']].reset_index(drop=True)
            
            print(f"Found {len(races)} races in {self.year} season")
            return races
            
        except Exception as e:
            print(f"ERROR: Could not fetch calendar: {e}")
            return pd.DataFrame()
    
    def test_single_race(self, race_name, num_runs=3, verbose=True):
        """
        Test a single Grand Prix with multiple runs for statistical robustness.
        
        Args:
            race_name: Name of the Grand Prix
            num_runs: Number of independent runs (default: 3)
            verbose: Print detailed progress
            
        Returns:
            dict: Results for this race
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"TESTING: {race_name.upper()}")
            print(f"{'='*70}")
        
        race_results = {
            'race': race_name,
            'year': self.year,
            'status': 'PENDING',
            'greedy_time': None,
            'greedy_strategy': None,
            'ga_runs': [],
            'ga_mean': None,
            'ga_std': None,
            'ga_best': None,
            'improvement_mean': None,
            'improvement_std': None,
            'model_quality': {}
        }
        
        try:
            # Load data
            if verbose:
                print(f"\n[1/4] Loading telemetry data...")
            
            data_engine = TyreDataModeler(self.year, race_name, visualize_fits=False)
            data_engine.load_and_clean_data()
            data_engine.analyze_degradation()
            
            tyre_models, total_laps, pit_loss = data_engine.get_simulation_data()
            
            # Store model quality
            race_results['model_quality'] = data_engine.model_quality
            race_results['total_laps'] = total_laps
            race_results['pit_loss'] = pit_loss
            
            if verbose:
                print(f"Circuit Info: {total_laps} laps, Pit Loss: {pit_loss:.2f}s")
            
            # Run Greedy (once, deterministic)
            if verbose:
                print(f"\n[2/4] Running Greedy Algorithm...")
            
            greedy = GreedySolver(tyre_models, total_laps, pit_loss=pit_loss)
            greedy_time, greedy_stints = greedy.solve()
            
            race_results['greedy_time'] = float(greedy_time)
            race_results['greedy_strategy'] = greedy_stints
            
            if verbose:
                print(f"Greedy Time: {greedy_time:.2f}s")
                print(f"Greedy Strategy: {greedy_stints}")
            
            # Run GA multiple times
            if verbose:
                print(f"\n[3/4] Running Genetic Algorithm ({num_runs} runs)...")
            
            ga_times = []
            ga_strategies = []
            
            for run in range(num_runs):
                # Set different seed for each run
                run_seed = config.RANDOM_SEED + run
                random.seed(run_seed)
                np.random.seed(run_seed)
                
                ga = GeneticOptimizer(
                    tyre_models=tyre_models,
                    total_laps=total_laps,
                    pop_size=config.GA_SETTINGS['POP_SIZE'],
                    generations=config.GA_SETTINGS['GENERATIONS'],
                    mutation_rate=config.GA_SETTINGS['MUTATION_RATE'],
                    pit_loss=pit_loss
                )
                
                best_solution = ga.run()
                
                ga_times.append(float(best_solution.fitness))
                ga_strategies.append(best_solution.genes)
                
                race_results['ga_runs'].append({
                    'run_id': run + 1,
                    'time': float(best_solution.fitness),
                    'strategy': best_solution.genes,
                    'seed': run_seed
                })
                
                if verbose:
                    improvement = greedy_time - best_solution.fitness
                    print(f"  Run {run+1}: {best_solution.fitness:.2f}s (Δ = {improvement:+.2f}s)")
            
            # Calculate statistics
            if verbose:
                print(f"\n[4/4] Computing statistics...")
            
            ga_mean = np.mean(ga_times)
            ga_std = np.std(ga_times)
            ga_best = min(ga_times)
            
            race_results['ga_mean'] = float(ga_mean)
            race_results['ga_std'] = float(ga_std)
            race_results['ga_best'] = float(ga_best)
            
            improvements = [greedy_time - t for t in ga_times]
            race_results['improvement_mean'] = float(np.mean(improvements))
            race_results['improvement_std'] = float(np.std(improvements))
            race_results['improvement_best'] = float(max(improvements))
            
            race_results['status'] = 'SUCCESS'
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"RESULTS FOR {race_name.upper()}")
                print(f"{'='*70}")
                print(f"Greedy:     {greedy_time:.2f}s")
                print(f"GA Mean:    {ga_mean:.2f}s ± {ga_std:.2f}s")
                print(f"GA Best:    {ga_best:.2f}s")
                print(f"Improvement: {race_results['improvement_mean']:.2f}s ± {race_results['improvement_std']:.2f}s")
                print(f"Status:     ✓ SUCCESS")
                
        except Exception as e:
            race_results['status'] = 'FAILED'
            race_results['error'] = str(e)
            
            if verbose:
                print(f"\n✗ FAILED: {e}")
        
        return race_results
    
    def run_batch_test(self, circuit_filter=None, num_runs=3, max_circuits=None):
        """
        Run tests on multiple circuits.
        
        Args:
            circuit_filter: List of circuit names to test (None = all)
            num_runs: Number of GA runs per circuit
            max_circuits: Maximum number of circuits to test (None = all)
        """
        print(f"\n{'#'*70}")
        print(f"# BATCH TESTING: {self.year} F1 SEASON")
        print(f"# Runs per circuit: {num_runs}")
        print(f"{'#'*70}")
        
        races = self.get_available_races()
        
        if races.empty:
            print("ERROR: No races found")
            return
        
        # Apply filters
        if circuit_filter:
            races = races[races['EventName'].isin(circuit_filter)]
            print(f"\nFiltered to {len(races)} circuits: {circuit_filter}")
        
        if max_circuits:
            races = races.head(max_circuits)
            print(f"\nLimited to first {max_circuits} circuits")
        
        print(f"\nTesting {len(races)} circuits...")
        
        # Test each race
        for idx, row in races.iterrows():
            race_name = row['EventName']
            
            result = self.test_single_race(race_name, num_runs=num_runs, verbose=True)
            self.results.append(result)
            
            # Save after each race (in case of crash)
            self.save_results(partial=True)
        
        # Final save
        self.save_results(partial=False)
        
        # Print summary
        self.print_summary()
    
    def save_results(self, partial=False):
        """
        Save results to JSON and CSV files.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON (full data)
        json_filename = f"batch_results_{self.year}_{timestamp}.json" if not partial else f"batch_results_{self.year}_partial.json"
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump({
                'year': self.year,
                'timestamp': timestamp,
                'num_circuits': len(self.results),
                'results': self.results
            }, f, indent=2)
        
        if not partial:
            print(f"\n✓ Full results saved to: {json_path}")
        
        # Save CSV (for easy analysis)
        csv_data = []
        for r in self.results:
            if r['status'] == 'SUCCESS':
                csv_data.append({
                    'race': r['race'],
                    'year': r['year'],
                    'total_laps': r['total_laps'],
                    'pit_loss': r['pit_loss'],
                    'greedy_time': r['greedy_time'],
                    'ga_mean': r['ga_mean'],
                    'ga_std': r['ga_std'],
                    'ga_best': r['ga_best'],
                    'improvement_mean': r['improvement_mean'],
                    'improvement_std': r['improvement_std'],
                    'improvement_best': r['improvement_best'],
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_filename = f"batch_results_{self.year}_{timestamp}.csv" if not partial else f"batch_results_{self.year}_partial.csv"
            csv_path = self.output_dir / csv_filename
            df.to_csv(csv_path, index=False)
            
            if not partial:
                print(f"✓ CSV summary saved to: {csv_path}")
    
    def print_summary(self):
        """
        Print a summary table of all results.
        """
        print(f"\n{'='*70}")
        print(f"BATCH TEST SUMMARY: {self.year} SEASON")
        print(f"{'='*70}")
        
        successful = [r for r in self.results if r['status'] == 'SUCCESS']
        failed = [r for r in self.results if r['status'] == 'FAILED']
        
        print(f"\nCircuits tested: {len(self.results)}")
        print(f"  ✓ Successful: {len(successful)}")
        print(f"  ✗ Failed: {len(failed)}")
        
        if failed:
            print(f"\nFailed circuits:")
            for r in failed:
                print(f"  - {r['race']}: {r.get('error', 'Unknown error')}")
        
        if successful:
            print(f"\n{'='*70}")
            print(f"DETAILED RESULTS")
            print(f"{'='*70}")
            print(f"{'Circuit':<30} {'Greedy':>10} {'GA Mean':>12} {'Improvement':>12}")
            print(f"{'-'*70}")
            
            for r in successful:
                improvement = r['improvement_mean']
                symbol = "+" if improvement > 0 else ""
                print(f"{r['race']:<30} {r['greedy_time']:>10.2f}s {r['ga_mean']:>10.2f}s  {symbol}{improvement:>10.2f}s")
            
            # Aggregate statistics
            improvements = [r['improvement_mean'] for r in successful]
            
            print(f"{'-'*70}")
            print(f"\nAGGREGATE STATISTICS:")
            print(f"  Mean Improvement:   {np.mean(improvements):.2f}s ± {np.std(improvements):.2f}s")
            print(f"  Median Improvement: {np.median(improvements):.2f}s")
            print(f"  Best Improvement:   {np.max(improvements):.2f}s ({successful[np.argmax(improvements)]['race']})")
            print(f"  Worst Improvement:  {np.min(improvements):.2f}s ({successful[np.argmin(improvements)]['race']})")
            
            # Win rate
            wins = sum(1 for imp in improvements if imp > 0)
            win_rate = wins / len(improvements) * 100
            print(f"\n  GA Win Rate: {wins}/{len(improvements)} ({win_rate:.1f}%)")
            
            print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Batch test F1 strategy optimizer on multiple circuits')
    parser.add_argument('--year', type=int, default=2024, help='F1 season year (default: 2024)')
    parser.add_argument('--runs', type=int, default=3, help='Number of GA runs per circuit (default: 3)')
    parser.add_argument('--circuits', type=str, default=None, help='Comma-separated list of circuits to test (default: all)')
    parser.add_argument('--max', type=int, default=None, help='Maximum number of circuits to test (default: all)')
    parser.add_argument('--output', type=str, default='results', help='Output directory (default: results/)')
    
    args = parser.parse_args()
    
    # Parse circuit filter
    circuit_filter = None
    if args.circuits:
        circuit_filter = [c.strip() for c in args.circuits.split(',')]
    
    # Create tester
    tester = BatchTester(args.year, output_dir=args.output)
    
    # Run batch test
    tester.run_batch_test(
        circuit_filter=circuit_filter,
        num_runs=args.runs,
        max_circuits=args.max
    )

if __name__ == "__main__":
    main()