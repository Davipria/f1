import os
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import config

if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

class TyreDataModeler:
    def __init__(self, year, gp, session_type='R', visualize_fits=False):
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.visualize_fits = visualize_fits
        self.laps = None
        self.models = {} 
        self.model_quality = {} 
        self.pit_loss = config.DEFAULT_PIT_LOSS
        self.session_avg_pace = 90.0
        
    def load_and_clean_data(self):
        print(f"Loading {self.gp} {self.year}...")
        session = fastf1.get_session(self.year, self.gp, self.session_type)
        session.load()
        
        self.pit_loss = self._calculate_pit_loss(session)
        print(f"--> Pit Loss Calculated (Median): {self.pit_loss:.2f}s")
        
        laps = session.laps.pick_quicklaps()
        laps = laps.loc[(laps['PitOutTime'].isnull()) & (laps['PitInTime'].isnull())]
        
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        self.laps = laps[['Driver', 'LapNumber', 'LapTimeSec', 'Compound', 'TyreLife']]
        
        if not self.laps.empty:
            self.session_avg_pace = self.laps['LapTimeSec'].median()
            print(f"--> Session Avg Pace detected: {self.session_avg_pace:.2f}s")
        
        print(f"Data Loaded. {len(self.laps)} clean laps found.")

    def _calculate_pit_loss(self, session):
        laps = session.laps.pick_track_status('1')
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        
        clean_laps = laps[laps['PitOutTime'].isnull() & laps['PitInTime'].isnull()]
        if clean_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_race = clean_laps['LapTimeSec'].median()
        
        in_laps = laps[~laps['PitInTime'].isnull()]
        if in_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_in = in_laps['LapTimeSec'].median()
        
        out_laps = laps[~laps['PitOutTime'].isnull()]
        out_laps = out_laps[out_laps['LapNumber'] > 1] 
        if out_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_out = out_laps['LapTimeSec'].median()
        
        loss = (avg_in + avg_out) - (2 * avg_race)
        return max(15.0, loss)

    def analyze_degradation(self):
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        
        if self.visualize_fits:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Tyre Degradation Analysis: {self.gp} {self.year}", fontsize=14, fontweight='bold')
        
        # Extraction and fitting of the degradation model
        for idx, compound in enumerate(compounds):
            comp_laps = self.laps[self.laps['Compound'] == compound]
            
            if len(comp_laps) < 3:
                print(f"WARNING: No data for {compound} (0-2 laps). Using DYNAMIC defaults.")
                self._use_defaults(compound)
                continue

            Q1 = comp_laps['LapTimeSec'].quantile(0.25)
            Q3 = comp_laps['LapTimeSec'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            comp_laps_clean = comp_laps[
                (comp_laps['LapTimeSec'] >= lower_bound) & 
                (comp_laps['LapTimeSec'] <= upper_bound)
            ]
            
            if len(comp_laps_clean) < 3:
                comp_laps_clean = comp_laps

            X = comp_laps_clean[['TyreLife']].values 
            y = comp_laps_clean['LapTimeSec'].values 
            
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            reg = LinearRegression().fit(X_poly, y)
            
            c0 = reg.intercept_
            c1, c2 = reg.coef_[1], reg.coef_[2]
            
            if c2 < 0:
                print(f"  > Fixing negative quadratic for {compound} ({c2:.6f} -> 0.0)")
                c2 = 0.0
            
            y_pred = reg.predict(X_poly)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            self.model_quality[compound] = {'R2': r2, 'MAE': mae}
            
            self.models[compound] = {
                'base_pace': c0,
                'linear_degradation': c1,
                'quadratic_degradation': c2
            }
            print(f"{compound}: Base={c0:.2f}s, Lin={c1:.4f}, Quad={c2:.6f}, R2={r2:.2f} (RAW)")
            
            if self.visualize_fits:
                ax = axes[idx]
                ax.scatter(X, y, alpha=0.5, s=20, label='Actual Laps', color='blue')
                X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_plot = c0 + c1 * X_range + c2 * (X_range ** 2)
                ax.plot(X_range, y_plot, color='red', linewidth=2, label=f'Fit')
                ax.set_title(f"{compound}", fontweight='bold')
                ax.set_xlabel("Tyre Life")
                ax.set_ylabel("Lap Time")
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Sanity check: correction of absurd values
        print("\n[SANITY CHECK] Verifying physics consistency...")
        
        # Create a copy of the keys to iterate, since we may modify self.models
        compounds_found = [c for c in compounds if c in self.models]
        
        if len(compounds_found) == 3:
            for target in compounds_found:
                # Take the other two compounds
                others = [c for c in compounds_found if c != target]
                
                pace_target = self.models[target]['base_pace']
                
                # Calculate the average of the other two
                pace_others = [self.models[c]['base_pace'] for c in others]
                avg_pace_others = sum(pace_others) / len(pace_others)
                
                diff = abs(pace_target - avg_pace_others)
                
                if diff > 5.0:
                    print(f"⚠️ ANOMALY DETECTED IN {target}!")
                    print(f"   Base Pace: {pace_target:.2f}s | Others Avg: {avg_pace_others:.2f}s | Diff: {diff:.2f}s")
                    print(f"   >>> OVERWRITING WITH AVERAGE OF {others[0]} AND {others[1]}")
                    
                    # Calculate the average of the degradations to avoid "monsters" (e.g. 17s/lap)
                    lin_others = [self.models[c]['linear_degradation'] for c in others]
                    quad_others = [self.models[c]['quadratic_degradation'] for c in others]
                    
                    avg_lin = sum(lin_others) / len(lin_others)
                    avg_quad = sum(quad_others) / len(quad_others)
                    
                    # Overwrite the model
                    self.models[target] = {
                        'base_pace': avg_pace_others,
                        'linear_degradation': avg_lin,
                        'quadratic_degradation': avg_quad
                    }
                    print(f"   New Model: Base={avg_pace_others:.2f}s, Lin={avg_lin:.4f}, Quad={avg_quad:.6f}")

        if self.visualize_fits:
            plt.tight_layout()
            plt.show()

    def _use_defaults(self, compound):
        base_pace = self.session_avg_pace
        if compound == 'HARD': base_pace += 0.5
        elif compound == 'SOFT': base_pace -= 0.5
        
        quad_coef = config.NON_LINEAR_WEAR.get(compound, 0.002)
        linear_map = {'SOFT': 0.05, 'MEDIUM': 0.03, 'HARD': 0.02}
        linear_coef = linear_map.get(compound, 0.03)
        
        self.models[compound] = {
            'base_pace': base_pace,
            'linear_degradation': linear_coef,
            'quadratic_degradation': quad_coef
        }
        self.model_quality[compound] = {'R2': 0.0, 'MAE': float('inf')}

    def get_simulation_data(self):
        total_laps = int(self.laps['LapNumber'].max())
        for comp in ['SOFT', 'MEDIUM', 'HARD']:
            if comp not in self.models:
                self._use_defaults(comp)
        return self.models, total_laps, self.pit_loss
    
    def print_model_summary(self):
        print("\n" + "="*60)
        print("MODEL QUALITY SUMMARY")
        print("="*60)
        for compound, quality in self.model_quality.items():
            r2 = quality['R2']
            mae = quality['MAE']
            print(f"{compound:8} | R²={r2:.3f} | MAE={mae:.2f}s")
        print("="*60 + "\n")