import os
import fastf1
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
        print(f"Pit Loss: {self.pit_loss:.2f}s")
        
        laps = session.laps.pick_quicklaps()
        laps = laps.loc[(laps['PitOutTime'].isnull()) & (laps['PitInTime'].isnull())]
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        self.laps = laps[['Driver', 'LapNumber', 'LapTimeSec', 'Compound', 'TyreLife']]
        
        if not self.laps.empty:
            self.session_avg_pace = self.laps['LapTimeSec'].median()
        print(f"Loaded {len(self.laps)} clean laps.")

    def _calculate_pit_loss(self, session):
        laps = session.laps.pick_track_status('1')
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        
        clean = laps[laps['PitOutTime'].isnull() & laps['PitInTime'].isnull()]
        if clean.empty: return config.DEFAULT_PIT_LOSS
        avg_race = clean['LapTimeSec'].median()
        
        in_laps = laps[~laps['PitInTime'].isnull()]
        if in_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_in = in_laps['LapTimeSec'].median()
        
        out_laps = laps[~laps['PitOutTime'].isnull() & (laps['LapNumber'] > 1)]
        if out_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_out = out_laps['LapTimeSec'].median()
        
        return max(15.0, (avg_in + avg_out) - (2 * avg_race))

    def analyze_degradation(self):
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        
        if self.visualize_fits:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Tyre Degradation: {self.gp} {self.year}", fontsize=14, fontweight='bold')
        
        for idx, comp in enumerate(compounds):
            comp_laps = self.laps[self.laps['Compound'] == comp]
            
            if len(comp_laps) < 3:
                print(f"WARNING: No data for {comp}. Using defaults.")
                self._use_defaults(comp)
                continue

            # IQR outlier removal
            Q1, Q3 = comp_laps['LapTimeSec'].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            clean = comp_laps[(comp_laps['LapTimeSec'] >= Q1 - 1.5*IQR) & 
                             (comp_laps['LapTimeSec'] <= Q3 + 1.5*IQR)]
            if len(clean) < 3: clean = comp_laps

            X, y = clean[['TyreLife']].values, clean['LapTimeSec'].values
            
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            reg = LinearRegression().fit(X_poly, y)
            
            c0, c1, c2 = reg.intercept_, reg.coef_[1], reg.coef_[2]
            if c2 < 0: c2 = 0.0
            
            y_pred = reg.predict(X_poly)
            self.model_quality[comp] = {'R2': r2_score(y, y_pred), 'MAE': mean_absolute_error(y, y_pred)}
            self.models[comp] = {'base_pace': c0, 'linear_degradation': c1, 'quadratic_degradation': c2}
            
            print(f"{comp}: Base={c0:.2f}s, Lin={c1:.4f}, Quad={c2:.6f}")
            
            if self.visualize_fits:
                axes[idx].scatter(X, y, alpha=0.5, s=20, color='blue')
                X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                axes[idx].plot(X_range, c0 + c1*X_range + c2*X_range**2, 'r-', linewidth=2)
                axes[idx].set_title(comp, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)

        # Sanity check
        found = [c for c in compounds if c in self.models]
        if len(found) == 3:
            for target in found:
                others = [c for c in found if c != target]
                pace_target = self.models[target]['base_pace']
                avg_others = sum(self.models[c]['base_pace'] for c in others) / 2
                
                if abs(pace_target - avg_others) > 5.0:
                    print(f"⚠️ ANOMALY in {target}! Overwriting with avg of {others}")
                    self.models[target] = {
                        'base_pace': avg_others,
                        'linear_degradation': sum(self.models[c]['linear_degradation'] for c in others) / 2,
                        'quadratic_degradation': sum(self.models[c]['quadratic_degradation'] for c in others) / 2
                    }

        if self.visualize_fits:
            plt.tight_layout()
            plt.show()

    def _use_defaults(self, compound):
        base = self.session_avg_pace + (0.5 if compound == 'HARD' else -0.5 if compound == 'SOFT' else 0)
        lin_map = {'SOFT': 0.05, 'MEDIUM': 0.03, 'HARD': 0.02}
        self.models[compound] = {
            'base_pace': base,
            'linear_degradation': lin_map[compound],
            'quadratic_degradation': config.NON_LINEAR_WEAR.get(compound, 0.002)
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
        for comp, q in self.model_quality.items():
            print(f"{comp:8} | R²={q['R2']:.3f} | MAE={q['MAE']:.2f}s")
        print("="*60 + "\n")