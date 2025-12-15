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
        self.model_quality = {}  # NEW: Store R² for each compound
        self.pit_loss = config.DEFAULT_PIT_LOSS
        
    def load_and_clean_data(self):
        print(f"Loading {self.gp} {self.year}...")
        session = fastf1.get_session(self.year, self.gp, self.session_type)
        session.load()
        
        # Calculate the pit loss
        self.pit_loss = self._calculate_pit_loss(session)
        print(f"--> Pit Loss Calculated (Median): {self.pit_loss:.2f}s")
        
        laps = session.laps.pick_quicklaps()
        laps = laps.loc[(laps['PitOutTime'].isnull()) & (laps['PitInTime'].isnull())]
        
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        self.laps = laps[['Driver', 'LapNumber', 'LapTimeSec', 'Compound', 'TyreLife']]
        print(f"Data Loaded. {len(self.laps)} clean laps found.")

    def _calculate_pit_loss(self, session):
        """
        Calculate the time loss using the median method to ignore slow pit stops or incidents.
        Formula: Loss = (Median InLap + Median OutLap) - (2 * Median CleanLap)
        """
        laps = session.laps.pick_track_status('1')
        laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
        
        # 1. Clean Laps (Median Race Lap)
        clean_laps = laps[laps['PitOutTime'].isnull() & laps['PitInTime'].isnull()]
        if clean_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_race = clean_laps['LapTimeSec'].median()
        
        # 2. In-Laps (Median)
        in_laps = laps[~laps['PitInTime'].isnull()]
        if in_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_in = in_laps['LapTimeSec'].median()
        
        # 3. Out-Laps (Median)
        out_laps = laps[~laps['PitOutTime'].isnull()]
        out_laps = out_laps[out_laps['LapNumber'] > 1] 
        if out_laps.empty: return config.DEFAULT_PIT_LOSS
        avg_out = out_laps['LapTimeSec'].median()
        
        loss = (avg_in + avg_out) - (2 * avg_race)
        return max(15.0, loss)

    def analyze_degradation(self):
        """
        FIXED: Uses Polynomial Regression (degree 2) to capture non-linear degradation.
        
        Model: LapTime = c0 + c1*t + c2*t²
        where:
            c0 = base_pace (fresh tyre lap time)
            c1 = linear_degradation (fuel burn + light wear)
            c2 = quadratic_degradation (structural fatigue, "the cliff")
        """
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        
        if self.visualize_fits:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Tyre Degradation Analysis: {self.gp} {self.year}", fontsize=14, fontweight='bold')
        
        for idx, compound in enumerate(compounds):
            comp_laps = self.laps[self.laps['Compound'] == compound]
            
            # Check minimum data requirement
            if len(comp_laps) < 10:
                print(f"WARNING: Insufficient data for {compound} ({len(comp_laps)} laps). Using defaults.")
                self._use_defaults(compound)
                continue

            # --- IMPROVED OUTLIER REMOVAL (IQR Method) ---
            Q1 = comp_laps['LapTimeSec'].quantile(0.25)
            Q3 = comp_laps['LapTimeSec'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Remove outliers beyond 1.5*IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            comp_laps_clean = comp_laps[
                (comp_laps['LapTimeSec'] >= lower_bound) & 
                (comp_laps['LapTimeSec'] <= upper_bound)
            ]
            
            outliers_removed = len(comp_laps) - len(comp_laps_clean)
            if outliers_removed > 0:
                print(f"{compound}: Removed {outliers_removed} outliers (IQR method)")

            # Check if enough data remains after cleaning
            if len(comp_laps_clean) < 10:
                print(f"WARNING: Too few laps after cleaning for {compound}. Using defaults.")
                self._use_defaults(compound)
                continue

            # --- POLYNOMIAL REGRESSION (DEGREE 2) ---
            X = comp_laps_clean[['TyreLife']].values 
            y = comp_laps_clean['LapTimeSec'].values 
            
            # Create polynomial features: [1, t, t²]
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            
            # Fit the model
            reg = LinearRegression().fit(X_poly, y)
            
            # Extract coefficients: y = c0 + c1*t + c2*t²
            c0 = reg.intercept_  # Base pace
            c1, c2 = reg.coef_[1], reg.coef_[2]  # Linear and quadratic terms
            
            # --- MODEL VALIDATION ---
            y_pred = reg.predict(X_poly)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            self.model_quality[compound] = {'R2': r2, 'MAE': mae}
            
            # Quality check: R² should be > 0.3 for meaningful regression
            if r2 < 0.3:
                print(f"WARNING: Poor model fit for {compound} (R²={r2:.3f}). Using defaults.")
                self._use_defaults(compound)
            else:
                # CRITICAL FIX: Store quadratic coefficient instead of hardcoded value
                self.models[compound] = {
                    'base_pace': c0,
                    'linear_degradation': c1,
                    'quadratic_degradation': c2  # NEW: Real coefficient from data
                }
                print(f"{compound}: Base={c0:.2f}s, Linear={c1:.4f}s/lap, Quadratic={c2:.6f}s/lap², R²={r2:.3f}, MAE={mae:.3f}s")
            
            # --- VISUALIZATION (Optional) ---
            if self.visualize_fits:
                ax = axes[idx]
                
                # Scatter plot of actual data
                ax.scatter(X, y, alpha=0.5, s=20, label='Actual Laps', color='blue')
                
                # Plot the fitted curve
                X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                X_range_poly = poly.transform(X_range)
                y_range_pred = reg.predict(X_range_poly)
                ax.plot(X_range, y_range_pred, color='red', linewidth=2, label=f'Fit (R²={r2:.2f})')
                
                # Styling
                ax.set_title(f"{compound}", fontweight='bold')
                ax.set_xlabel("Tyre Life (laps)")
                ax.set_ylabel("Lap Time (s)")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        if self.visualize_fits:
            plt.tight_layout()
            plt.show()

    def _use_defaults(self, compound):
        """
        Fallback to config-based values when data is insufficient or model quality is poor.
        """
        # Use base pace from config or a reasonable estimate
        base_pace = 90.0  # Generic baseline
        
        # Use the hardcoded NON_LINEAR_WEAR as quadratic coefficient
        quad_coef = config.NON_LINEAR_WEAR.get(compound, 0.002)
        
        # Estimate linear degradation from compound type
        linear_map = {'SOFT': 0.05, 'MEDIUM': 0.03, 'HARD': 0.02}
        linear_coef = linear_map.get(compound, 0.03)
        
        self.models[compound] = {
            'base_pace': base_pace,
            'linear_degradation': linear_coef,
            'quadratic_degradation': quad_coef
        }
        self.model_quality[compound] = {'R2': 0.0, 'MAE': float('inf')}

    def get_simulation_data(self):
        """
        Returns the calibrated models, race length, and pit loss.
        
        IMPORTANT: Models now contain 'quadratic_degradation' instead of relying on config.
        """
        total_laps = int(self.laps['LapNumber'].max())
        
        # Ensure all compounds have models (use defaults if missing)
        for comp in ['SOFT', 'MEDIUM', 'HARD']:
            if comp not in self.models:
                self._use_defaults(comp)
        
        return self.models, total_laps, self.pit_loss
    
    def print_model_summary(self):
        """
        Utility function to print a summary of model quality.
        """
        print("\n" + "="*60)
        print("MODEL QUALITY SUMMARY")
        print("="*60)
        for compound, quality in self.model_quality.items():
            r2 = quality['R2']
            mae = quality['MAE']
            status = "✓ GOOD" if r2 > 0.5 else "⚠ POOR" if r2 > 0.3 else "✗ BAD"
            print(f"{compound:8} | R²={r2:.3f} | MAE={mae:.2f}s | {status}")
        print("="*60 + "\n")