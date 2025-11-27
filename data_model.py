import os
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import config

if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

class TyreDataModeler:
    def __init__(self, year, gp, session_type='R'):
        self.year = year
        self.gp = gp
        self.session_type = session_type
        self.laps = None
        self.models = {} 
        self.pit_loss = config.DEFAULT_PIT_LOSS
        
    def load_and_clean_data(self):
        print(f"Loading {self.gp} {self.year}...")
        session = fastf1.get_session(self.year, self.gp, self.session_type)
        session.load()
        
        # Calculate the pit loss
        self.pit_loss = self._calculate_pit_loss(session)
        print(f"--> Pit Loss Calcolata (Mediana): {self.pit_loss:.2f}s")
        
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
        compounds = ['SOFT', 'MEDIUM', 'HARD']
        for compound in compounds:
            comp_laps = self.laps[self.laps['Compound'] == compound]
            if len(comp_laps) < 10: continue

            q_high = comp_laps['LapTimeSec'].quantile(0.95)
            comp_laps = comp_laps[comp_laps['LapTimeSec'] < q_high]

            X = comp_laps[['TyreLife']].values 
            y = comp_laps['LapTimeSec'].values 
            
            reg = LinearRegression().fit(X, y)
            self.models[compound] = {
                'base_pace': reg.intercept_,
                'degradation': reg.coef_[0]
            }

    def get_simulation_data(self):
        total_laps = int(self.laps['LapNumber'].max())
        defaults = {'base_pace': 100.0, 'degradation': 0.1}
        for comp in ['SOFT', 'MEDIUM', 'HARD']:
            if comp not in self.models:
                self.models[comp] = defaults.copy()
        return self.models, total_laps, self.pit_loss