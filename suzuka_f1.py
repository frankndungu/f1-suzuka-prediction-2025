import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Setup
cache_dir = 'f1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Define Suzuka-specific toggle
TARGET_CIRCUIT = 'Japanese Grand Prix'

drivers_2025 = [
    # Ferrari
    {'DriverNumber': 16, 'Abbreviation': 'LEC', 'FullName': 'Charles Leclerc', 'Team': 'Ferrari'},
    {'DriverNumber': 44, 'Abbreviation': 'HAM', 'FullName': 'Lewis Hamilton', 'Team': 'Ferrari'},

    # Mercedes-AMG Petronas
    {'DriverNumber': 63, 'Abbreviation': 'RUS', 'FullName': 'George Russell', 'Team': 'Mercedes'},
    {'DriverNumber': 72, 'Abbreviation': 'ANT', 'FullName': 'Andrea Kimi Antonelli', 'Team': 'Mercedes'},

    # Red Bull Racing
    {'DriverNumber': 1, 'Abbreviation': 'VER', 'FullName': 'Max Verstappen', 'Team': 'Red Bull Racing'},
    {'DriverNumber': 40, 'Abbreviation': 'LAW', 'FullName': 'Liam Lawson', 'Team': 'Red Bull Racing'},

    # McLaren
    {'DriverNumber': 4, 'Abbreviation': 'NOR', 'FullName': 'Lando Norris', 'Team': 'McLaren'},
    {'DriverNumber': 81, 'Abbreviation': 'PIA', 'FullName': 'Oscar Piastri', 'Team': 'McLaren'},

    # Aston Martin
    {'DriverNumber': 14, 'Abbreviation': 'ALO', 'FullName': 'Fernando Alonso', 'Team': 'Aston Martin'},
    {'DriverNumber': 18, 'Abbreviation': 'STR', 'FullName': 'Lance Stroll', 'Team': 'Aston Martin'},

    # Alpine
    {'DriverNumber': 10, 'Abbreviation': 'GAS', 'FullName': 'Pierre Gasly', 'Team': 'Alpine'},
    {'DriverNumber': 5, 'Abbreviation': 'DOO', 'FullName': 'Jack Doohan', 'Team': 'Alpine'},

    # Williams
    {'DriverNumber': 23, 'Abbreviation': 'ALB', 'FullName': 'Alexander Albon', 'Team': 'Williams'},
    {'DriverNumber': 55, 'Abbreviation': 'SAI', 'FullName': 'Carlos Sainz Jr.', 'Team': 'Williams'},

    # Haas
    {'DriverNumber': 31, 'Abbreviation': 'OCO', 'FullName': 'Esteban Ocon', 'Team': 'Haas F1 Team'},
    {'DriverNumber': 87, 'Abbreviation': 'BEA', 'FullName': 'Oliver Bearman', 'Team': 'Haas F1 Team'},

    # Kick Sauber
    {'DriverNumber': 27, 'Abbreviation': 'HUL', 'FullName': 'Nico Hülkenberg', 'Team': 'Kick Sauber'},
    {'DriverNumber': 50, 'Abbreviation': 'BOR', 'FullName': 'Gabriel Bortoleto', 'Team': 'Kick Sauber'},

    # Visa Cash App Racing Bulls (VCARB)
    {'DriverNumber': 22, 'Abbreviation': 'TSU', 'FullName': 'Yuki Tsunoda', 'Team': 'VCARB'},
    {'DriverNumber': 41, 'Abbreviation': 'HAD', 'FullName': 'Isack Hadjar', 'Team': 'VCARB'}
]
df_drivers = pd.DataFrame(drivers_2025)

# Load historical race data
seasons = [2022, 2023, 2024]
race_data = []
for season in seasons:
    for race in range(1, 23):
        try:
            session = fastf1.get_session(season, race, 'R')
            session.load()
            results = session.results[['DriverNumber', 'Position', 'Points', 'GridPosition']]
            results['Season'] = season
            results['RaceNumber'] = race
            results['Circuit'] = session.event['EventName']
            race_data.append(results)
        except:
            continue

# Combine
df_all = pd.concat(race_data)
df_all['Position'] = pd.to_numeric(df_all['Position'], errors='coerce').fillna(25)
df_all['GridPosition'] = pd.to_numeric(df_all['GridPosition'], errors='coerce').fillna(25)
df_all['IsJapanGP'] = df_all['Circuit'].str.contains('Japan|Suzuka', case=False).astype(int)
df_all['ExperienceCount'] = df_all.groupby('DriverNumber').cumcount() + 1
df_all['RollingPoints'] = df_all.groupby('DriverNumber')['Points'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

# Merge with driver info
df = pd.merge(df_all, df_drivers, on='DriverNumber', how='inner')

# Encode teams
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
team_encoded = encoder.fit_transform(df[['Team']])
team_encoded_df = pd.DataFrame(team_encoded, columns=encoder.get_feature_names_out(['Team']))

features = pd.concat([
    df[['GridPosition', 'Season', 'IsJapanGP', 'ExperienceCount', 'RollingPoints']].reset_index(drop=True),
    team_encoded_df.reset_index(drop=True)
], axis=1)

model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(features, df['Position'])

# Prediction for Suzuka
suzuka_preds = []
for driver in drivers_2025:
    team = driver['Team']
    past_perf = df[df['Team'] == team]

    # Boost McLaren's weight slightly based on strong recent form
    if team == 'McLaren':
        past_perf = past_perf.copy()
        past_perf['RollingPoints'] *= 1.15  # +15% performance boost
    avg_grid = past_perf['GridPosition'].mean()
    avg_points = past_perf['RollingPoints'].mean()
    exp = len(df[df['DriverNumber'] == driver['DriverNumber']]) + 1

    row = pd.DataFrame({
        'GridPosition': [avg_grid if not np.isnan(avg_grid) else 10],
        'Season': [2025],
        'IsJapanGP': [1],
        'ExperienceCount': [exp],
        'RollingPoints': [avg_points if not np.isnan(avg_points) else 0],
        'Team': [team],
        'FullName': [driver['FullName']]
    })
    suzuka_preds.append(row)

df_pred = pd.concat(suzuka_preds)
team_encoded_pred = encoder.transform(df_pred[['Team']])
team_encoded_pred_df = pd.DataFrame(team_encoded_pred, columns=encoder.get_feature_names_out(['Team']))

X_pred = pd.concat([
    df_pred[['GridPosition', 'Season', 'IsJapanGP', 'ExperienceCount', 'RollingPoints']].reset_index(drop=True),
    team_encoded_pred_df.reset_index(drop=True)
], axis=1)

# Predict
ensemble_preds = [model.predict(X_pred) for _ in range(25)]
df_pred['MeanPos'] = np.mean(ensemble_preds, axis=0)
df_pred['StdDev'] = np.std(ensemble_preds, axis=0)
df_pred = df_pred.sort_values('MeanPos').reset_index(drop=True)
df_pred['FinalPos'] = df_pred.index + 1

# Filter out unlikely podiums
likely_teams = ['Red Bull Racing', 'Ferrari', 'Mercedes', 'McLaren']
df_pred['ValidPodium'] = df_pred.apply(lambda row: row['Team'] in likely_teams or row['ExperienceCount'] > 10, axis=1)

# Visual
plt.figure(figsize=(10, 8))
sns.barplot(y='FullName', x='MeanPos', data=df_pred, palette='viridis', xerr=df_pred['StdDev'])
plt.xlabel('Predicted Finish Position (Lower is Better)')
plt.ylabel('Driver')
plt.title('🏁 Suzuka 2025 F1 AI Prediction')
plt.tight_layout()
plt.savefig('suzuka_prediction_confidence.png')

print("\nPredicted Podium for Suzuka 2025:")
print(df_pred[['FinalPos', 'FullName', 'Team']].head(3))
