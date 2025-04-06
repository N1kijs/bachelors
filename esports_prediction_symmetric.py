import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from datetime import datetime
import joblib
import os

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the match data.
    
    Args:
        file_path: Path to the Excel file containing match data
        
    Returns:
        Preprocessed DataFrame
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Extract date from match_id for chronological ordering
    # Example match_id: complexity-vs-big-05-03-2023
    df['match_date'] = df['match_id'].str.extract(r'(\d{2}-\d{2}-\d{4})')
    df['match_date'] = pd.to_datetime(df['match_date'], format='%d-%m-%Y', errors='coerce')
    
    # If match_date extraction fails, create a backup ordering based on match_id
    if df['match_date'].isna().any():
        print("Warning: Some match dates could not be extracted. Using match_id for ordering.")
        df['match_order'] = pd.factorize(df['match_id'])[0]
    else:
        df['match_order'] = df['match_date'].rank(method='dense')
    
    # Parse numeric columns
    numeric_cols = ['kills', 'deaths', 'assists', 'adr', 'map_score_team1', 'map_score_team2']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate K/D differential
    df['kd_diff'] = df['kills'] - df['deaths']
    
    # Parse open_duels (format: "3:3") with handling for missing values
    # First, replace NaN values with a default value
    df['open_duels'] = df['open_duels'].fillna('0:0')
    
    # Then extract the won and lost duels
    df['open_duels_won'] = pd.to_numeric(df['open_duels'].str.split(':', expand=True)[0], errors='coerce').fillna(0).astype(int)
    df['open_duels_lost'] = pd.to_numeric(df['open_duels'].str.split(':', expand=True)[1], errors='coerce').fillna(0).astype(int)
    df['open_duels_total'] = df['open_duels_won'] + df['open_duels_lost']
    
    # Handle division by zero
    df['open_duels_ratio'] = np.where(df['open_duels_total'] > 0, 
                                      df['open_duels_won'] / df['open_duels_total'], 
                                      0.5)  # Default to 0.5 ratio when no data
    
    # Handle multi_kills
    df['multi_kills'] = pd.to_numeric(df['multi_kills'], errors='coerce')
    
    # Add binary outcome (1 if player's team won, 0 if lost)
    # Note: map_winner column should contain the team name that won
    df['player_won'] = (df['team_name'].str.lower() == df['map_winner'].str.lower()).astype(int)
    
    return df

def calculate_player_stats(df):
    """
    Calculate player statistics based on historical data.
    For each player in each match, calculate their average performance 
    based on previous matches.
    
    Args:
        df: Preprocessed DataFrame with match data
        
    Returns:
        DataFrame with player statistics for each match
    """
    # Sort by player name and match order
    df = df.sort_values(['player_name', 'match_order'])
    
    # Initialize list to store player statistics
    player_stats = []
    
    # Get unique players
    unique_players = df['player_name'].unique()
    
    for player in unique_players:
        # Get matches for this player
        player_df = df[df['player_name'] == player].copy()
        
        # Process each match
        for i, (_, row) in enumerate(player_df.iterrows()):
            if i == 0:
                # First match for this player, use current match stats
                stats = {
                    'player_name': player,
                    'match_id': row['match_id'],
                    'match_order': row['match_order'],
                    'team_name': row['team_name'],
                    'map': row['map'],  # Store map information
                    'avg_kills': row['kills'],
                    'avg_deaths': row['deaths'],
                    'avg_assists': row['assists'],
                    'avg_adr': row['adr'],
                    'avg_open_duels_ratio': row['open_duels_ratio'],
                    'avg_multi_kills': row['multi_kills'],
                    'avg_kd_diff': row['kd_diff'],  # Add K/D differential
                    'win_rate': row['player_won'],
                    'player_won': row['player_won']
                }
            else:
                # Calculate average based on previous matches
                prev_df = player_df.iloc[:i]
                stats = {
                    'player_name': player,
                    'match_id': row['match_id'],
                    'match_order': row['match_order'],
                    'team_name': row['team_name'],
                    'map': row['map'],  # Store map information
                    'avg_kills': prev_df['kills'].mean(),
                    'avg_deaths': prev_df['deaths'].mean(),
                    'avg_assists': prev_df['assists'].mean(),
                    'avg_adr': prev_df['adr'].mean(),
                    'avg_open_duels_ratio': prev_df['open_duels_ratio'].mean(),
                    'avg_multi_kills': prev_df['multi_kills'].mean(),
                    'avg_kd_diff': prev_df['kd_diff'].mean(),  # Add K/D differential
                    'win_rate': prev_df['player_won'].mean(),
                    'player_won': row['player_won']
                }
            
            player_stats.append(stats)
    
    # Convert to DataFrame
    player_stats_df = pd.DataFrame(player_stats)
    
    return player_stats_df

def prepare_match_features(player_stats_df):
    """
    Prepare match features for logistic regression.
    Calculate team-level statistics and feature differences 
    with consistent ordering to ensure symmetry.
    
    Args:
        player_stats_df: DataFrame with player statistics
        
    Returns:
        DataFrame with match features
    """
    # Get unique matches
    unique_matches = player_stats_df[['match_id', 'match_order']].drop_duplicates().sort_values('match_order')
    
    # Initialize list to store match features
    match_features = []
    
    for _, match_row in unique_matches.iterrows():
        match_id = match_row['match_id']
        match_order = match_row['match_order']
        
        # Get data for this match
        match_df = player_stats_df[player_stats_df['match_id'] == match_id]
        
        # Store map information
        current_map = match_df['map'].iloc[0] if not match_df.empty else None
        
        # Get teams in this match
        teams = match_df['team_name'].unique()
        
        if len(teams) != 2:
            print(f"Warning: Match {match_id} does not have exactly 2 teams. Skipping.")
            continue
        
        # Sort teams alphabetically to ensure consistent ordering
        sorted_teams = sorted(teams)
        team1, team2 = sorted_teams
        
        # Get data for each team
        team1_df = match_df[match_df['team_name'] == team1]
        team2_df = match_df[match_df['team_name'] == team2]
        
        # Calculate team average stats
        team1_avg = {
            'avg_kills': team1_df['avg_kills'].mean(),
            'avg_deaths': team1_df['avg_deaths'].mean(),
            'avg_assists': team1_df['avg_assists'].mean(),
            'avg_adr': team1_df['avg_adr'].mean(),
            'avg_open_duels_ratio': team1_df['avg_open_duels_ratio'].mean(),
            'avg_multi_kills': team1_df['avg_multi_kills'].mean(),
            'avg_kd_diff': team1_df['avg_kd_diff'].mean(),  # Add K/D differential
            'win_rate': team1_df['win_rate'].mean()
        }
        
        team2_avg = {
            'avg_kills': team2_df['avg_kills'].mean(),
            'avg_deaths': team2_df['avg_deaths'].mean(),
            'avg_assists': team2_df['avg_assists'].mean(),
            'avg_adr': team2_df['avg_adr'].mean(),
            'avg_open_duels_ratio': team2_df['avg_open_duels_ratio'].mean(),
            'avg_multi_kills': team2_df['avg_multi_kills'].mean(),
            'avg_kd_diff': team2_df['avg_kd_diff'].mean(),  # Add K/D differential
            'win_rate': team2_df['win_rate'].mean()
        }
        
        # Calculate feature differences (team1 - team2)
        # For deaths, we invert the difference so that positive means better for team1
        feature_dict = {
            'match_id': match_id,
            'match_order': match_order,
            'map': current_map,  # Store map information
            'team1': team1,
            'team2': team2,
            'kills_diff': team1_avg['avg_kills'] - team2_avg['avg_kills'],
            'deaths_diff': team2_avg['avg_deaths'] - team1_avg['avg_deaths'],  # Inverted for consistency
            'kd_diff_diff': team1_avg['avg_kd_diff'] - team2_avg['avg_kd_diff'],  # Add K/D differential
            'assists_diff': team1_avg['avg_assists'] - team2_avg['avg_assists'],
            'adr_diff': team1_avg['avg_adr'] - team2_avg['avg_adr'],
            'open_duels_ratio_diff': team1_avg['avg_open_duels_ratio'] - team2_avg['avg_open_duels_ratio'],
            'multi_kills_diff': team1_avg['avg_multi_kills'] - team2_avg['avg_multi_kills'],
            'win_rate_diff': team1_avg['win_rate'] - team2_avg['win_rate'],
            'team1_won': 1 if team1_df['player_won'].iloc[0] == 1 else 0  # Assuming all players in team1 have same outcome
        }
        
        # Now create a mirrored entry with team2 and team1 swapped
        # This enforces symmetry in the training data
        mirrored_dict = {
            'match_id': match_id + "_mirrored",
            'match_order': match_order,
            'map': current_map,
            'team1': team2,  # Swap teams
            'team2': team1,  # Swap teams
            'kills_diff': team2_avg['avg_kills'] - team1_avg['avg_kills'],  # Swap differences
            'deaths_diff': team1_avg['avg_deaths'] - team2_avg['avg_deaths'],  # Keep inverted
            'kd_diff_diff': team2_avg['avg_kd_diff'] - team1_avg['avg_kd_diff'],
            'assists_diff': team2_avg['avg_assists'] - team1_avg['avg_assists'],
            'adr_diff': team2_avg['avg_adr'] - team1_avg['avg_adr'],
            'open_duels_ratio_diff': team2_avg['avg_open_duels_ratio'] - team1_avg['avg_open_duels_ratio'],
            'multi_kills_diff': team2_avg['avg_multi_kills'] - team1_avg['avg_multi_kills'],
            'win_rate_diff': team2_avg['win_rate'] - team1_avg['win_rate'],
            'team1_won': 1 if team2_df['player_won'].iloc[0] == 1 else 0  # Swap outcome
        }
        
        # Add both the original and mirrored features
        match_features.append(feature_dict)
        match_features.append(mirrored_dict)
    
    # Convert to DataFrame
    match_features_df = pd.DataFrame(match_features)
    
    return match_features_df

def split_chronological(df, test_size=0.3, newest_first=False):
    """
    Split data chronologically.
    
    Args:
        df: DataFrame with match features
        test_size: Proportion of data to use for testing
        newest_first: If True, train on newest data and test on oldest
        
    Returns:
        train_df, test_df: Training and testing DataFrames
    """
    # Sort by match_order
    df_sorted = df.sort_values('match_order', ascending=not newest_first)
    
    # Calculate split point
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    # Split the data
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    return train_df, test_df

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate a logistic regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        feature_names: Names of the features
        
    Returns:
        Dictionary with model and evaluation results
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.coef_[0]))
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def save_model(results, output_dir='output'):
    """
    Save the trained model and scaler to disk.
    
    Args:
        results: Dictionary with model results
        output_dir: Directory to save model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_path = os.path.join(output_dir, 'esports_prediction_model.pkl')
    
    # Prepare model data for saving
    model_data = {
        'model': results['model'],
        'scaler': results['scaler'],
        'feature_columns': results['feature_columns']
    }
    
    # Save model data
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")

def main(file_path, save_model_file=True):
    """
    Main function to run the analysis.
    
    Args:
        file_path: Path to the Excel file with match data
        save_model_file: Whether to save the model to disk
        
    Returns:
        Dictionary with results
    """
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    print("Calculating player statistics...")
    player_stats_df = calculate_player_stats(df)
    
    print("Preparing match features...")
    match_features = prepare_match_features(player_stats_df)
    print(f"Total matches for modeling: {len(match_features) // 2}")  # Divide by 2 because we have mirrored entries
    
    # Feature columns for the model
    feature_columns = [
        'kills_diff', 'deaths_diff', 'assists_diff', 'adr_diff',
        'open_duels_ratio_diff', 'multi_kills_diff', 'win_rate_diff',
        'kd_diff_diff'  # Add the K/D differential feature
    ]
    
    # Split 1: Train on oldest, test on newest
    print("\n--- Split 1: Train on Oldest, Test on Newest ---")
    train_df_old, test_df_old = split_chronological(match_features, test_size=0.3, newest_first=False)
    print(f"Training data: {len(train_df_old)} matches")
    print(f"Testing data: {len(test_df_old)} matches")
    
    X_train_old = train_df_old[feature_columns].values
    y_train_old = train_df_old['team1_won'].values
    X_test_old = test_df_old[feature_columns].values
    y_test_old = test_df_old['team1_won'].values
    
    print("Training and evaluating model...")
    results_old = train_and_evaluate(X_train_old, y_train_old, X_test_old, y_test_old, feature_columns)
    
    print(f"Accuracy: {results_old['accuracy']:.4f}")
    print("Classification Report:")
    cls_report_old = results_old['classification_report']
    print(f"  Precision: {cls_report_old['weighted avg']['precision']:.4f}")
    print(f"  Recall: {cls_report_old['weighted avg']['recall']:.4f}")
    print(f"  F1-score: {cls_report_old['weighted avg']['f1-score']:.4f}")
    
    print("\nFeature Importance:")
    for feature, importance in sorted(results_old['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # Split 2: Train on newest, test on oldest
    print("\n--- Split 2: Train on Newest, Test on Oldest ---")
    train_df_new, test_df_new = split_chronological(match_features, test_size=0.3, newest_first=True)
    print(f"Training data: {len(train_df_new)} matches")
    print(f"Testing data: {len(test_df_new)} matches")
    
    X_train_new = train_df_new[feature_columns].values
    y_train_new = train_df_new['team1_won'].values
    X_test_new = test_df_new[feature_columns].values
    y_test_new = test_df_new['team1_won'].values
    
    print("Training and evaluating model...")
    results_new = train_and_evaluate(X_train_new, y_train_new, X_test_new, y_test_new, feature_columns)
    
    print(f"Accuracy: {results_new['accuracy']:.4f}")
    print("Classification Report:")
    cls_report_new = results_new['classification_report']
    print(f"  Precision: {cls_report_new['weighted avg']['precision']:.4f}")
    print(f"  Recall: {cls_report_new['weighted avg']['recall']:.4f}")
    print(f"  F1-score: {cls_report_new['weighted avg']['f1-score']:.4f}")
    
    print("\nFeature Importance:")
    for feature, importance in sorted(results_new['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    # Save final model (using the old-to-new split which is more practical)
    results_old['feature_columns'] = feature_columns
    if save_model_file:
        save_model(results_old)
    
    return {
        'oldest_to_newest': results_old,
        'newest_to_oldest': results_new,
        'feature_columns': feature_columns
    }


# python esports_prediction_symmetric.py --data match_data.xlsx
# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train symmetric esports prediction model')
    parser.add_argument('--data', required=True, help='Path to the Excel file with match data')
    parser.add_argument('--no-save', action='store_true', help='Do not save model to disk')
    
    args = parser.parse_args()
    results = main(args.data, not args.no_save)