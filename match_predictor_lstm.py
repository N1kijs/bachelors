import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import argparse
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class LSTMModel(nn.Module):
    """
    LSTM model for sequence prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, dropout=0)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size // 2, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Input shape: batch_size, seq_len, input_size
        lstm_out, _ = self.lstm1(x)
        
        # Apply batch norm to last timestep output
        lstm_out_last = lstm_out[:, -1, :]
        normalized = self.bn1(lstm_out_last)
        dropped = self.dropout1(normalized)
        
        # Reshape for second LSTM
        reshaped = dropped.unsqueeze(1)
        lstm_out2, _ = self.lstm2(reshaped)
        
        # Apply batch norm to last timestep output
        lstm_out2_last = lstm_out2[:, -1, :]
        normalized2 = self.bn2(lstm_out2_last)
        dropped2 = self.dropout2(normalized2)
        
        # Dense layers
        fc1_out = self.fc1(dropped2)
        normalized3 = self.bn3(fc1_out)
        dropped3 = self.dropout3(normalized3)
        activated = self.relu(dropped3)
        
        # Output layer
        fc2_out = self.fc2(activated)
        output = self.sigmoid(fc2_out)
        
        return output


class LSTMMatchPredictor:
    """
    Utility class for predicting esports match outcomes using PyTorch LSTM model.
    """
    
    def __init__(self, model_path, model_info_path, scaler_path, features_path, data_path=None):
        """
        Initialize the predictor with a trained LSTM model.
        
        Args:
            model_path: Path to saved LSTM model weights
            model_info_path: Path to saved model architecture info
            scaler_path: Path to saved feature scaler
            features_path: Path to saved feature columns
            data_path: Optional path to historical data for player statistics
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model architecture info
        self.model_info = joblib.load(model_info_path)
        
        # Create model with same architecture
        self.model = LSTMModel(
            input_size=self.model_info['input_size'],
            hidden_size=self.model_info['hidden_size'],
            num_layers=self.model_info['num_layers'],
            dropout=self.model_info['dropout']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        
        # Load scaler and feature columns
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)
        
        # Get sequence length
        self.sequence_length = self.model_info['sequence_length']
        
        # Common maps in competitive play (sorted by popularity)
        self.common_maps = [
            'dust2', 'mirage', 'inferno', 'nuke', 'overpass', 
            'ancient', 'vertigo', 'anubis'
        ]
        
        self.player_stats = {}
        self.team_stats = {}
        self.team_sequences = {}
        self.match_maps = {}  # Store map played for each match
        
        # Load historical data if provided
        if data_path:
            self.load_historical_data(data_path)
    
    def load_historical_data(self, data_path):
        """
        Load historical match data to calculate player and team statistics.
        
        Args:
            data_path: Path to Excel file with historical match data
        """
        print(f"Loading historical data from {data_path}...")
        df = pd.read_excel(data_path)
        
        # Ensure match_order column exists
        if 'match_order' not in df.columns:
            print("'match_order' column not found. Creating it from match_id...")
            # Extract date from match_id for chronological ordering
            # Example match_id: complexity-vs-big-05-03-2023
            df['match_date'] = df['match_id'].str.extract(r'(\d{2}-\d{2}-\d{4})')
            df['match_date'] = pd.to_datetime(df['match_date'], format='%d-%m-%Y', errors='coerce')
            
            # If match_date extraction fails, create a backup ordering based on match_id
            if df['match_date'].isna().any():
                print("  Warning: Some match dates could not be extracted. Using match_id for ordering.")
                df['match_order'] = pd.factorize(df['match_id'])[0]
            else:
                df['match_order'] = df['match_date'].rank(method='dense')
        
        # Store map information for each match
        for _, row in df.iterrows():
            if pd.notna(row['match_id']) and pd.notna(row['map']):
                self.match_maps[row['match_id']] = row['map']
        
        # Extract player statistics
        for _, row in df.iterrows():
            player_name = row['player_name']
            team_name = row['team_name']
            match_id = row['match_id']
            
            # Initialize player stats if needed
            if player_name not in self.player_stats:
                self.player_stats[player_name] = {
                    'kills': [],
                    'deaths': [],
                    'assists': [],
                    'adr': [],
                    'open_duels_won': [],
                    'open_duels_total': [],
                    'multi_kills': [],
                    'kd_diff': [],
                    'wins': [],
                    'matches': [],
                    'match_order': [],
                    'maps': [],
                    'team': team_name
                }
            
            # Parse open_duels (format: "3:3") with error handling
            try:
                # First check if it's a string and contains ':'
                if isinstance(row['open_duels'], str) and ':' in row['open_duels']:
                    open_duels_parts = row['open_duels'].split(':')
                    open_duels_won = int(open_duels_parts[0])
                    open_duels_lost = int(open_duels_parts[1])
                else:
                    # Default values for missing or improperly formatted data
                    open_duels_won = 0
                    open_duels_lost = 0
                open_duels_total = open_duels_won + open_duels_lost
            except (ValueError, TypeError, AttributeError):
                # Handle any conversion errors
                open_duels_won = 0
                open_duels_lost = 0
                open_duels_total = 0
            
            # Check if player won, with error handling
            try:
                if pd.isna(row['map_winner']) or pd.isna(row['team_name']):
                    player_won = 0  # Default to loss if data is missing
                else:
                    # Try to compare in lowercase to handle case differences
                    team_name_lower = str(row['team_name']).lower()
                    map_winner_lower = str(row['map_winner']).lower()
                    player_won = 1 if team_name_lower == map_winner_lower else 0
            except (AttributeError, TypeError):
                player_won = 0  # Default to loss if any errors
            
            # Append stats with error handling
            try:
                kills = int(row['kills']) if pd.notna(row['kills']) else 0
                deaths = int(row['deaths']) if pd.notna(row['deaths']) else 0
                assists = int(row['assists']) if pd.notna(row['assists']) else 0
                adr = float(row['adr']) if pd.notna(row['adr']) else 0.0
                multi_kills = int(row['multi_kills']) if pd.notna(row['multi_kills']) else 0
                map_name = str(row['map']) if pd.notna(row['map']) else 'unknown'
                match_order = float(row['match_order']) if pd.notna(row['match_order']) else 0.0
                
                # Calculate K/D differential
                kd_diff = kills - deaths
                
                # Update player stats
                self.player_stats[player_name]['kills'].append(kills)
                self.player_stats[player_name]['deaths'].append(deaths)
                self.player_stats[player_name]['assists'].append(assists)
                self.player_stats[player_name]['adr'].append(adr)
                self.player_stats[player_name]['open_duels_won'].append(open_duels_won)
                self.player_stats[player_name]['open_duels_total'].append(open_duels_total)
                self.player_stats[player_name]['multi_kills'].append(multi_kills)
                self.player_stats[player_name]['kd_diff'].append(kd_diff)
                self.player_stats[player_name]['wins'].append(player_won)
                self.player_stats[player_name]['matches'].append(match_id)
                self.player_stats[player_name]['match_order'].append(match_order)
                self.player_stats[player_name]['maps'].append(map_name)
                
                # Update team stats too
                if team_name not in self.team_stats:
                    self.team_stats[team_name] = {
                        'match_ids': [],
                        'match_order': [],
                        'maps': [],
                        'opponents': [],
                        'stats': [],
                        'wins': []
                    }
                
                # Only add match once per team (first player encountered)
                if match_id not in self.team_stats[team_name]['match_ids']:
                    self.team_stats[team_name]['match_ids'].append(match_id)
                    self.team_stats[team_name]['match_order'].append(match_order)
                    self.team_stats[team_name]['maps'].append(map_name)
                    self.team_stats[team_name]['wins'].append(player_won)
                    
                    # Extract opponent team
                    opponent_found = False
                    for p_name, p_data in self.player_stats.items():
                        if (p_data['team'] != team_name and 
                            match_id in p_data['matches']):
                            opponent_team = p_data['team']
                            opponent_found = True
                            break
                            
                    if opponent_found:
                        self.team_stats[team_name]['opponents'].append(opponent_team)
                    else:
                        self.team_stats[team_name]['opponents'].append('unknown')
            
            except (ValueError, TypeError) as e:
                print(f"Warning: Error parsing stats for player {player_name} in match {row['match_id']}: {e}")
        
        print(f"Loaded data for {len(self.player_stats)} players and {len(self.team_stats)} teams")
        
        # Process team sequences
        self._prepare_team_sequences()
    
    def _prepare_team_sequences(self):
        """
        Process the loaded data into team sequences for LSTM prediction.
        """
        for team_name, team_data in self.team_stats.items():
            # Sort matches by match_order
            indices = np.argsort(team_data['match_order'])
            
            sorted_matches = {}
            for key in ['match_ids', 'match_order', 'maps', 'opponents', 'wins']:
                sorted_matches[key] = [team_data[key][i] for i in indices]
            
            # Store sorted match data
            self.team_stats[team_name].update(sorted_matches)
            
            # Get player stats for each match
            match_stats = []
            for match_id in self.team_stats[team_name]['match_ids']:
                # Get players for this team and match
                team_match_players = []
                for player_name, player_data in self.player_stats.items():
                    if player_data['team'] == team_name and match_id in player_data['matches']:
                        match_idx = player_data['matches'].index(match_id)
                        team_match_players.append({
                            'player': player_name,
                            'kills': player_data['kills'][match_idx],
                            'deaths': player_data['deaths'][match_idx],
                            'assists': player_data['assists'][match_idx],
                            'adr': player_data['adr'][match_idx],
                            'open_duels_ratio': (player_data['open_duels_won'][match_idx] / 
                                               player_data['open_duels_total'][match_idx] 
                                               if player_data['open_duels_total'][match_idx] > 0 else 0.5),
                            'multi_kills': player_data['multi_kills'][match_idx],
                            'kd_diff': player_data['kd_diff'][match_idx]
                        })
                
                # Calculate team averages
                if team_match_players:
                    avg_stats = {
                        'kills': np.mean([p['kills'] for p in team_match_players]),
                        'deaths': np.mean([p['deaths'] for p in team_match_players]),
                        'assists': np.mean([p['assists'] for p in team_match_players]),
                        'adr': np.mean([p['adr'] for p in team_match_players]),
                        'open_duels_ratio': np.mean([p['open_duels_ratio'] for p in team_match_players]),
                        'multi_kills': np.mean([p['multi_kills'] for p in team_match_players]),
                        'kd_diff': np.mean([p['kd_diff'] for p in team_match_players])
                    }
                    match_stats.append(avg_stats)
                else:
                    # No player data for this match
                    match_stats.append({
                        'kills': 0,
                        'deaths': 0,
                        'assists': 0,
                        'adr': 0,
                        'open_duels_ratio': 0.5,
                        'multi_kills': 0,
                        'kd_diff': 0
                    })
            
            # Store match stats
            self.team_stats[team_name]['stats'] = match_stats
            
            # Prepare sequences for LSTM
            if len(match_stats) >= self.sequence_length:
                sequences = []
                for i in range(len(match_stats) - self.sequence_length + 1):
                    seq = match_stats[i:i+self.sequence_length]
                    sequences.append(seq)
                
                self.team_sequences[team_name] = sequences
    
    def get_team_sequence(self, team_name, opponent=None, map_name=None, recent=True):
        """
        Get the most recent sequence for a team.
        
        Args:
            team_name: Name of the team
            opponent: Optional opponent team name to filter by
            map_name: Optional map name to filter by
            recent: Whether to get the most recent sequence
            
        Returns:
            Team sequence for LSTM prediction
        """
        if team_name not in self.team_stats:
            print(f"Warning: No historical data for team {team_name}")
            return None
        
        team_data = self.team_stats[team_name]
        
        # If not enough matches for sequence
        if len(team_data['stats']) < self.sequence_length:
            print(f"Warning: Not enough matches for team {team_name} (has {len(team_data['stats'])}, need {self.sequence_length})")
            return None
        
        # Filter by opponent if provided
        if opponent:
            opponent_indices = [i for i, opp in enumerate(team_data['opponents']) if opp == opponent]
            
            # If we have enough matches against this opponent
            if len(opponent_indices) >= self.sequence_length:
                # Sort by match order
                opponent_indices = sorted(opponent_indices, key=lambda i: team_data['match_order'][i])
                
                # Get sequence
                if recent:
                    # Most recent sequence
                    seq_indices = opponent_indices[-self.sequence_length:]
                else:
                    # Overall sequence 
                    seq_indices = opponent_indices[:self.sequence_length]
                
                return [team_data['stats'][i] for i in seq_indices]
        
        # Filter by map if provided
        if map_name:
            map_indices = [i for i, m in enumerate(team_data['maps']) if m.lower() == map_name.lower()]
            
            # If we have enough matches on this map
            if len(map_indices) >= self.sequence_length:
                # Sort by match order
                map_indices = sorted(map_indices, key=lambda i: team_data['match_order'][i])
                
                # Get sequence
                if recent:
                    # Most recent sequence
                    seq_indices = map_indices[-self.sequence_length:]
                else:
                    # Overall sequence
                    seq_indices = map_indices[:self.sequence_length]
                
                return [team_data['stats'][i] for i in seq_indices]
        
        # Default: return most recent sequence
        if recent:
            return team_data['stats'][-self.sequence_length:]
        else:
            return team_data['stats'][:self.sequence_length]
    
    def get_team_vs_team_features(self, team1_sequence, team2_sequence):
        """
        Calculate feature differences for team vs team prediction.
        
        Args:
            team1_sequence: Sequence of stats for team 1
            team2_sequence: Sequence of stats for team 2
            
        Returns:
            Feature differences for LSTM prediction
        """
        if not team1_sequence or not team2_sequence:
            return None
        
        # Calculate differences for each match in sequence
        diffs = []
        for t1_stats, t2_stats in zip(team1_sequence, team2_sequence):
            diff = {
                'kills_diff': t1_stats['kills'] - t2_stats['kills'],
                'deaths_diff': t2_stats['deaths'] - t1_stats['deaths'],  # Inverted for consistency
                'assists_diff': t1_stats['assists'] - t2_stats['assists'],
                'adr_diff': t1_stats['adr'] - t2_stats['adr'],
                'open_duels_ratio_diff': t1_stats['open_duels_ratio'] - t2_stats['open_duels_ratio'],
                'multi_kills_diff': t1_stats['multi_kills'] - t2_stats['multi_kills'],
                'kd_diff_diff': t1_stats['kd_diff'] - t2_stats['kd_diff']
            }
            diffs.append(diff)
        
        return diffs
    
    def predict_match(self, team1_name, team2_name, map_name=None):
        """
        Predict match outcome using LSTM model.
        
        Args:
            team1_name: Name of team 1
            team2_name: Name of team 2
            map_name: Optional map name
            
        Returns:
            Dictionary with prediction results
        """
        # Get team sequences
        team1_seq = self.get_team_sequence(team1_name, opponent=team2_name, map_name=map_name)
        team2_seq = self.get_team_sequence(team2_name, opponent=team1_name, map_name=map_name)
        
        # If we don't have enough data for head-to-head, use general sequences
        if not team1_seq or len(team1_seq) < self.sequence_length:
            print(f"Not enough head-to-head data. Using general sequences for {team1_name}.")
            team1_seq = self.get_team_sequence(team1_name, map_name=map_name)
            
        if not team2_seq or len(team2_seq) < self.sequence_length:
            print(f"Not enough head-to-head data. Using general sequences for {team2_name}.")
            team2_seq = self.get_team_sequence(team2_name, map_name=map_name)
        
        # If still don't have enough data, return error
        if not team1_seq or not team2_seq:
            return {
                'error': f"Not enough historical data for {team1_name} vs {team2_name}",
                'team1_win_probability': 0.5,  # Default to 50% probability
                'team1_win': None,
                'team2_win_probability': 0.5,
                'team2_win': None
            }
        
        # Calculate feature differences
        feature_diffs = self.get_team_vs_team_features(team1_seq, team2_seq)
        
        if not feature_diffs:
            return {
                'error': "Failed to calculate feature differences",
                'team1_win_probability': 0.5,
                'team1_win': None,
                'team2_win_probability': 0.5,
                'team2_win': None
            }
        
        # Extract features in the correct order (same as during training)
        diff_values = []
        for diff in feature_diffs:
            diff_values.append([diff[col] for col in self.feature_columns])
        
        # Transform to array for LSTM
        diff_array = np.array([diff_values])
        
        # Scale features
        diff_flat = diff_array.reshape(-1, diff_array.shape[2])
        diff_flat_scaled = self.scaler.transform(diff_flat)
        diff_scaled = diff_flat_scaled.reshape(diff_array.shape)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(diff_scaled).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            team1_win_prob = self.model(input_tensor).cpu().numpy()[0][0]
        
        team1_win = team1_win_prob >= 0.5
        
        # For symmetry, swap teams and predict again
        reversed_diffs = []
        for diff in feature_diffs:
            reversed_diff = {k: -v for k, v in diff.items()}
            # Special handling for deaths_diff which is already inverted
            reversed_diff['deaths_diff'] = -diff['deaths_diff']
            reversed_diffs.append(reversed_diff)
        
        # Extract reversed features
        reversed_values = []
        for diff in reversed_diffs:
            reversed_values.append([diff[col] for col in self.feature_columns])
        
        # Predict with swapped teams
        reverse_array = np.array([reversed_values])
        reverse_flat = reverse_array.reshape(-1, reverse_array.shape[2])
        reverse_flat_scaled = self.scaler.transform(reverse_flat)
        reverse_scaled = reverse_flat_scaled.reshape(reverse_array.shape)
        
        # Convert to tensor
        reverse_tensor = torch.FloatTensor(reverse_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            team2_win_as_team1_prob = self.model(reverse_tensor).cpu().numpy()[0][0]
        
        # Average the probabilities for better symmetry
        team1_win_prob_final = (team1_win_prob + (1 - team2_win_as_team1_prob)) / 2
        team1_win_final = team1_win_prob_final >= 0.5
        
        return {
            'map': map_name,
            'team1_stats': team1_seq[-1],  # Most recent stats
            'team2_stats': team2_seq[-1],  # Most recent stats
            'team1_win_probability': float(team1_win_prob_final),
            'team1_win': bool(team1_win_final),
            'team2_win_probability': 1 - float(team1_win_prob_final),
            'team2_win': not bool(team1_win_final),
            'head_to_head_sequence': feature_diffs,
            'raw_prediction': float(team1_win_prob)
        }
    
    def predict_series(self, team1_name, team2_name, maps=None):
        """
        Predict outcome of a best-of-3 series.
        
        Args:
            team1_name: Name of team 1
            team2_name: Name of team 2
            maps: Optional list of maps for the series
            
        Returns:
            Dictionary with series prediction results
        """
        # If maps not provided, use common maps
        if not maps:
            maps = self.common_maps[:3]  # Use first 3 maps
        elif len(maps) > 3:
            print("Warning: More than 3 maps provided. Using only first 3 for Bo3 series.")
            maps = maps[:3]
        
        # Predict each map
        map_predictions = []
        for map_name in maps:
            prediction = self.predict_match(team1_name, team2_name, map_name)
            map_predictions.append(prediction)
        
        # Calculate series win probability
        if len(maps) == 3:
            # Calculate probability of winning exactly 2 maps or all 3 maps
            p1 = map_predictions[0]['team1_win_probability']
            p2 = map_predictions[1]['team1_win_probability']
            p3 = map_predictions[2]['team1_win_probability']
            
            # Probability of winning 2 out of 3 maps
            series_win_prob = (
                p1 * p2 * (1-p3) +  # Win maps 1 & 2, lose map 3
                p1 * (1-p2) * p3 +  # Win maps 1 & 3, lose map 2
                (1-p1) * p2 * p3    # Lose map 1, win maps 2 & 3
            )
            
            # Add probability of winning all 3 maps
            series_win_prob += p1 * p2 * p3
        else:
            # Use average for other lengths
            series_win_prob = np.mean([p['team1_win_probability'] for p in map_predictions])
        
        # Determine expected series score
        if series_win_prob >= 0.5:
            # Team 1 favored to win
            team1_expected_maps = 2 if series_win_prob < 0.8 else 3
            team2_expected_maps = 3 - team1_expected_maps
        else:
            # Team 2 favored to win
            team2_expected_maps = 2 if (1 - series_win_prob) < 0.8 else 3
            team1_expected_maps = 3 - team2_expected_maps
        
        return {
            'maps': maps,
            'map_predictions': map_predictions,
            'team1_series_win_probability': float(series_win_prob),
            'team2_series_win_probability': 1 - float(series_win_prob),
            'team1_expected_maps': int(team1_expected_maps),
            'team2_expected_maps': int(team2_expected_maps),
            'expected_score': f"{team1_expected_maps}-{team2_expected_maps}"
        }
    
    def print_map_prediction(self, prediction, team1_name, team2_name):
        """
        Print prediction results for a single map in a human-readable format.
        
        Args:
            prediction: Prediction results from predict_match
            team1_name: Name of team 1
            team2_name: Name of team 2
        """
        map_name = prediction.get('map', 'unknown')
        
        print("\n" + "-"*60)
        print(f"MAP PREDICTION (PyTorch LSTM): {map_name}")
        print("-"*60)
        
        # Check if we have an error
        if 'error' in prediction:
            print(f"Error: {prediction['error']}")
            print(f"Default probability: {team1_name}: 50%, {team2_name}: 50%")
            return
        
        # Print K/D differential
        team1_kd = prediction['team1_stats'].get('kd_diff', 0)
        team2_kd = prediction['team2_stats'].get('kd_diff', 0)
        print(f"K/D Differential: {team1_name}: {team1_kd:.2f}, {team2_name}: {team2_kd:.2f}")
        
        # Print average win probabilities
        print(f"{team1_name} win probability: {prediction['team1_win_probability']*100:.1f}%")
        print(f"{team2_name} win probability: {prediction['team2_win_probability']*100:.1f}%")
        
        if prediction['team1_win'] is not None:
            print(f"Predicted winner: {team1_name if prediction['team1_win'] else team2_name}")
        else:
            print("Predicted winner: Unable to determine")
    
    def print_series_prediction(self, series_prediction, team1_name, team2_name):
        """
        Print prediction results for a Bo3 series in a human-readable format.
        
        Args:
            series_prediction: Prediction results from predict_series
            team1_name: Name of team 1
            team2_name: Name of team 2
        """
        print("\n" + "="*60)
        print(f"SERIES PREDICTION (PyTorch LSTM): {team1_name} vs {team2_name} (Best of 3)")
        print("="*60)
        
        # Print individual map predictions
        for map_prediction in series_prediction['map_predictions']:
            self.print_map_prediction(map_prediction, team1_name, team2_name)
        
        print("\n" + "="*60)
        print("OVERALL SERIES PREDICTION")
        print("="*60)
        
        # Print series win probabilities
        print(f"{team1_name} series win probability: {series_prediction['team1_series_win_probability']*100:.1f}%")
        print(f"{team2_name} series win probability: {series_prediction['team2_series_win_probability']*100:.1f}%")
        
        # Print expected score
        print(f"Expected score: {team1_name} {series_prediction['team1_expected_maps']} - {series_prediction['team2_expected_maps']} {team2_name}")
        
        # Print predicted winner
        if series_prediction['team1_series_win_probability'] >= 0.5:
            print(f"Predicted series winner: {team1_name}")
        else:
            print(f"Predicted series winner: {team2_name}")
        
        print("="*60)
    
    def print_detailed_prediction(self, prediction, team1_name, team2_name):
        """
        Print detailed prediction results for a single map.
        
        Args:
            prediction: Prediction results from predict_match
            team1_name: Name of team 1
            team2_name: Name of team 2
        """
        map_name = prediction.get('map', 'unknown')
        
        print("\n" + "="*60)
        print(f"DETAILED MAP PREDICTION (PyTorch LSTM): {team1_name} vs {team2_name} on {map_name}")
        print("="*60)
        
        # Check if we have an error
        if 'error' in prediction:
            print(f"Error: {prediction['error']}")
            print("Unable to provide detailed prediction due to insufficient data.")
            return
        
        print(f"\nTeam Statistics (Most Recent):")
        print("-"*60)
        print(f"{'Statistic':<20} {team1_name:<20} {team2_name:<20}")
        print("-"*60)
        
        stats = [
            ('Kills', 'kills'),
            ('Deaths', 'deaths'),
            ('K/D Diff', 'kd_diff'),
            ('Assists', 'assists'),
            ('ADR', 'adr'),
            ('Open Duel Ratio', 'open_duels_ratio'),
            ('Multi Kills', 'multi_kills')
        ]
        
        for stat_name, stat_key in stats:
            team1_val = prediction['team1_stats'][stat_key]
            team2_val = prediction['team2_stats'][stat_key]
            
            if stat_key == 'open_duels_ratio':
                print(f"{stat_name:<20} {team1_val*100:.1f}%{' ↑' if team1_val > team2_val else '':<20} {team2_val*100:.1f}%{' ↑' if team2_val > team1_val else '':<20}")
            else:
                print(f"{stat_name:<20} {team1_val:.1f}{' ↑' if team1_val > team2_val else '':<20} {team2_val:.1f}{' ↑' if team2_val > team1_val else '':<20}")
        
        # Print sequence trends if available
        if 'head_to_head_sequence' in prediction:
            print("\nSequence Trend (Last 5 Matches):")
            print("-"*60)
            
            for i, diff in enumerate(prediction['head_to_head_sequence']):
                match_num = i + 1
                kills_diff = diff['kills_diff']
                deaths_diff = diff['deaths_diff']
                adr_diff = diff['adr_diff']
                kd_diff = diff['kd_diff_diff']
                
                advantage = f"{team1_name} advantage" if (kills_diff + deaths_diff + adr_diff) > 0 else f"{team2_name} advantage"
                print(f"Match {match_num}: K/D Diff: {kd_diff:.2f}, ADR Diff: {adr_diff:.2f} - {advantage}")
        
        print("\nPrediction:")
        print("-"*60)
        print(f"{team1_name} win probability: {prediction['team1_win_probability']*100:.1f}%")
        print(f"{team2_name} win probability: {prediction['team2_win_probability']*100:.1f}%")
        
        if prediction['team1_win'] is not None:
            print(f"Predicted winner: {team1_name if prediction['team1_win'] else team2_name}")
        else:
            print("Predicted winner: Unable to determine")
        
        print("="*60)


def main():
    """
    Main function to run prediction from command line.
    """
    parser = argparse.ArgumentParser(description='Predict esports match outcomes using PyTorch LSTM model')
    parser.add_argument('--model', required=True, help='Path to saved PyTorch LSTM model weights')
    parser.add_argument('--model-info', required=True, help='Path to saved model architecture info')
    parser.add_argument('--scaler', required=True, help='Path to saved feature scaler')
    parser.add_argument('--features', required=True, help='Path to saved feature columns')
    parser.add_argument('--data', required=True, help='Path to historical match data (Excel file)')
    parser.add_argument('--team1', required=True, help='Name of team 1')
    parser.add_argument('--team2', required=True, help='Name of team 2')
    parser.add_argument('--maps', help='Comma-separated list of maps in the Bo3 series (max 3)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics for each map')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = LSTMMatchPredictor(args.model, args.model_info, args.scaler, args.features, args.data)
    
    # Parse maps
    maps = None
    if args.maps:
        maps = [map_name.strip() for map_name in args.maps.split(',')]
        
        # Limit to max 3 maps for a Bo3 series
        if len(maps) > 3:
            print("Warning: More than 3 maps provided. Using only the first 3 for Bo3 series.")
            maps = maps[:3]
    
    # Make series prediction
    series_prediction = predictor.predict_series(args.team1, args.team2, maps)
    
    # Print results
    predictor.print_series_prediction(series_prediction, args.team1, args.team2)
    
    # Print detailed statistics if requested
    if args.detailed:
        for map_prediction in series_prediction['map_predictions']:
            predictor.print_detailed_prediction(map_prediction, args.team1, args.team2)


if __name__ == "__main__":
    main()

# Example usage:
# python match_predictor_lstm.py --model output/esports_lstm_pytorch_model.pth --model-info output/lstm_model_info.pkl --scaler output/lstm_pytorch_scaler.pkl --features output/lstm_pytorch_features.pkl --data match_data.xlsx --team1 "Complexity" --team2 "BIG"