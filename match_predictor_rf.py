import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
import warnings
import itertools

# Suppress specific warnings that might occur during data processing
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class RFMatchPredictor:
    """
    Utility class for predicting esports match outcomes using the Random Forest model.
    """
    
    def __init__(self, model_path, data_path=None):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved Random Forest model file
            data_path: Optional path to historical data for player statistics
        """
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_columns = self.model_data['feature_columns']
        
        # Common maps in competitive play (sorted by popularity)
        self.common_maps = [
            'dust2', 'mirage', 'inferno', 'nuke', 'overpass', 
            'ancient', 'vertigo', 'anubis'
        ]
        
        self.player_stats = {}
        self.match_maps = {}  # Store map played for each match
        
        # Load historical data if provided
        if data_path:
            self.load_historical_data(data_path)
    
    def load_historical_data(self, data_path):
        """
        Load historical match data to calculate player statistics.
        
        Args:
            data_path: Path to Excel file with historical match data
        """
        print(f"Loading historical data from {data_path}...")
        df = pd.read_excel(data_path)
        
        # Store map information for each match
        for _, row in df.iterrows():
            if pd.notna(row['match_id']) and pd.notna(row['map']):
                self.match_maps[row['match_id']] = row['map']
        
        # Extract player statistics
        for _, row in df.iterrows():
            player_name = row['player_name']
            
            if player_name not in self.player_stats:
                self.player_stats[player_name] = {
                    'kills': [],
                    'deaths': [],
                    'assists': [],
                    'adr': [],
                    'open_duels_won': [],
                    'open_duels_total': [],
                    'multi_kills': [],
                    'kd_diff': [],  # Add K/D differential
                    'wins': [],
                    'matches': [],
                    'maps': []  # Store map for each match
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
                
                # Calculate K/D differential
                kd_diff = kills - deaths
                
                self.player_stats[player_name]['kills'].append(kills)
                self.player_stats[player_name]['deaths'].append(deaths)
                self.player_stats[player_name]['assists'].append(assists)
                self.player_stats[player_name]['adr'].append(adr)
                self.player_stats[player_name]['open_duels_won'].append(open_duels_won)
                self.player_stats[player_name]['open_duels_total'].append(open_duels_total)
                self.player_stats[player_name]['multi_kills'].append(multi_kills)
                self.player_stats[player_name]['kd_diff'].append(kd_diff)  # Store K/D differential
                self.player_stats[player_name]['wins'].append(player_won)
                self.player_stats[player_name]['matches'].append(row['match_id'])
                self.player_stats[player_name]['maps'].append(map_name)
            except (ValueError, TypeError):
                print(f"Warning: Error parsing stats for player {player_name} in match {row['match_id']}. Using default values.")
        
        print(f"Loaded data for {len(self.player_stats)} players")
    
    def get_player_stats(self, player_name, map_filter=None, recent_n=None):
        """
        Get average statistics for a player.
        
        Args:
            player_name: Name of the player
            map_filter: If provided, filter statistics by specific map
            recent_n: If provided, use only the N most recent matches
            
        Returns:
            Dictionary with player statistics
        """
        if player_name not in self.player_stats:
            print(f"Warning: No historical data for player {player_name}. Using default values.")
            return {
                'avg_kills': 15.0,
                'avg_deaths': 15.0,
                'avg_assists': 3.0,
                'avg_adr': 70.0,
                'avg_open_duels_ratio': 0.5,
                'avg_multi_kills': 3.0,
                'avg_kd_diff': 0.0,  # Default K/D differential
                'win_rate': 0.5,
                'matches_played': 0
            }
        
        player_data = self.player_stats[player_name]
        
        # Create a mask for map filtering if needed
        map_mask = np.ones(len(player_data['matches']), dtype=bool)
        if map_filter is not None:
            map_mask = np.array([
                player_data['maps'][i].lower() == map_filter.lower() 
                for i in range(len(player_data['maps']))
            ])
            # If no matches on this map, use all maps
            if not any(map_mask):
                print(f"No data for {player_name} on map {map_filter}. Using all maps.")
                map_mask = np.ones(len(player_data['matches']), dtype=bool)
        
        # Apply map filter
        filtered_indices = np.where(map_mask)[0]
        
        # If recent_n is provided, use only the most recent matches after map filtering
        if recent_n is not None and recent_n < len(filtered_indices):
            sorted_indices = sorted(filtered_indices, reverse=True)  # Assumes more recent matches are at the end
            filtered_indices = sorted_indices[:recent_n]
        
        # Get filtered data
        kills = [player_data['kills'][i] for i in filtered_indices]
        deaths = [player_data['deaths'][i] for i in filtered_indices]
        assists = [player_data['assists'][i] for i in filtered_indices]
        adr = [player_data['adr'][i] for i in filtered_indices]
        open_duels_won = [player_data['open_duels_won'][i] for i in filtered_indices]
        open_duels_total = [player_data['open_duels_total'][i] for i in filtered_indices]
        multi_kills = [player_data['multi_kills'][i] for i in filtered_indices]
        kd_diff = [player_data['kd_diff'][i] for i in filtered_indices]  # Get K/D differential
        wins = [player_data['wins'][i] for i in filtered_indices]
        
        # Calculate averages
        avg_kills = np.mean(kills) if kills else 15.0
        avg_deaths = np.mean(deaths) if deaths else 15.0
        avg_assists = np.mean(assists) if assists else 3.0
        avg_adr = np.mean(adr) if adr else 70.0
        avg_open_duels_ratio = np.sum(open_duels_won) / np.sum(open_duels_total) if np.sum(open_duels_total) > 0 else 0.5
        avg_multi_kills = np.mean(multi_kills) if multi_kills else 3.0
        avg_kd_diff = np.mean(kd_diff) if kd_diff else 0.0  # Calculate average K/D differential
        
        # Calculate win rate - simple average
        win_rate = np.mean(wins) if wins else 0.5
        
        # Number of matches played with this filter
        matches_played = len(kills)
        
        return {
            'avg_kills': avg_kills,
            'avg_deaths': avg_deaths,
            'avg_assists': avg_assists,
            'avg_adr': avg_adr,
            'avg_open_duels_ratio': avg_open_duels_ratio,
            'avg_multi_kills': avg_multi_kills,
            'avg_kd_diff': avg_kd_diff,  # Include K/D differential
            'win_rate': win_rate,
            'matches_played': matches_played
        }
    
    def get_team_stats(self, players, map_filter=None, recent_n=None):
        """
        Calculate average statistics for a team based on player statistics.
        
        Args:
            players: List of player names in the team
            map_filter: If provided, filter statistics by specific map
            recent_n: If provided, use only the N most recent matches
            
        Returns:
            Dictionary with team statistics
        """
        team_stats = {
            'avg_kills': 0,
            'avg_deaths': 0,
            'avg_assists': 0,
            'avg_adr': 0,
            'avg_open_duels_ratio': 0,
            'avg_multi_kills': 0,
            'avg_kd_diff': 0,  # Add K/D differential
            'win_rate': 0,
            'total_matches': 0
        }
        
        # Collect player stats
        player_stats_list = []
        for player in players:
            player_stats = self.get_player_stats(player, map_filter, recent_n)
            player_stats_list.append(player_stats)
            
            # Sum up most stats (excluding win_rate which needs special handling)
            for stat in ['avg_kills', 'avg_deaths', 'avg_assists', 'avg_adr', 
                         'avg_open_duels_ratio', 'avg_multi_kills', 'avg_kd_diff']:
                team_stats[stat] += player_stats[stat]
            
            # Add to total matches
            team_stats['total_matches'] += player_stats['matches_played']
        
        # Calculate simple averages for most stats
        num_players = len(players)
        for stat in ['avg_kills', 'avg_deaths', 'avg_assists', 'avg_adr', 
                      'avg_open_duels_ratio', 'avg_multi_kills', 'avg_kd_diff']:
            team_stats[stat] /= num_players
        
        # Calculate weighted win rate based on matches played
        if team_stats['total_matches'] > 0:
            weighted_win_rate = 0
            for player_stats in player_stats_list:
                # Weight by matches played (players with more matches have more influence)
                weight = player_stats['matches_played'] / team_stats['total_matches'] if team_stats['total_matches'] > 0 else 0
                weighted_win_rate += player_stats['win_rate'] * weight
            
            team_stats['win_rate'] = weighted_win_rate
        else:
            # Default win rate if no matches
            team_stats['win_rate'] = 0.5
        
        return team_stats
    
    def _calculate_directional_prediction(self, team1_stats, team2_stats, team_order):
        """
        Helper method to calculate prediction in one direction (team1 vs team2).
        This is a private method not meant to be called directly.
        
        Args:
            team1_stats: Statistics for team 1
            team2_stats: Statistics for team 2
            team_order: Identifier for which order teams are being predicted in
            
        Returns:
            Dictionary with prediction results
        """
        # Map feature columns to team stats keys
        feature_mapping = {
            'kills_diff': 'avg_kills',
            'deaths_diff': 'avg_deaths',  # This will be inverted below
            'kd_diff_diff': 'avg_kd_diff',  # K/D differential
            'assists_diff': 'avg_assists',
            'adr_diff': 'avg_adr',
            'open_duels_ratio_diff': 'avg_open_duels_ratio',
            'multi_kills_diff': 'avg_multi_kills',
            'win_rate_diff': 'win_rate'
        }
        
        # Calculate feature differences (team1 - team2)
        features = {}
        for feature in self.feature_columns:
            if feature in feature_mapping:
                stat_key = feature_mapping[feature]
                
                # Special handling for deaths: invert the difference for consistency
                if feature == 'deaths_diff':
                    # For deaths, LOWER is better, so we invert the difference
                    features[feature] = team2_stats[stat_key] - team1_stats[stat_key]
                else:
                    # For all other stats, HIGHER is better
                    features[feature] = team1_stats[stat_key] - team2_stats[stat_key]
            else:
                # If feature not found in mapping, use default of 0
                features[feature] = 0
        
        # Prepare input for prediction
        X = np.array([[features[col] for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        team1_win_prob = self.model.predict_proba(X_scaled)[0, 1]
        team1_win = self.model.predict(X_scaled)[0]
        
        return {
            'team_order': team_order,
            'team1_win_probability': team1_win_prob,
            'team1_win': bool(team1_win),
            'team2_win_probability': 1 - team1_win_prob,
            'team2_win': not bool(team1_win),
            'features': features
        }
    
    def predict_map(self, teamA_players, teamB_players, map_name=None, recent_n=None):
        """
        Predict the outcome of a match on a specific map using a symmetric approach.
        This method ensures that swapping team order doesn't affect the prediction result.
        
        Args:
            teamA_players: List of player names in team A
            teamB_players: List of player names in team B
            map_name: If provided, filter statistics by specific map
            recent_n: If provided, use only the N most recent matches
            
        Returns:
            Dictionary with prediction results
        """
        # Get team statistics
        teamA_stats = self.get_team_stats(teamA_players, map_name, recent_n)
        teamB_stats = self.get_team_stats(teamB_players, map_name, recent_n)
        
        # First, predict with teamA as team1 and teamB as team2
        pred_AB = self._calculate_directional_prediction(teamA_stats, teamB_stats, "A_vs_B")
        
        # Then, predict with teamB as team1 and teamA as team2
        pred_BA = self._calculate_directional_prediction(teamB_stats, teamA_stats, "B_vs_A")
        
        # Average the predictions (for swapped prediction, we need to invert the result)
        teamA_win_prob = (pred_AB['team1_win_probability'] + (1 - pred_BA['team1_win_probability'])) / 2
        teamA_win = teamA_win_prob >= 0.5
        
        # Create symmetric prediction result
        return {
            'map': map_name,
            'teamA_win_probability': teamA_win_prob,
            'teamA_win': teamA_win,
            'teamB_win_probability': 1 - teamA_win_prob,
            'teamB_win': not teamA_win,
            'teamA_stats': teamA_stats,
            'teamB_stats': teamB_stats,
            # For diagnostic purposes, include both directional predictions
            'pred_AB': pred_AB,
            'pred_BA': pred_BA
        }
    
    def predict_series(self, teamA_players, teamB_players, maps=None, recent_n=None):
        """
        Predict the outcome of a best-of-3 series.
        
        Args:
            teamA_players: List of player names in team A
            teamB_players: List of player names in team B
            maps: List of maps in the series (if None, use all common maps)
            recent_n: If provided, use only the N most recent matches
            
        Returns:
            Dictionary with series prediction results
        """
        # If maps not provided, use all common maps
        if maps is None or len(maps) == 0:
            maps_to_predict = self.common_maps
        else:
            maps_to_predict = maps
        
        # Predict each map
        map_predictions = []
        for map_name in maps_to_predict:
            map_prediction = self.predict_map(teamA_players, teamB_players, map_name, recent_n)
            map_predictions.append(map_prediction)
        
        # Calculate series win probability
        # For Bo3, teamA wins if they win 2 maps
        if len(maps_to_predict) == 3:
            # Calculate probability of winning exactly 2 maps or all 3 maps
            p1 = map_predictions[0]['teamA_win_probability']
            p2 = map_predictions[1]['teamA_win_probability']
            p3 = map_predictions[2]['teamA_win_probability']
            
            # Probability of winning 2 out of 3 maps
            series_win_prob = (
                p1 * p2 * (1-p3) +  # Win maps 1 & 2, lose map 3
                p1 * (1-p2) * p3 +  # Win maps 1 & 3, lose map 2
                (1-p1) * p2 * p3    # Lose map 1, win maps 2 & 3
            )
            
            # Add probability of winning all 3 maps
            series_win_prob += p1 * p2 * p3
        else:
            # If not exactly 3 maps, use average win probability as approximation
            series_win_prob = np.mean([pred['teamA_win_probability'] for pred in map_predictions])
        
        # Determine expected series score
        if series_win_prob > 0.5:
            # Team A favored to win
            teamA_expected_maps = max(min(round(series_win_prob * 3), 3), 2)
            teamB_expected_maps = 3 - teamA_expected_maps
        else:
            # Team B favored to win
            teamB_expected_maps = max(min(round((1-series_win_prob) * 3), 3), 2)
            teamA_expected_maps = 3 - teamB_expected_maps
            
        return {
            'maps': maps_to_predict,
            'map_predictions': map_predictions,
            'teamA_series_win_probability': series_win_prob,
            'teamB_series_win_probability': 1 - series_win_prob,
            'teamA_expected_maps': teamA_expected_maps,
            'teamB_expected_maps': teamB_expected_maps,
            'expected_score': f"{teamA_expected_maps}-{teamB_expected_maps}"
        }
    
    def print_map_prediction(self, prediction, teamA_name, teamB_name):
        """
        Print prediction results for a single map in a human-readable format.
        
        Args:
            prediction: Prediction results from predict_map
            teamA_name: Name of team A
            teamB_name: Name of team B
        """
        map_name = prediction['map']
        
        print("\n" + "-"*60)
        print(f"MAP PREDICTION (Random Forest): {map_name}")
        print("-"*60)
        
        # Print match count information
        teamA_matches = prediction['teamA_stats'].get('total_matches', 0)
        teamB_matches = prediction['teamB_stats'].get('total_matches', 0)
        print(f"Matches analyzed: {teamA_name}: {teamA_matches}, {teamB_name}: {teamB_matches}")
        
        # Display K/D differential
        teamA_kd = prediction['teamA_stats'].get('avg_kd_diff', 0)
        teamB_kd = prediction['teamB_stats'].get('avg_kd_diff', 0)
        print(f"K/D Differential: {teamA_name}: {teamA_kd:.2f}, {teamB_name}: {teamB_kd:.2f}")
        
        # Print win probabilities
        print(f"{teamA_name} win probability: {prediction['teamA_win_probability']*100:.1f}%")
        print(f"{teamB_name} win probability: {prediction['teamB_win_probability']*100:.1f}%")
        print(f"Predicted winner: {teamA_name if prediction['teamA_win'] else teamB_name}")
        
        # Confidence level based on total matches analyzed
        total_matches = teamA_matches + teamB_matches
        confidence = "High" if total_matches > 50 else "Medium" if total_matches > 20 else "Low"
        print(f"Confidence: {confidence}")
    
    def print_series_prediction(self, series_prediction, teamA_name, teamB_name):
        """
        Print prediction results for a Bo3 series in a human-readable format.
        
        Args:
            series_prediction: Prediction results from predict_series
            teamA_name: Name of team A
            teamB_name: Name of team B
        """
        print("\n" + "="*60)
        print(f"SERIES PREDICTION (Random Forest): {teamA_name} vs {teamB_name} (Best of 3)")
        print("="*60)
        
        # Print individual map predictions
        for map_prediction in series_prediction['map_predictions']:
            self.print_map_prediction(map_prediction, teamA_name, teamB_name)
        
        print("\n" + "="*60)
        print("OVERALL SERIES PREDICTION")
        print("="*60)
        
        # Print series win probabilities
        print(f"{teamA_name} series win probability: {series_prediction['teamA_series_win_probability']*100:.1f}%")
        print(f"{teamB_name} series win probability: {series_prediction['teamB_series_win_probability']*100:.1f}%")
        
        # Print expected score
        print(f"Expected score: {teamA_name} {series_prediction['teamA_expected_maps']} - {series_prediction['teamB_expected_maps']} {teamB_name}")
        
        # Print predicted winner
        if series_prediction['teamA_series_win_probability'] > 0.5:
            print(f"Predicted series winner: {teamA_name}")
        else:
            print(f"Predicted series winner: {teamB_name}")
        
        print("="*60)
    
    def print_detailed_prediction(self, prediction, teamA_name, teamB_name):
        """
        Print detailed prediction results for a single map.
        
        Args:
            prediction: Prediction results from predict_map
            teamA_name: Name of team A
            teamB_name: Name of team B
        """
        map_name = prediction['map']
        
        print("\n" + "="*60)
        print(f"DETAILED MAP PREDICTION (Random Forest): {teamA_name} vs {teamB_name} on {map_name}")
        print("="*60)
        
        print(f"\nTeam Statistics:")
        print("-"*60)
        print(f"{'Statistic':<20} {teamA_name:<20} {teamB_name:<20}")
        print("-"*60)
        
        stats = [
            ('Avg Kills', 'avg_kills'),
            ('Avg Deaths', 'avg_deaths'),
            ('Avg K/D Diff', 'avg_kd_diff'),  # Add K/D differential
            ('Avg Assists', 'avg_assists'),
            ('Avg ADR', 'avg_adr'),
            ('Avg Open Duel Ratio', 'avg_open_duels_ratio'),
            ('Avg Multi Kills', 'avg_multi_kills'),
            ('Win Rate', 'win_rate')
        ]
        
        for stat_name, stat_key in stats:
            teamA_val = prediction['teamA_stats'][stat_key]
            teamB_val = prediction['teamB_stats'][stat_key]
            
            if stat_key == 'win_rate' or stat_key == 'avg_open_duels_ratio':
                print(f"{stat_name:<20} {teamA_val*100:.1f}%{' ↑' if teamA_val > teamB_val else '':<20} {teamB_val*100:.1f}%{' ↑' if teamB_val > teamA_val else '':<20}")
            else:
                print(f"{stat_name:<20} {teamA_val:.1f}{' ↑' if teamA_val > teamB_val else '':<20} {teamB_val:.1f}{' ↑' if teamB_val > teamA_val else '':<20}")
        
        # Print match count information
        teamA_matches = prediction['teamA_stats'].get('total_matches', 0)
        teamB_matches = prediction['teamB_stats'].get('total_matches', 0)
        print(f"{'Matches Analyzed':<20} {teamA_matches:<20} {teamB_matches:<20}")

        # Display original predictions for transparency
        print("\nSymmetric Prediction Details:")
        print("-"*60)
        print(f"Prediction ({teamA_name} as team1): {prediction['pred_AB']['team1_win_probability']*100:.1f}% win")
        print(f"Prediction ({teamB_name} as team1): {prediction['pred_BA']['team1_win_probability']*100:.1f}% win")
        print(f"Resulting symmetric prediction: {prediction['teamA_win_probability']*100:.1f}% win for {teamA_name}")

        print("\nPrediction:")
        print("-"*60)
        print(f"{teamA_name} win probability: {prediction['teamA_win_probability']*100:.1f}%")
        print(f"{teamB_name} win probability: {prediction['teamB_win_probability']*100:.1f}%")
        print(f"Predicted winner: {teamA_name if prediction['teamA_win'] else teamB_name}")
        
        # Confidence level based on total matches analyzed
        total_matches = teamA_matches + teamB_matches
        confidence = "High" if total_matches > 50 else "Medium" if total_matches > 20 else "Low"
        print(f"Confidence: {confidence} (based on {total_matches} matches analyzed)")
        
        print("="*60)


def main():
    """
    Main function to run the prediction from command line.
    """
    parser = argparse.ArgumentParser(description='Predict esports match outcomes with Random Forest model')
    parser.add_argument('--model', required=True, help='Path to saved Random Forest model file')
    parser.add_argument('--data', help='Path to historical match data (Excel file)')
    parser.add_argument('--team1', required=True, help='Name of team 1')
    parser.add_argument('--team2', required=True, help='Name of team 2')
    parser.add_argument('--team1-players', required=True, help='Comma-separated list of players in team 1')
    parser.add_argument('--team2-players', required=True, help='Comma-separated list of players in team 2')
    parser.add_argument('--maps', help='Comma-separated list of maps in the Bo3 series (max 3)')
    parser.add_argument('--recent', type=int, help='Use only N most recent matches for each player')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics for each map')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RFMatchPredictor(args.model, args.data)
    
    # Parse player lists
    team1_players = [player.strip() for player in args.team1_players.split(',')]
    team2_players = [player.strip() for player in args.team2_players.split(',')]
    
    # Parse maps
    maps = None
    if args.maps:
        maps = [map_name.strip() for map_name in args.maps.split(',')]
        
        # Limit to max 3 maps for a Bo3 series
        if len(maps) > 3:
            print("Warning: More than 3 maps provided. Using only the first 3 for Bo3 series.")
            maps = maps[:3]
    
    # Make series prediction
    series_prediction = predictor.predict_series(team1_players, team2_players, maps, args.recent)
    
    # Print results
    predictor.print_series_prediction(series_prediction, args.team1, args.team2)
    
    # Print detailed statistics if requested
    if args.detailed:
        for map_prediction in series_prediction['map_predictions']:
            predictor.print_detailed_prediction(map_prediction, args.team1, args.team2)


if __name__ == "__main__":
    main()

# Example usage:
# python match_predictor_rf.py --model output/esports_prediction_rf_model.pkl --data match_data.xlsx --team1 "Complexity" --team2 "BIG" --team1-players "floppy,JT,Grim,Hallzerk,FaNg" --team2-players "tabseN,faveN,k1to,Krimbo,hyped" --maps "inferno,nuke,mirage" --recent 10 --detailed