import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import os
import re
from datetime import datetime
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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

def prepare_sequence_data(df, sequence_length=5, step=1):
    """
    Prepare sequential data for LSTM.
    
    Args:
        df: Preprocessed DataFrame with match data
        sequence_length: Number of previous matches to include in each sequence
        step: Step size for creating sequences
        
    Returns:
        X_sequences, y_sequences: Sequences for training LSTM
    """
    # Get unique players and teams
    unique_players = df['player_name'].unique()
    unique_teams = df['team_name'].unique()
    
    # Initialize lists to store sequences
    X_sequences = []
    y_sequences = []
    match_ids = []
    
    # Create sequences for each player
    print("Creating player sequences...")
    player_stats = []
    
    for player in unique_players:
        # Get matches for this player in chronological order
        player_df = df[df['player_name'] == player].sort_values('match_order')
        
        if len(player_df) < sequence_length + 1:
            continue  # Skip players with too few matches
        
        # Process each match, calculate stats
        for i in range(len(player_df)):
            if i == 0:
                # First match, use actual stats
                stats = {
                    'player_name': player,
                    'match_id': player_df.iloc[i]['match_id'],
                    'match_order': player_df.iloc[i]['match_order'],
                    'team_name': player_df.iloc[i]['team_name'],
                    'map': player_df.iloc[i]['map'],
                    'kills': player_df.iloc[i]['kills'],
                    'deaths': player_df.iloc[i]['deaths'],
                    'assists': player_df.iloc[i]['assists'],
                    'adr': player_df.iloc[i]['adr'],
                    'open_duels_ratio': player_df.iloc[i]['open_duels_ratio'],
                    'multi_kills': player_df.iloc[i]['multi_kills'],
                    'kd_diff': player_df.iloc[i]['kd_diff'],
                    'player_won': player_df.iloc[i]['player_won']
                }
            else:
                # For subsequent matches, use past performance
                prev_matches = player_df.iloc[:i]
                
                # Simple moving average for recent performance (last 5 matches or all if fewer)
                recent_n = min(5, len(prev_matches))
                recent_matches = prev_matches.iloc[-recent_n:]
                
                stats = {
                    'player_name': player,
                    'match_id': player_df.iloc[i]['match_id'],
                    'match_order': player_df.iloc[i]['match_order'],
                    'team_name': player_df.iloc[i]['team_name'],
                    'map': player_df.iloc[i]['map'],
                    'kills': recent_matches['kills'].mean(),
                    'deaths': recent_matches['deaths'].mean(),
                    'assists': recent_matches['assists'].mean(),
                    'adr': recent_matches['adr'].mean(),
                    'open_duels_ratio': recent_matches['open_duels_ratio'].mean(),
                    'multi_kills': recent_matches['multi_kills'].mean(),
                    'kd_diff': recent_matches['kd_diff'].mean(),
                    'player_won': player_df.iloc[i]['player_won']
                }
            
            player_stats.append(stats)
    
    # Convert to DataFrame
    player_stats_df = pd.DataFrame(player_stats)
    
    # Create team-level data from player stats
    print("Creating team-level statistics...")
    team_match_stats = []
    
    # Get unique matches
    unique_matches = player_stats_df['match_id'].unique()
    
    for match_id in unique_matches:
        # Get player stats for this match
        match_df = player_stats_df[player_stats_df['match_id'] == match_id]
        
        # Get teams in this match
        teams = match_df['team_name'].unique()
        
        if len(teams) != 2:
            continue  # Skip matches without exactly 2 teams
        
        # For consistency in feature generation, sort team names alphabetically
        team1, team2 = sorted(teams)
        
        # Calculate team averages
        team1_df = match_df[match_df['team_name'] == team1]
        team2_df = match_df[match_df['team_name'] == team2]
        
        if len(team1_df) == 0 or len(team2_df) == 0:
            continue  # Skip if either team has no players
        
        # Team 1 stats
        team1_stats = {
            'match_id': match_id,
            'match_order': team1_df['match_order'].iloc[0],
            'team1': team1,
            'team2': team2,
            'map': team1_df['map'].iloc[0],
            'team1_kills': team1_df['kills'].mean(),
            'team1_deaths': team1_df['deaths'].mean(),
            'team1_assists': team1_df['assists'].mean(),
            'team1_adr': team1_df['adr'].mean(),
            'team1_open_duels_ratio': team1_df['open_duels_ratio'].mean(),
            'team1_multi_kills': team1_df['multi_kills'].mean(),
            'team1_kd_diff': team1_df['kd_diff'].mean(),
            'team1_won': 1 if team1_df['player_won'].iloc[0] == 1 else 0
        }
        
        # Team 2 stats
        team2_stats = {
            'team2_kills': team2_df['kills'].mean(),
            'team2_deaths': team2_df['deaths'].mean(),
            'team2_assists': team2_df['assists'].mean(),
            'team2_adr': team2_df['adr'].mean(),
            'team2_open_duels_ratio': team2_df['open_duels_ratio'].mean(),
            'team2_multi_kills': team2_df['multi_kills'].mean(),
            'team2_kd_diff': team2_df['kd_diff'].mean()
        }
        
        # Combine stats
        match_stats = {**team1_stats, **team2_stats}
        team_match_stats.append(match_stats)
    
    # Convert to DataFrame and sort by match order
    team_stats_df = pd.DataFrame(team_match_stats).sort_values('match_order')
    
    # Feature columns for sequences
    feature_columns = [
        'team1_kills', 'team1_deaths', 'team1_assists', 'team1_adr',
        'team1_open_duels_ratio', 'team1_multi_kills', 'team1_kd_diff',
        'team2_kills', 'team2_deaths', 'team2_assists', 'team2_adr',
        'team2_open_duels_ratio', 'team2_multi_kills', 'team2_kd_diff'
    ]
    
    # Calculate feature diffs for more robust modeling
    team_stats_df['kills_diff'] = team_stats_df['team1_kills'] - team_stats_df['team2_kills']
    team_stats_df['deaths_diff'] = team_stats_df['team2_deaths'] - team_stats_df['team1_deaths']  # Reversed for consistency
    team_stats_df['assists_diff'] = team_stats_df['team1_assists'] - team_stats_df['team2_assists']
    team_stats_df['adr_diff'] = team_stats_df['team1_adr'] - team_stats_df['team2_adr']
    team_stats_df['open_duels_ratio_diff'] = team_stats_df['team1_open_duels_ratio'] - team_stats_df['team2_open_duels_ratio']
    team_stats_df['multi_kills_diff'] = team_stats_df['team1_multi_kills'] - team_stats_df['team2_multi_kills']
    team_stats_df['kd_diff_diff'] = team_stats_df['team1_kd_diff'] - team_stats_df['team2_kd_diff']
    
    # Add feature diffs for sequence modeling
    feature_columns += [
        'kills_diff', 'deaths_diff', 'assists_diff', 'adr_diff',
        'open_duels_ratio_diff', 'multi_kills_diff', 'kd_diff_diff'
    ]
    
    # Store feature columns for later use
    feature_columns_diff_only = [
        'kills_diff', 'deaths_diff', 'assists_diff', 'adr_diff',
        'open_duels_ratio_diff', 'multi_kills_diff', 'kd_diff_diff'
    ]
    
    # Create team sequences by match
    print("Creating team sequences...")
    
    # Group by team pairs to create sequences
    team_pairs = set()
    for _, row in team_stats_df.iterrows():
        # Keep consistent ordering for team pairs
        if row['team1'] < row['team2']:
            team_pairs.add((row['team1'], row['team2']))
        else:
            team_pairs.add((row['team2'], row['team1']))
    
    # Create sequences for each team pair
    for team1, team2 in team_pairs:
        # Get matches between these teams, in order
        pair_matches = team_stats_df[
            ((team_stats_df['team1'] == team1) & (team_stats_df['team2'] == team2)) |
            ((team_stats_df['team1'] == team2) & (team_stats_df['team2'] == team1))
        ].sort_values('match_order')
        
        # Ensure consistent team ordering in features by swapping when needed
        for i, (_, row) in enumerate(pair_matches.iterrows()):
            if row['team1'] == team2 and row['team2'] == team1:
                # Swap team stats for consistency
                for col in ['kills', 'deaths', 'assists', 'adr', 'open_duels_ratio', 'multi_kills', 'kd_diff']:
                    team1_col = f'team1_{col}'
                    team2_col = f'team2_{col}'
                    pair_matches.at[_, team1_col], pair_matches.at[_, team2_col] = \
                        row[team2_col], row[team1_col]
                
                # Invert the diffs
                for col in ['kills_diff', 'assists_diff', 'adr_diff', 'open_duels_ratio_diff', 'multi_kills_diff', 'kd_diff_diff']:
                    pair_matches.at[_, col] = -row[col]
                
                # Special handling for deaths_diff which is already inverted
                pair_matches.at[_, 'deaths_diff'] = -row['deaths_diff']
                
                # Update team order and result
                pair_matches.at[_, 'team1'] = team1
                pair_matches.at[_, 'team2'] = team2
                pair_matches.at[_, 'team1_won'] = 1 - row['team1_won']  # Invert the outcome
        
        # Create sequences
        if len(pair_matches) >= sequence_length + 1:
            for i in range(len(pair_matches) - sequence_length):
                # Get sequence of matches
                seq = pair_matches.iloc[i:i+sequence_length]
                next_match = pair_matches.iloc[i+sequence_length]
                
                # Extract features and target
                X_seq = seq[feature_columns_diff_only].values
                y_target = next_match['team1_won']
                
                X_sequences.append(X_seq)
                y_sequences.append(y_target)
                match_ids.append(next_match['match_id'])
    
    # Also create sequences for each individual team's performance over time
    print("Creating individual team sequences...")
    for team in unique_teams:
        # Get matches for this team (either as team1 or team2)
        team_matches = team_stats_df[
            (team_stats_df['team1'] == team) | (team_stats_df['team2'] == team)
        ].sort_values('match_order')
        
        # Ensure team is always in team1 position for consistent feature extraction
        for i, (_, row) in enumerate(team_matches.iterrows()):
            if row['team2'] == team:  # If team is in team2 position, swap
                # Swap team stats
                for col in ['kills', 'deaths', 'assists', 'adr', 'open_duels_ratio', 'multi_kills', 'kd_diff']:
                    team1_col = f'team1_{col}'
                    team2_col = f'team2_{col}'
                    team_matches.at[_, team1_col], team_matches.at[_, team2_col] = \
                        row[team2_col], row[team1_col]
                
                # Invert the diffs
                for col in ['kills_diff', 'assists_diff', 'adr_diff', 'open_duels_ratio_diff', 'multi_kills_diff', 'kd_diff_diff']:
                    team_matches.at[_, col] = -row[col]
                
                # Special handling for deaths_diff which is already inverted
                team_matches.at[_, 'deaths_diff'] = -row['deaths_diff']
                
                # Swap teams and invert result
                team_matches.at[_, 'team1'] = team
                team_matches.at[_, 'team2'] = row['team1']
                team_matches.at[_, 'team1_won'] = 1 - row['team1_won']
        
        # Create sequences
        if len(team_matches) >= sequence_length + 1:
            for i in range(0, len(team_matches) - sequence_length, step):
                # Get sequence of matches
                seq = team_matches.iloc[i:i+sequence_length]
                next_match = team_matches.iloc[i+sequence_length]
                
                # Extract features and target
                X_seq = seq[feature_columns_diff_only].values
                y_target = next_match['team1_won']
                
                X_sequences.append(X_seq)
                y_sequences.append(y_target)
                match_ids.append(next_match['match_id'])
    
    # Convert to numpy arrays
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
    
    # Return the sequences and match IDs
    return X_sequences, y_sequences, match_ids, feature_columns_diff_only

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

def split_data(X, y, match_ids, test_size=0.2, validation_size=0.1, random_seed=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature array
        y: Target array
        match_ids: List of match IDs
        test_size: Proportion to use for testing
        validation_size: Proportion to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Train, validation, and test data
    """
    np.random.seed(random_seed)
    
    # Calculate split indices
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_split = int(n_samples * (1 - test_size))
    val_split = int(n_samples * (1 - test_size - validation_size))
    
    # Create splits
    train_indices = indices[:val_split]
    val_indices = indices[val_split:test_split]
    test_indices = indices[test_split:]
    
    # Split data
    X_train = X[train_indices]
    y_train = y[train_indices]
    match_ids_train = [match_ids[i] for i in train_indices]
    
    X_val = X[val_indices]
    y_val = y[val_indices]
    match_ids_val = [match_ids[i] for i in val_indices]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    match_ids_test = [match_ids[i] for i in test_indices]
    
    return (X_train, y_train, match_ids_train), (X_val, y_val, match_ids_val), (X_test, y_test, match_ids_test)

def train_lstm_model(X_train, y_train, X_val, y_val, input_size, 
                    hidden_size=64, num_layers=2, dropout=0.2, 
                    lr=0.001, epochs=50, batch_size=32, device=None):
    """
    Train the LSTM model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_size: Size of input features
        hidden_size: Size of LSTM hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on (CPU or GPU)
        
    Returns:
        Trained model and training history
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create TensorDataset and DataLoader
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                preds = (outputs > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_lstm_model(model, X_test, y_test, device=None):
    """
    Evaluate the LSTM model.
    
    Args:
        model: Trained LSTM model
        X_test, y_test: Test data
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_prob = model(X_test_tensor).cpu().numpy().flatten()
    
    # Convert probabilities to predictions
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Return evaluation results
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def plot_training_history(history, output_dir='output'):
    """
    Plot training history.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lstm_pytorch_training_history.png")
    plt.close()

def save_lstm_model(model, X_scaler, feature_columns, sequence_length, output_dir='output'):
    """
    Save the LSTM model and related components.
    
    Args:
        model: Trained LSTM model
        X_scaler: Feature scaler
        feature_columns: Feature column names
        sequence_length: Length of sequences
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'esports_lstm_pytorch_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save model architecture
    model_info = {
        'input_size': model.lstm1.input_size,
        'hidden_size': model.lstm1.hidden_size,
        'num_layers': 2,
        'dropout': model.dropout1.p,
        'sequence_length': sequence_length
    }
    model_info_path = os.path.join(output_dir, 'lstm_model_info.pkl')
    joblib.dump(model_info, model_info_path)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'lstm_pytorch_scaler.pkl')
    joblib.dump(X_scaler, scaler_path)
    
    # Save feature columns
    feature_path = os.path.join(output_dir, 'lstm_pytorch_features.pkl')
    joblib.dump(feature_columns, feature_path)
    
    print(f"Model saved to {model_path}")
    print(f"Model info saved to {model_info_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Feature columns saved to {feature_path}")

def main(file_path, output_dir='output', sequence_length=5, epochs=50, batch_size=32):
    """
    Main function to run the LSTM analysis.
    
    Args:
        file_path: Path to the Excel file with match data
        output_dir: Directory to save model and results
        sequence_length: Length of sequences for LSTM
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with results
    """
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)
    
    print("\nPreparing sequence data...")
    X_sequences, y_sequences, match_ids, feature_columns = prepare_sequence_data(df, sequence_length)
    
    # Scale features (fit on X_sequences)
    X_scaler = MinMaxScaler()
    
    # Reshape for scaling (combine time steps and samples)
    X_flat = X_sequences.reshape(-1, X_sequences.shape[2])
    X_flat_scaled = X_scaler.fit_transform(X_flat)
    
    # Reshape back to sequences
    X_sequences_scaled = X_flat_scaled.reshape(X_sequences.shape)
    
    print(f"\nFeature columns: {feature_columns}")
    print(f"X shape after scaling: {X_sequences_scaled.shape}")
    print(f"y shape: {y_sequences.shape}")
    
    # Split data
    print("\nSplitting data...")
    (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, y_test, ids_test) = split_data(
        X_sequences_scaled, y_sequences, match_ids
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    print("\nTraining LSTM model...")
    input_size = X_train.shape[2]  # Number of features
    model, history = train_lstm_model(
        X_train, y_train, 
        X_val, y_val, 
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        lr=0.001,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation = evaluate_lstm_model(model, X_test, y_test, device)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Save model
    save_lstm_model(model, X_scaler, feature_columns, sequence_length, output_dir)
    
    return {
        'model': model,
        'scaler': X_scaler,
        'feature_columns': feature_columns,
        'evaluation': evaluation,
        'history': history,
        'sequence_length': sequence_length
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PyTorch LSTM esports prediction model')
    parser.add_argument('--data', required=True, help='Path to the Excel file with match data')
    parser.add_argument('--output', default='output', help='Directory to save model and results')
    parser.add_argument('--seq-length', type=int, default=5, help='Sequence length for LSTM')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    results = main(args.data, args.output, args.seq_length, args.epochs, args.batch_size)