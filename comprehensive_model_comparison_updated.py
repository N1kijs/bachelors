import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import torch
import joblib
import os
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import model components
from esports_prediction_symmetric import load_and_preprocess_data, calculate_player_stats, prepare_match_features
from esports_prediction_random_forest import prepare_match_features as prepare_match_features_rf
from esports_prediction_lstm import prepare_sequence_data, LSTMModel

class ModelComparison:
    """
    Class for comprehensive comparison of esports prediction models.
    """
    
    def __init__(self, data_path, lr_model_path=None, rf_model_path=None, 
                 lstm_model_path=None, lstm_model_info_path=None,
                 lstm_scaler_path=None, lstm_features_path=None, 
                 output_dir='comparison'):
        """
        Initialize the model comparison.
        
        Args:
            data_path: Path to match data file
            lr_model_path: Path to logistic regression model
            rf_model_path: Path to random forest model
            lstm_model_path: Path to PyTorch LSTM model weights
            lstm_model_info_path: Path to LSTM model architecture info
            lstm_scaler_path: Path to LSTM scaler
            lstm_features_path: Path to LSTM features
            output_dir: Directory for output files
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.models = {}
        
        # Set device for LSTM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load logistic regression model if provided
        if lr_model_path:
            try:
                print("Loading logistic regression model...")
                self.models['Logistic Regression'] = joblib.load(lr_model_path)
                print("  ✓ Logistic regression model loaded successfully")
            except Exception as e:
                print(f"  ✗ Error loading logistic regression model: {e}")
        
        # Load random forest model if provided
        if rf_model_path:
            try:
                print("Loading random forest model...")
                self.models['Random Forest'] = joblib.load(rf_model_path)
                print("  ✓ Random forest model loaded successfully")
            except Exception as e:
                print(f"  ✗ Error loading random forest model: {e}")
        
        # Load LSTM model if provided
        if all([lstm_model_path, lstm_model_info_path, lstm_scaler_path, lstm_features_path]):
            try:
                print("Loading LSTM model...")
                # Load model architecture info
                model_info = joblib.load(lstm_model_info_path)
                
                # Create model with same architecture
                lstm_model = LSTMModel(
                    input_size=model_info['input_size'],
                    hidden_size=model_info['hidden_size'],
                    num_layers=model_info['num_layers'],
                    dropout=model_info['dropout']
                ).to(self.device)
                
                # Load model weights
                lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device))
                lstm_model.eval()  # Set to evaluation mode
                
                # Load scaler and feature columns
                lstm_scaler = joblib.load(lstm_scaler_path)
                lstm_features = joblib.load(lstm_features_path)
                
                self.models['LSTM'] = {
                    'model': lstm_model,
                    'scaler': lstm_scaler,
                    'feature_columns': lstm_features,
                    'sequence_length': model_info['sequence_length'],
                    'model_info': model_info
                }
                print("  ✓ LSTM model loaded successfully")
            except Exception as e:
                print(f"  ✗ Error loading LSTM model: {e}")
        
        # Load data if any models are loaded
        if self.models:
            self.load_data()
    
    def load_data(self):
        """
        Load and preprocess data for all models.
        """
        print("\nLoading and preprocessing data...")
        self.df = load_and_preprocess_data(self.data_path)
        
        print("Calculating player statistics...")
        self.player_stats_df = calculate_player_stats(self.df)
        
        # Prepare features specifically for each model
        self.prepare_model_specific_data()
    
    def prepare_model_specific_data(self):
        """
        Prepare data specifically for each model.
        """
        if 'Logistic Regression' in self.models or 'Random Forest' in self.models:
            print("Preparing match features for tree-based models...")
            self.match_features = prepare_match_features(self.player_stats_df)
            
            # First, separate non-mirrored entries
            non_mirrored_matches = self.match_features[~self.match_features['match_id'].str.contains('_mirrored', na=False)]
            
            # Sort by match order
            non_mirrored_matches = non_mirrored_matches.sort_values('match_order')
            
            # Calculate split point ensuring we have test data
            split_ratio = 0.7
            num_matches = len(non_mirrored_matches)
            
            if num_matches <= 1:
                print("ERROR: Not enough non-mirrored matches for training and testing")
                self.train_data = non_mirrored_matches
                self.test_data = pd.DataFrame()  # Empty DataFrame
            else:
                # Ensure at least one sample for testing
                split_idx = min(int(num_matches * split_ratio), num_matches - 1)
                
                # Split the data
                self.train_data = non_mirrored_matches.iloc[:split_idx]
                self.test_data = non_mirrored_matches.iloc[split_idx:]
            
            print(f"  Train data: {len(self.train_data)} matches")
            print(f"  Test data: {len(self.test_data)} matches")
        
        if 'LSTM' in self.models:
            print("Preparing sequence data for LSTM model...")
            sequence_length = self.models['LSTM'].get('sequence_length', 5)
            self.X_sequences, self.y_sequences, self.match_ids, _ = prepare_sequence_data(
                self.df, sequence_length)
            
            # Ensure we have sequences for evaluation
            if len(self.X_sequences) > 0:
                # Use chronological split for LSTM data
                split_ratio = 0.7
                num_sequences = len(self.X_sequences)
                
                # Ensure at least one sample for testing
                split_idx = min(int(num_sequences * split_ratio), num_sequences - 1)
                
                # Create indices for splitting
                indices = np.arange(num_sequences)
                np.random.seed(42)  # For reproducibility
                np.random.shuffle(indices)
                
                # Split
                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]
                
                # Create train/test sets
                self.X_train_lstm = self.X_sequences[train_indices]
                self.y_train_lstm = self.y_sequences[train_indices]
                self.X_test_lstm = self.X_sequences[test_indices]
                self.y_test_lstm = self.y_sequences[test_indices]
                
                print(f"  LSTM train data: {len(self.X_train_lstm)} sequences")
                print(f"  LSTM test data: {len(self.X_test_lstm)} sequences")
            else:
                print("  ✗ No sequence data available for LSTM")
                self.models.pop('LSTM', None)
    
    def evaluate_models(self):
        """
        Evaluate all models on the test data.
        
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Evaluate logistic regression and random forest models
        for model_name in ['Logistic Regression', 'Random Forest']:
            if model_name in self.models:
                print(f"\nEvaluating {model_name}...")
                model_data = self.models[model_name]
                
                # Skip evaluation if no test data
                if len(self.test_data) == 0:
                    print(f"  Skipping evaluation for {model_name} - no test data available")
                    continue
                
                # Get model, scaler, and feature columns
                model = model_data['model']
                scaler = model_data['scaler']
                feature_columns = model_data['feature_columns']
                
                # Prepare test data
                X_test = self.test_data[feature_columns].values
                y_test = self.test_data['team1_won'].values
                
                # Scale features
                X_test_scaled = scaler.transform(X_test)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Calculate ROC curve and AUC
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Calculate precision-recall curve
                p, r, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(r, p)
                
                # Store results
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': conf_matrix,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                    'roc_curve': (fpr, tpr),
                    'roc_auc': roc_auc,
                    'pr_curve': (r, p),
                    'pr_auc': pr_auc
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  ROC AUC: {roc_auc:.4f}")
                print(f"  PR AUC: {pr_auc:.4f}")
        
        # Evaluate LSTM model
        if 'LSTM' in self.models:
            print("\nEvaluating LSTM...")
            model_data = self.models['LSTM']
            
            # Skip evaluation if no test data
            if len(self.X_test_lstm) == 0:
                print(f"  Skipping evaluation for LSTM - no test data available")
                return results
            
            # Get model, scaler, and feature columns
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Prepare test data
            X_test = self.X_test_lstm
            y_test = self.y_test_lstm
            
            # Flatten for scaling (combine time steps and samples)
            X_flat = X_test.reshape(-1, X_test.shape[2])
            X_flat_scaled = scaler.transform(X_flat)
            
            # Reshape back to sequences
            X_test_scaled = X_flat_scaled.reshape(X_test.shape)
            
            # Convert to tensor
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                y_prob = model(X_test_tensor).cpu().numpy().flatten()
            
            y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Calculate precision-recall curve
            p, r, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(r, p)
            
            # Store results
            results['LSTM'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'roc_curve': (fpr, tpr),
                'roc_auc': roc_auc,
                'pr_curve': (r, p),
                'pr_auc': pr_auc
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            print(f"  PR AUC: {pr_auc:.4f}")
        
        return results
    
    def plot_roc_curves(self, results):
        """
        Plot ROC curves for all models.
        
        Args:
            results: Dictionary with evaluation results
        """
        if not results:
            print("No results to plot ROC curves")
            return
            
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            fpr, tpr = result['roc_curve']
            roc_auc = result['roc_auc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'))
        plt.close()
    
    def plot_precision_recall_curves(self, results):
        """
        Plot precision-recall curves for all models.
        
        Args:
            results: Dictionary with evaluation results
        """
        if not results:
            print("No results to plot precision-recall curves")
            return
            
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            recall, precision = result['pr_curve']
            pr_auc = result['pr_auc']
            
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curves.png'))
        plt.close()
    
    def plot_confusion_matrices(self, results):
        """
        Plot confusion matrices for all models.
        
        Args:
            results: Dictionary with evaluation results
        """
        if not results:
            print("No results to plot confusion matrices")
            return
            
        for model_name, result in results.items():
            plt.figure(figsize=(8, 6))
            cm = result['confusion_matrix']
            
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
            plt.close()
    
    def plot_performance_comparison(self, results):
        """
        Plot performance comparison bar chart for all models.
        
        Args:
            results: Dictionary with evaluation results
        """
        if not results:
            print("No results to plot performance comparison")
            return
            
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
        
        # Prepare data for plotting
        model_names = list(results.keys())
        data = {
            'Model': [],
            'Metric': [],
            'Value': []
        }
        
        for model_name in model_names:
            for metric, metric_name in zip(metrics, metric_names):
                data['Model'].append(model_name)
                data['Metric'].append(metric_name)
                data['Value'].append(results[model_name][metric])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Plot
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Metric', y='Value', hue='Model', data=df)
        plt.title('Model Performance Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Model')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'))
        plt.close()
        
        # Also create a metrics table
        metrics_table = pd.pivot_table(df, values='Value', index=['Model'], columns=['Metric'])
        metrics_table.to_csv(os.path.join(self.output_dir, 'performance_metrics.csv'))
        
        return metrics_table
    
    def plot_prediction_distribution(self, results):
        """
        Plot prediction probability distributions for all models.
        
        Args:
            results: Dictionary with evaluation results
        """
        if not results:
            print("No results to plot prediction distributions")
            return
            
        n_models = len(results)
        
        if n_models == 0:
            return
            
        fig_height = 5 * n_models
        plt.figure(figsize=(14, fig_height))
        
        for i, (model_name, result) in enumerate(results.items(), 1):
            plt.subplot(n_models, 1, i)
            
            y_prob = result['y_prob']
            y_test = result['y_test']
            
            # Plot distribution
            sns.histplot(y_prob[y_test == 1], color='green', alpha=0.5, bins=20, label='True Positives')
            sns.histplot(y_prob[y_test == 0], color='red', alpha=0.5, bins=20, label='True Negatives')
            
            plt.title(f'{model_name} - Prediction Probability Distribution')
            plt.xlabel('Prediction Probability')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_distributions.png'))
        plt.close()
    
    def run_comprehensive_comparison(self):
        """
        Run a comprehensive comparison of all models.
        
        Returns:
            Dictionary with all results
        """
        # Make sure we have data
        if not hasattr(self, 'df'):
            print("No data loaded. Loading data...")
            self.load_data()
        
        # Evaluate models
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        evaluation_results = self.evaluate_models()
        
        # Skip visualizations if no results
        if not evaluation_results:
            print("\nNo models could be evaluated. Skipping visualizations.")
            return {'evaluation': {}, 'metrics': None}
        
        # Create visualizations
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Plot ROC curves
        print("  Plotting ROC curves...")
        self.plot_roc_curves(evaluation_results)
        
        # Plot precision-recall curves
        print("  Plotting precision-recall curves...")
        self.plot_precision_recall_curves(evaluation_results)
        
        # Plot confusion matrices
        print("  Plotting confusion matrices...")
        self.plot_confusion_matrices(evaluation_results)
        
        # Plot performance comparison
        print("  Plotting performance comparison...")
        metrics_table = self.plot_performance_comparison(evaluation_results)
        
        # Plot prediction distributions
        print("  Plotting prediction distributions...")
        self.plot_prediction_distribution(evaluation_results)
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        print(f"Results saved to {self.output_dir} directory")
        
        # Print summary metrics
        print("\nSummary Metrics:")
        if metrics_table is not None:
            print(metrics_table)
        
        return {
            'evaluation': evaluation_results,
            'metrics': metrics_table
        }


def main():
    """
    Main function to run the comparison script.
    """
    parser = argparse.ArgumentParser(description='Compare esports prediction models')
    parser.add_argument('--data', required=True, help='Path to match data file')
    parser.add_argument('--lr-model', help='Path to logistic regression model')
    parser.add_argument('--rf-model', help='Path to random forest model')
    parser.add_argument('--lstm-model', help='Path to PyTorch LSTM model weights')
    parser.add_argument('--lstm-model-info', help='Path to LSTM model architecture info')
    parser.add_argument('--lstm-scaler', help='Path to LSTM scaler')
    parser.add_argument('--lstm-features', help='Path to LSTM features')
    parser.add_argument('--output', default='comparison', help='Directory for output files')
    
    args = parser.parse_args()
    
    # Create comparison object
    comparison = ModelComparison(
        args.data,
        args.lr_model,
        args.rf_model,
        args.lstm_model,
        args.lstm_model_info,
        args.lstm_scaler,
        args.lstm_features,
        args.output
    )
    
    # Run comparison
    results = comparison.run_comprehensive_comparison()


if __name__ == "__main__":
    main()

# Example usage:
# python comprehensive_model_comparison_updated.py --data match_data.xlsx --lr-model output/esports_prediction_model.pkl --rf-model output/esports_prediction_rf_model.pkl --lstm-model output/esports_lstm_pytorch_model.pth --lstm-model-info output/lstm_model_info.pkl --lstm-scaler output/lstm_pytorch_scaler.pkl --lstm-features output/lstm_pytorch_features.pkl