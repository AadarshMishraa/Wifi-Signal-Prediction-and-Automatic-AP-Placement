import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

class WiFiVisualizer:
    def __init__(self, output_dir="visualizations"):
        """Initialize the WiFi data visualizer.
        
        Args:
            output_dir (str): Directory to store visualizations
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def create_dashboard(self, data, model_results):
        """Create a comprehensive visualization dashboard.
        
        Args:
            data (pd.DataFrame): Original data
            model_results (dict): Results from model training
        """
        print("Creating visualizations...")
        
        # Create individual plots
        self._plot_signal_distribution(data)
        self._plot_signal_over_time(data)
        self._plot_model_comparison(model_results)
        self._plot_feature_importance(model_results)
        self._plot_prediction_accuracy(model_results)
        
        print(f"Visualizations saved in {self.output_dir}/")
        
    def _plot_signal_distribution(self, data):
        """Plot signal strength distribution."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='rssi', hue='ssid', multiple="stack")
        plt.title('Signal Strength Distribution by Access Point')
        plt.xlabel('RSSI (dBm)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.output_dir, 'signal_distribution.png'))
        plt.close()
        
    def _plot_signal_over_time(self, data):
        """Plot signal strength over time."""
        plt.figure(figsize=(12, 6))
        for ssid in data['ssid'].unique():
            ssid_data = data[data['ssid'] == ssid]
            plt.plot(ssid_data['timestamp'], ssid_data['rssi'], label=ssid, alpha=0.7)
        plt.title('Signal Strength Over Time')
        plt.xlabel('Time')
        plt.ylabel('RSSI (dBm)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'signal_time_series.png'))
        plt.close()
        
    def _plot_model_comparison(self, model_results):
        """Plot model performance comparison."""
        models = list(model_results.keys())
        rmse_scores = [results['metrics']['rmse'] for results in model_results.values()]
        r2_scores = [results['metrics']['r2'] for results in model_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        ax1.bar(models, rmse_scores)
        ax1.set_title('RMSE by Model')
        ax1.set_ylabel('RMSE')
        
        # R² comparison
        ax2.bar(models, r2_scores)
        ax2.set_title('R² Score by Model')
        ax2.set_ylabel('R²')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'))
        plt.close()
        
    def _plot_feature_importance(self, model_results):
        """Plot feature importance for each model."""
        for model_name, results in model_results.items():
            if 'feature_importance' in results:
                importance_dict = results['feature_importance']
                features = list(importance_dict.keys())
                importances = list(importance_dict.values())
                
                # Sort by absolute importance
                sorted_idx = np.argsort(np.abs(importances))
                pos = np.arange(len(features)) + .5
                
                plt.figure(figsize=(12, len(features)/2))
                plt.barh(pos, np.array(importances)[sorted_idx])
                plt.yticks(pos, np.array(features)[sorted_idx])
                plt.xlabel('Feature Importance')
                plt.title(f'Feature Importance - {model_name.upper()}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'feature_importance_{model_name}.png'))
                plt.close()
        
    def _plot_prediction_accuracy(self, model_results):
        """Plot prediction accuracy for each model."""
        for model_name, results in model_results.items():
            predictions = results['predictions']
            actual = results['actual']
            
            plt.figure(figsize=(10, 6))
            plt.scatter(actual, predictions, alpha=0.5)
            plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
            plt.xlabel('Actual Signal Strength (dBm)')
            plt.ylabel('Predicted Signal Strength (dBm)')
            plt.title(f'Prediction Accuracy - {model_name.upper()}')
            
            # Add metrics to plot
            rmse = results['metrics']['rmse']
            r2 = results['metrics']['r2']
            plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.2f}',
                    transform=plt.gca().transAxes, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'prediction_accuracy_{model_name}.png'))
            plt.close()
