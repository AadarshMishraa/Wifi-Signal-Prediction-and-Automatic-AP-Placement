import os
import json
from datetime import datetime
import shutil
import pandas as pd

class ResultsManager:
    def __init__(self, base_dir="results"):
        """Initialize the results manager.
        
        Args:
            base_dir (str): Base directory for storing results
        """
        self.base_dir = base_dir
        self.current_run = None
        
    def start_new_run(self, description=None):
        """Start a new test run.
        
        Args:
            description (str): Optional description of the run
            
        Returns:
            str: Path to the run directory
        """
        # Create timestamp-based run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        
        # Create run directory structure
        run_dir = os.path.join(self.base_dir, run_id)
        subdirs = ['data', 'visualizations', 'models', 'floor_plans']
        
        os.makedirs(run_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
        
        # Store run information
        self.current_run = {
            'id': run_id,
            'timestamp': timestamp,
            'description': description,
            'path': run_dir,
            'metrics': {},
            'files': {subdir: [] for subdir in subdirs}
        }
        
        # Save initial run info
        self._save_run_info()
        
        return run_dir
    
    def save_data(self, data, filename, category='data'):
        """Save data file to the current run.
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Name of the file
            category (str): Category of data (data, visualizations, models)
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_new_run() first.")
            
        filepath = os.path.join(self.current_run['path'], category, filename)
        
        # Save based on file type
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            # Assume it's a file to be copied
            shutil.copy2(data, filepath)
            
        self.current_run['files'][category].append(filename)
        self._save_run_info()
        
    def save_metrics(self, metrics, model_name):
        """Save model metrics for the current run.
        
        Args:
            metrics (dict): Dictionary of metrics
            model_name (str): Name of the model
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_new_run() first.")
            
        self.current_run['metrics'][model_name] = metrics
        self._save_run_info()
        
    def save_visualization(self, figure_path, description=None):
        """Save a visualization to the current run.
        
        Args:
            figure_path (str): Path to the visualization file
            description (str): Optional description of the visualization
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_new_run() first.")
            
        filename = os.path.basename(figure_path)
        dest_path = os.path.join(self.current_run['path'], 'visualizations', filename)
        
        shutil.copy2(figure_path, dest_path)
        
        self.current_run['files']['visualizations'].append({
            'filename': filename,
            'description': description
        })
        self._save_run_info()
        
    def save_floor_plan(self, floor_plan_path, floor_number=None, description=None):
        """Save a floor plan to the current run.
        
        Args:
            floor_plan_path (str): Path to the floor plan image
            floor_number (int): Optional floor number
            description (str): Optional description
        """
        if self.current_run is None:
            raise ValueError("No active run. Call start_new_run() first.")
            
        filename = os.path.basename(floor_plan_path)
        dest_path = os.path.join(self.current_run['path'], 'floor_plans', filename)
        
        shutil.copy2(floor_plan_path, dest_path)
        
        self.current_run['files']['floor_plans'].append({
            'filename': filename,
            'floor_number': floor_number,
            'description': description
        })
        self._save_run_info()
        
    def _save_run_info(self):
        """Save run information to JSON file."""
        info_path = os.path.join(self.current_run['path'], 'run_info.json')
        with open(info_path, 'w') as f:
            json.dump(self.current_run, f, indent=2)
            
    def get_run_info(self, run_id=None):
        """Get information about a specific run.
        
        Args:
            run_id (str): ID of the run to get info for. If None, returns current run.
            
        Returns:
            dict: Run information
        """
        if run_id is None:
            if self.current_run is None:
                raise ValueError("No active run.")
            return self.current_run
            
        info_path = os.path.join(self.base_dir, run_id, 'run_info.json')
        if not os.path.exists(info_path):
            raise ValueError(f"Run {run_id} not found.")
            
        with open(info_path, 'r') as f:
            return json.load(f)
            
    def list_runs(self):
        """List all available runs.
        
        Returns:
            list: List of run information dictionaries
        """
        runs = []
        if os.path.exists(self.base_dir):
            for run_id in os.listdir(self.base_dir):
                try:
                    runs.append(self.get_run_info(run_id))
                except:
                    continue
        return sorted(runs, key=lambda x: x['timestamp'], reverse=True)
    
    def generate_report(self, run_id=None, output_path=None):
        """Generate a markdown report for a run.
        
        Args:
            run_id (str): ID of the run to report on. If None, uses current run.
            output_path (str): Path to save the report. If None, saves in run directory.
        """
        run_info = self.get_run_info(run_id)
        
        # Generate report content
        report = [
            f"# Test Run Report: {run_info['id']}",
            f"\nRun Date: {run_info['timestamp']}",
        ]
        
        if run_info['description']:
            report.append(f"\nDescription: {run_info['description']}")
            
        # Add metrics section
        if run_info['metrics']:
            report.append("\n## Model Performance")
            for model_name, metrics in run_info['metrics'].items():
                report.append(f"\n### {model_name}")
                for metric, value in metrics.items():
                    report.append(f"- {metric}: {value}")
                    
        # Add visualizations section
        if run_info['files']['visualizations']:
            report.append("\n## Visualizations")
            for viz in run_info['files']['visualizations']:
                desc = viz['description'] if isinstance(viz, dict) else 'No description'
                filename = viz['filename'] if isinstance(viz, dict) else viz
                report.append(f"\n### {filename}")
                report.append(f"Description: {desc}")
                report.append(f"![{filename}](visualizations/{filename})")
                
        # Add floor plans section
        if run_info['files']['floor_plans']:
            report.append("\n## Floor Plans")
            for plan in run_info['files']['floor_plans']:
                if isinstance(plan, dict):
                    floor_num = f"Floor {plan.get('floor_number', '')}" if plan.get('floor_number') else ''
                    desc = plan.get('description', 'No description')
                    filename = plan['filename']
                    report.append(f"\n### {floor_num}")
                    report.append(f"Description: {desc}")
                    report.append(f"![{filename}](floor_plans/{filename})")
                else:
                    report.append(f"\n![Floor Plan](floor_plans/{plan})")
        
        # Save report
        if output_path is None:
            output_path = os.path.join(run_info['path'], 'report.md')
            
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
            
        return output_path
