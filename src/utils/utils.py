import yaml
import json

def get_monitor_metrics_mode():
    return {
        'val_loss': 'min',
        'val_acc': 'max',
        'val_f1': 'max',
        'val_precision': 'max',
    }

def read_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error reading the config file: {e}")
        return None
    
def read_disease_combination_counts(path):
    with open(path, 'r') as f:
        disease_combination_counts = json.load(f)
    return disease_combination_counts