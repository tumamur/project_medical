import yaml
import json

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