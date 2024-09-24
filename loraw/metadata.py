import os
import sys
import json

config_file_path = os.path.join(os.getcwd(), 'config', 'dataset_config.json')

def get_custom_metadata(info, audio):
    with open(config_file_path, 'r') as file:
            data = json.load(file)

    try:
        with open(config_file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1) # Exit the program to avoid infinite loop

    default_prompt = data.get('datasets', [{}])[0].get('default_prompt')
    if default_prompt is None:
        print(f"Failed to read default prompt in {config_file_path}")
        sys.exit(1)

    dataset_path = data.get('datasets', [{}])[0].get('path')
    if dataset_path is None:
        print(f"Failed to read dataset path in {config_file_path}")
        sys.exit(1)

    base_name = os.path.splitext(info["relpath"])[0] # Remove the extension
    text_path = os.path.join(dataset_path, base_name + '.txt') # Create the path for the .txt file

    try:
        # Try to open and read the corresponding text file
        with open(text_path, 'r') as f:
            text = f.read()
        return {"prompt": text}
    
    except FileNotFoundError:
        # Handle the case where the .txt file doesn't exist
        print(f"Text file {text_path}, using default prompt ({default_prompt})")
        return {"prompt": default_prompt}
    
    except Exception as e:
        # Handle any other errors that might occur
        print(f"Error reading {text_path}: {e}, using default prompt ({default_prompt})")
        return {"prompt": default_prompt}