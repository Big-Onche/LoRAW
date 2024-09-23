import os

def get_custom_metadata(info, audio):
    directory = os.path.dirname(os.path.abspath(__file__)) # Get the directory where the current .py file is located
    base_name = os.path.splitext(info["relpath"])[0] # Remove the extension
    text_path = os.path.join(directory, base_name + '.txt') # Create the path for the .txt file
    
    try:
        # Try to open and read the corresponding text file
        with open(text_path, 'r') as f:
            text = f.read()
        return {"prompt": text}
    
    except FileNotFoundError:
        # Handle the case where the .txt file doesn't exist
        print(f"Warning: Text file {text_path} not found for {info['relpath']}")
        return {"prompt": "shotgun shot with shells falling"}
    
    except Exception as e:
        # Handle any other errors that might occur
        print(f"Error reading {text_path}: {e}")
        return {"prompt": "shotgun shot with shells falling"}