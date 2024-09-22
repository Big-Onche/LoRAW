import tkinter as tk
from tkinter import filedialog, font as tkFont
import os
import subprocess
import sys

# Create main window
root = tk.Tk()
root.title("Stable Audio 1.0 LoRA Training Setup")

# Set fixed window size
root.geometry("500x350")
root.resizable(False, False)

# Center the window on the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (450 // 2)
y = (screen_height // 2) - (350 // 2)
root.geometry(f"+{x}+{y}")

# Pre-filled fields
current_dir = os.getcwd()
pretrained_ckpt_path = tk.StringVar(value=os.path.join(current_dir, 'models', 'checkpoints', 'model.ckpt'))
model_config = tk.StringVar(value=os.path.join(current_dir, 'models', 'checkpoints', 'model_config.json'))
save_dir = tk.StringVar(value=os.path.join(current_dir, 'models', 'loras'))
lora_ckpt_path = tk.StringVar()
dataset_config = tk.StringVar(value=os.path.join(current_dir, 'datasets', 'exemple', 'config.json'))
batch_size = tk.IntVar(value=4)

# Set font styles
font_style = tkFont.Font(family="Helvetica", size=10)

# Update ints
def update_var(val, var):
    var.set(int(val))

# Browse functions
def browse_folder(title, var):
    folder_path = filedialog.askdirectory(title=title)
    if folder_path:
        var.set(folder_path)

def browse_file(title, type, var):
    file_path = filedialog.askopenfilename(
        title=title, 
        filetypes=type
    )
    if file_path:
        var.set(file_path)

# Launch loraw/train.py with args
def launch_script():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Set PYTHONPATH to include current directory and loraw subfolder
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{current_dir}{os.pathsep}{os.path.join(current_dir, 'loraw')}"
    
    # Determine the path to the Python executable inside the virtual environment
    if sys.platform == "win32":
        python_executable = os.path.join(current_dir, 'env', 'Scripts', 'python.exe')
    else:
        python_executable = os.path.join(current_dir, 'env', 'bin', 'python')

    # Ensure the virtual environment's Python is being used
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"Python executable not found at: {python_executable}")

    # Launch train.py with arguments
    command = [
        python_executable, 'loraw/train.py',
        f'--pretrained-ckpt-path={pretrained_ckpt_path.get()}',
        f'--model-config={model_config.get()}',
        f'--save-dir={save_dir.get()}',
        f'--lora-ckpt-path={lora_ckpt_path.get()}',
        f'--dataset-config={dataset_config.get()}',
        f'--batch-size={batch_size.get()}',
		f'--use-lora=true'
        #f'--precision= '
    ]

    root.destroy() # Close window

    # Run the command with the modified environment
    subprocess.run(command, env=env)

# GUI layout
for i in range(6):
    root.grid_rowconfigure(i, weight=1)

tk.Label(root, text="Checkpoint Path", anchor='w', font=font_style).grid(row=0, column=0, padx=5, pady=5, sticky='w')
tk.Entry(root, textvariable=pretrained_ckpt_path, width=40).grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=lambda: browse_file("Select model checkpoint", [("ckpt Files", "*.ckpt")], pretrained_ckpt_path)).grid(row=0, column=2, padx=5, pady=5)

tk.Label(root, text="Model Config", anchor='w', font=font_style).grid(row=1, column=0, padx=5, pady=5, sticky='w')
tk.Entry(root, textvariable=model_config, width=40).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=lambda: browse_file("Select model configuration file", [("JSON Files", "*.json")], model_config)).grid(row=1, column=2, padx=5, pady=5)

tk.Label(root, text="Dataset Config", anchor='w', font=font_style).grid(row=4, column=0, padx=5, pady=5, sticky='w')
tk.Entry(root, textvariable=dataset_config, width=40).grid(row=4, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=lambda: browse_file("Select dataset configuration file", [("JSON Files", "*.json")], dataset_config)).grid(row=4, column=2, padx=5, pady=5)

tk.Label(root, text="Pretrained LoRA Checkpoint", anchor='w', font=font_style).grid(row=3, column=0, padx=5, pady=5, sticky='w')
tk.Entry(root, textvariable=lora_ckpt_path, width=40).grid(row=3, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=lambda: browse_file("Select LoRA checkpoint", [("ckpt Files", "*.ckpt")], lora_ckpt_path)).grid(row=3, column=2, padx=5, pady=5)

tk.Label(root, text="Save Directory", anchor='w', font=font_style).grid(row=2, column=0, padx=5, pady=5, sticky='w')
tk.Entry(root, textvariable=save_dir, width=40).grid(row=2, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=lambda: browse_folder("Select LoRA save directory", save_dir)).grid(row=2, column=2, padx=5, pady=5)

tk.Label(root, text="Batch Size", anchor='w', font=font_style).grid(row=5, column=0, padx=5, pady=5, sticky='w')

# Create a scale (slider)
slider = tk.Scale(
    root,
    from_=1,
    to=16,
    orient='horizontal',
    command=lambda val: update_var(val, batch_size),
    length=245,
    sliderlength=20
)
slider.set(batch_size.get()) 
slider.grid(row=5, column=1, padx=5, pady=5)

tk.Button(root, text="Launch LoRA training", command=launch_script, bg='blue', fg='white').grid(row=6, columnspan=3, pady=10)

root.mainloop()
