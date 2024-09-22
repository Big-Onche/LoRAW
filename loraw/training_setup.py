import os
import subprocess
import sys
import customtkinter as ctk

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        # Create a small window for the tooltip
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25  # Position the tooltip slightly right
        y += self.widget.winfo_rooty() + 25  # Position the tooltip slightly down

        # Create a Toplevel window for the tooltip
        self.tooltip_window = ctk.CTkToplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # Remove window decorations
        self.tooltip_window.geometry(f"+{x}+{y}")

        # Create a frame to hold the label
        frame = ctk.CTkFrame(self.tooltip_window, fg_color="#48484B")
        frame.pack(padx=5, pady=0)

        # Create the label inside the frame
        label = ctk.CTkLabel(frame, text=self.text, text_color="white")
        label.pack(padx=10, pady=5)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pre-filled fields
        current_dir = os.getcwd()
        self.pretrained_ckpt_path = ctk.StringVar(value=os.path.join(current_dir, 'models', 'checkpoints', 'model.ckpt'))
        self.model_config = ctk.StringVar(value=os.path.join(current_dir, 'models', 'checkpoints', 'model_config.json'))
        self.save_dir = ctk.StringVar(value=os.path.join(current_dir, 'models', 'loras'))
        self.lora_ckpt_path = ctk.StringVar()
        self.dataset_config = ctk.StringVar(value=os.path.join(current_dir, 'datasets', 'exemple', 'config.json'))
        self.batch_size = ctk.IntVar(value=4)
        self.ckpt_every = ctk.IntVar(value=500)
        self.train_lora = ctk.StringVar(value="true")
        self.lora_rank = ctk.IntVar(value=16)
        self.lora_alpha = ctk.IntVar(value=16)

        # configure window
        self.title("LoRAW Training Setup")
        width = 730
        height = 450
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"+{x}+{y}")

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        self.UI_browse_row(0, "Checkpoint Path", self.pretrained_ckpt_path, True, "Select model checkpoint", "ckpt") # Pretrained checkpoint path
        self.UI_browse_row(1, "Model Config", self.model_config, True, "Select model configuration file", "json") # Model config
        self.UI_browse_row(2, "Dataset Config", self.dataset_config, True, "Select dataset configuration file", "json") # Dataset config
        self.UI_browse_row(3, "Pretrained LoRA Checkpoint", self.lora_ckpt_path, True, "Select LoRA checkpoint", "ckpt") # Pretrained lora checkpoint path
        self.UI_browse_row(4, "Save Directory", self.save_dir, "Select save folder") # Save directory
        self.UI_slider_row(5, "Batch Size", self.batch_size, 1, 16) # Batch size slider
        self.UI_slider_row(6, "Checkpoint every (steps)", self.ckpt_every, 10, 10000) # Checkpoint every slider
        self.UI_slider_row(7, "Network Rank (Dimension)", self.lora_rank, 4, 128, True) # LoRA rank
        self.UI_slider_row(8, "Network Alpha", self.lora_alpha, 4, 128, True) # LoRA alpha
        self.UI_checkbox(9, "Train LoRA", self.train_lora, "Check this box to enable LoRA training, and uncheck for full model finetuning") # LoRA training checkbox
        ctk.CTkButton(self, text="Launch Training", command=self.launch_training).grid(row=10, columnspan=3, pady=10) # Launch button

    def UI_browse_row(self, row, ui_text, var, browse_text, is_file=False, extension=""):
        ctk.CTkLabel(self, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
        ctk.CTkEntry(self, textvariable=var, width=400).grid(row=row, column=1, padx=5, pady=5)
        if is_file:
            ctk.CTkButton(self, text="Browse", command=lambda: self.browse_file(browse_text, [(f"{extension} files", f"*.{extension}")], var)).grid(row=row, column=2, padx=5, pady=5)
        else:
            ctk.CTkButton(self, text="Browse", command=lambda: self.browse_folder(var)).grid(row=row, column=2, padx=5, pady=5)

    def UI_slider_row(self, row, ui_text, var, min_val, max_val, pow_two=False):
        def enforce_power_of_two(value):
            pow_values = [4, 8, 16, 32, 64, 128]
    
            if pow_two:
                # Ensure the value is within the range of min_val and max_val
                if min_val <= value <= max_val:
                    # Calculate the index based on the slider value
                    index = pow_values.index(min(pow_values, key=lambda x: abs(x - value)))
                    var.set(pow_values[index])
                else:
                    # Set to the nearest bound if out of range
                    if value < min_val:
                        var.set(pow_values[0])  # Set to the smallest power of 2
                    elif value > max_val:
                        var.set(pow_values[-1])  # Set to the largest power of 2

        def on_slider_change(value):
            value = float(value)
            if pow_two:
                enforce_power_of_two(value)
            else:
                var.set(int(value))  # Set the value directly if pow_two is False
    
        # Create label
        ctk.CTkLabel(self, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
    
        # Create slider and bind it to on_slider_change to adjust to nearest power of two if needed
        slider = ctk.CTkSlider(self, from_=min_val, to=max_val, variable=var, number_of_steps=max_val - min_val + 1, command=on_slider_change)
        slider.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
    
        # Create entry field
        ctk.CTkEntry(self, textvariable=var, width=140).grid(row=row, column=2, padx=5, pady=5)

        # Initial enforcement to ensure it's set correctly
        if pow_two:
            enforce_power_of_two(var.get())

    def UI_checkbox(self, row, ui_text, var, tooltip_text = ""):
        ctk.CTkLabel(self, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
        checkbox = ctk.CTkCheckBox(self, text="", variable=var, onvalue="true", offvalue="false")
        checkbox.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
        if tooltip_text:
            ToolTip(checkbox, tooltip_text)

    def browse_file(self, title, type, var):
        file_path = ctk.filedialog.askopenfilename(
            title=title, 
            filetypes=type
        )
        if file_path:
            var.set(file_path)

    def browse_folder(self, var):
        folder_path = ctk.filedialog.askdirectory()
        if folder_path:
            var.set(folder_path)

    def launch_training(self):
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
            f'--pretrained-ckpt-path={self.pretrained_ckpt_path.get()}',
            f'--model-config={self.model_config.get()}',
            f'--save-dir={self.save_dir.get()}',
            f'--lora-ckpt-path={self.lora_ckpt_path.get()}',
            f'--dataset-config={self.dataset_config.get()}',
            f'--batch-size={self.batch_size.get()}',
            f'--checkpoint-every={self.ckpt_every.get()}',
            f'--use-lora={self.train_lora.get()}',
            f'--lora-rank={self.lora_rank.get()}',
            f'--lora-alpha={self.lora_alpha.get()}'
        ]

        self.destroy()  # Close window

        # Run the command with the modified environment
        subprocess.run(command, env=env)

if __name__ == "__main__":
    app = App()
    app.mainloop()
