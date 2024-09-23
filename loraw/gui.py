import os
import subprocess
import sys
import customtkinter as ctk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")


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
        frame.pack(padx=0, pady=0)

        # Create the label inside the frame
        label = ctk.CTkLabel(frame, text=self.text, text_color="white")
        label.pack(padx=10, pady=5)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None



class Tabs(ctk.CTkTabview):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        self.ui = UIPresets(self)

        # Create tabs
        self.add("Run Stable Audio 1.0")
        self.add("Train a LoRA")
        self.add("Finetune Stable Audio")
		
        # Configure overall tab appearance
        self.configure(height=480, border_color="red")
        for button in self._segmented_button._buttons_dict.values():
            button.configure(width=170, height=30, font=("TkDefaultFont", 14, "bold"))

        # Configure columns
        for tab_name in ["Run Stable Audio 1.0", "Train a LoRA", "Finetune Stable Audio"]:
            tab = self.tab(tab_name)
            tab.grid_columnconfigure(0, minsize=180)
            tab.grid_columnconfigure(1, minsize=400)

        # Set default tab
        self.set("Run Stable Audio 1.0")
        self.show_inference_tab()

        # Handle tab switching
       	self.configure(command=self.on_tab_change)


    def on_tab_change(self):
        selected_tab_name = self.get()

        if selected_tab_name == "Run Stable Audio 1.0":
            self.show_inference_tab()
        elif selected_tab_name == "Train a LoRA":
            self.show_lora_train_tab()
        elif selected_tab_name == "Finetune Stable Audio":
            self.show_finetune_tab()

    def show_inference_tab(self):
        tab = self.tab("Run Stable Audio 1.0")

        self.ui.UI_browse_row(tab, 0, "Checkpoint Path", self.app.pretrained_ckpt_path, "Select model checkpoint", True, "ckpt")
        self.ui.UI_browse_row(tab, 1, "Model Config", self.app.model_config, "Select model configuration file", True, "json")
        self.ui.UI_browse_row(tab, 2, "LoRA Directory", self.app.lora_dir)
        self.ui.UI_checkbox(tab, 3, "Use half precision", self.app.model_half, "You should enable half precision if you have less than 8GB VRAM")
        ctk.CTkButton(tab, text="Launch Gradio", command=lambda: self.app.launch(False)).grid(row=4, columnspan=3, pady=10)


    def show_lora_train_tab(self):
        tab = self.tab("Train a LoRA")
        self.app.train_lora=ctk.StringVar(value="true")
		
        self.ui.UI_browse_row(tab, 0, "Checkpoint Path", self.app.pretrained_ckpt_path, "Select model checkpoint", True, "ckpt")
        self.ui.UI_browse_row(tab, 1, "Model Config", self.app.model_config, "Select model configuration file", True, "json")
        self.ui.UI_browse_row(tab, 2, "Dataset Config", self.app.dataset_config, "Select dataset configuration file", True, "json")
        self.ui.UI_browse_row(tab, 3, "Pretrained LoRA Checkpoint", self.app.lora_ckpt_path, "Select LoRA checkpoint", True, "ckpt")
        self.ui.UI_browse_row(tab, 4, "Save Directory", self.app.save_dir)
        self.ui.UI_slider_row(tab, 5, "Batch Size", self.app.batch_size, 1, 16)
        self.ui.UI_slider_row(tab, 6, "Checkpoint every (steps)", self.app.ckpt_every, 10, 10000)
        self.ui.UI_slider_row(tab, 7, "Network Rank (Dimension)", self.app.lora_rank, 4, 128, True)
        self.ui.UI_slider_row(tab, 8, "Network Alpha", self.app.lora_alpha, 4, 128, False, True)
        self.ui.UI_slider_row(tab, 9, "Learning rate", self.app.learning_rate, 0.00001, 0.001, True)
        ctk.CTkButton(tab, text="Launch Training", command=lambda: self.app.launch(True)).grid(row=10, columnspan=3, pady=10)


    def show_finetune_tab(self):
        tab = self.tab("Finetune Stable Audio")
        self.app.train_lora=ctk.StringVar(value="false")

        self.ui.UI_browse_row(tab, 0, "Checkpoint Path", self.app.pretrained_ckpt_path, "Select model checkpoint", True, "ckpt")
        self.ui.UI_browse_row(tab, 1, "Model Config", self.app.model_config, "Select model configuration file", True, "json")
        self.ui.UI_browse_row(tab, 2, "Dataset Config", self.app.dataset_config, "Select dataset configuration file", True, "json")
        self.ui.UI_browse_row(tab, 3, "Save Directory", self.app.save_dir)
        self.ui.UI_slider_row(tab, 4, "Batch Size", self.app.batch_size, 1, 16)
        self.ui.UI_slider_row(tab, 5, "Checkpoint every (steps)", self.app.ckpt_every, 10, 10000)
        self.ui.UI_dropdown(tab, 6, "Precision", self.app.precision, options=["16-mixed", "16-true"], tooltip_text="16-true: 8GB VRAM. 16-mixed: 12GB VRAM")
        ctk.CTkButton(tab, text="Launch Training", command=lambda: self.app.launch(True)).grid(row=7, columnspan=3, pady=10)


class UIPresets:
    def __init__(self, app):
        self.app = app


    def UI_browse_row(self, parent, row, ui_text, var, browse_text="", is_file=False, extension=""):
        ctk.CTkLabel(parent, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
        ctk.CTkEntry(parent, textvariable=var, width=400).grid(row=row, column=1, padx=5, pady=5)
        if is_file:
            ctk.CTkButton(parent, text="Browse", command=lambda: self.browse_file(browse_text, [(f"{extension} files", f"*.{extension}")], var)).grid(row=row, column=2, padx=5, pady=5)
        else:
            ctk.CTkButton(parent, text="Browse", command=lambda: self.browse_folder(var)).grid(row=row, column=2, padx=5, pady=5)


    def UI_slider_row(self, parent, row, ui_text, var, min_val, max_val, float_val=False, pow_two=False):
        def enforce_power_of_two(value):
            pow_values = [4, 8, 16, 32, 64, 128]
            if pow_two:
                if min_val <= value <= max_val:
                    index = pow_values.index(min(pow_values, key=lambda x: abs(x - value)))
                    var.set(pow_values[index])
                else:
                    if value < min_val:
                        var.set(pow_values[0])
                    elif value > max_val:
                        var.set(pow_values[-1])

        def on_slider_change(value):
            value = float(value)
            if value < 1:
                value = round(value, 5)
            if pow_two:
                enforce_power_of_two(value)
            else:
                var.set(float(value) if float_val else int(value))

        if max_val < 1:
            steps = 100  # Allow continuous movement
        else:
            steps = max_val - min_val + 1  # Use discrete steps for smaller values

        ctk.CTkLabel(parent, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
        slider = ctk.CTkSlider(parent, from_=min_val, to=max_val, variable=var, number_of_steps=steps, command=on_slider_change)
        slider.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
        ctk.CTkEntry(parent, textvariable=var, width=140).grid(row=row, column=2, padx=5, pady=5)

        if pow_two:
            enforce_power_of_two(var.get())


    def UI_checkbox(self, parent, row, ui_text, var, tooltip_text=""):
        ctk.CTkLabel(parent, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
        checkbox = ctk.CTkCheckBox(parent, text="", variable=var, onvalue="true", offvalue="false")
        checkbox.grid(row=row, column=1, padx=5, pady=5, sticky='ew')
        if tooltip_text:
            ToolTip(checkbox, tooltip_text)


    def UI_dropdown(self, parent, row, ui_text, var, options, tooltip_text=""):
        ctk.CTkLabel(parent, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
        dropdown = ctk.CTkOptionMenu(parent, variable=var, values=options)
        dropdown.grid(row=row, column=1, padx=5, pady=5, sticky='ew')

        if tooltip_text:
            ToolTip(dropdown, tooltip_text)


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



class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pre-filled fields
        current_dir = os.getcwd()
        self.pretrained_ckpt_path = ctk.StringVar(value=os.path.join(current_dir, 'models', 'checkpoints', 'model.ckpt'))
        self.model_config = ctk.StringVar(value=os.path.join(current_dir, 'models', 'checkpoints', 'model_config.json'))
        self.save_dir = ctk.StringVar(value=os.path.join(current_dir, 'models', 'loras'))
        self.lora_dir = self.save_dir
        self.lora_ckpt_path = ctk.StringVar()
        self.dataset_config = ctk.StringVar(value=os.path.join(current_dir, 'datasets', 'example', 'config.json'))
        self.batch_size = ctk.IntVar(value=4)
        self.ckpt_every = ctk.IntVar(value=500)
        self.train_lora = ctk.StringVar(value="true")
        self.lora_rank = ctk.IntVar(value=16)
        self.lora_alpha = ctk.IntVar(value=16)
        self.learning_rate = ctk.DoubleVar(value=1e-4)
        self.model_half = ctk.StringVar(value="false")
        self.precision = ctk.StringVar(value="16-mixed")

        # Configure window
        self.title("LoRAW UI v0.3")
        width = 792
        height = 510
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"+{x}+{y}")

        # Create tabs
        self.tabs = Tabs(self, app=self)
        self.tabs.grid(row=0, column=0, padx=20, pady=20)


    def launch(self, training=False):
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
        if training:
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
            if self.train_lora.get() == "true":
                command.append(f'--learning-rate={self.learning_rate.get()}')
            else:
                command.append(f'--precision={self.precision.get()}')
        else:
            command = [
                python_executable, 'loraw/run_gradio.py',
                f'--ckpt-path={self.pretrained_ckpt_path.get()}',
                f'--model-config={self.model_config.get()}',
                f'--lora-dir={self.lora_dir.get()}',
            ]
            if self.model_half.get() == "true":
                command.append('--model-half')

        self.destroy()  # Close window

        # Run the command with the modified environment
        subprocess.run(command, env=env)

if __name__ == "__main__":
    app = App()
    app.mainloop()