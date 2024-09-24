import os
import subprocess
import sys
import json
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
        self.add("Demos settings")

        # Configure overall tab appearance
        self.configure(height=520)
        for button in self._segmented_button._buttons_dict.values():
            button.configure(width=170, height=30, font=("TkDefaultFont", 14, "bold"))

        # Configure columns
        for tab_name in ["Run Stable Audio 1.0", "Train a LoRA", "Finetune Stable Audio", "Demos settings"]:
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
        elif selected_tab_name == "Demos settings":
            self.show_demo_tab()

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
        self.ui.UI_browse_row(tab, 2, "Dataset Path", self.app.dataset_path, "Select dataset folder")
        self.ui.UI_browse_row(tab, 3, "Pretrained LoRA Checkpoint", self.app.lora_ckpt_path, "Select LoRA checkpoint", True, "ckpt")
        self.ui.UI_browse_row(tab, 4, "Save Directory", self.app.save_dir)
        self.ui.UI_slider_row(tab, 5, "Batch Size", self.app.batch_size, 1, 16)
        self.ui.UI_slider_row(tab, 6, "Checkpoint every (steps)", self.app.ckpt_every, 10, 10000)
        self.ui.UI_slider_row(tab, 7, "Network Rank (Dimension)", self.app.lora_rank, 4, 128, False, True)
        self.ui.UI_slider_row(tab, 8, "Network Alpha", self.app.lora_alpha, 4, 128, False, True)
        self.ui.UI_slider_row(tab, 9, "Learning rate", self.app.learning_rate, 0.00001, 0.001, True)
        self.ui.UI_string_row(tab, 10, "Default prompt (trigger)", self.app.default_prompt)
        ctk.CTkButton(tab, text="Launch Training", command=lambda: self.app.launch(True)).grid(row=11, columnspan=3, pady=10)


    def show_finetune_tab(self):
        tab = self.tab("Finetune Stable Audio")
        self.app.train_lora=ctk.StringVar(value="false")

        self.ui.UI_browse_row(tab, 0, "Checkpoint Path", self.app.pretrained_ckpt_path, "Select model checkpoint", True, "ckpt")
        self.ui.UI_browse_row(tab, 1, "Model Config", self.app.model_config, "Select model configuration file", True, "json")
        self.ui.UI_browse_row(tab, 2, "Dataset Path", self.app.dataset_path, "Select dataset folder")
        self.ui.UI_browse_row(tab, 3, "Save Directory", self.app.save_dir)
        self.ui.UI_slider_row(tab, 4, "Batch Size", self.app.batch_size, 1, 16)
        self.ui.UI_slider_row(tab, 5, "Checkpoint every (steps)", self.app.ckpt_every, 10, 10000)
        self.ui.UI_dropdown(tab, 6, "Precision", self.app.precision, options=["16-mixed", "16-true"], tooltip_text="16-true: 8GB VRAM. 16-mixed: 12GB VRAM")
        ctk.CTkButton(tab, text="Launch Training", command=lambda: self.app.launch(True)).grid(row=7, columnspan=3, pady=10)


    def show_demo_tab(self):
        tab = self.tab("Demos settings")

        self.ui.UI_slider_row(tab, 0, "Demo every (steps)", self.app.demos_every, 10, 1000)
        self.ui.UI_slider_row(tab, 1, "Amount of demos", self.app.num_demos, 1, 3)
        self.ui.UI_slider_row(tab, 2, "Demos steps", self.app.demos_steps, 10, 150)
        self.ui.UI_slider_row(tab, 3, "Demos CFGs", self.app.demos_cfgs, 1, 20)

        self.ui.UI_h_bar(tab, 4)

        for it in range(self.app.num_demos.get()):
            p_row = 5 + 2 * it
            l_row = 6 + 2 * it
            self.ui.UI_string_row(tab, p_row, f"Demo {it+1} prompt", self.app.demos_prompts[it])
            self.ui.UI_slider_row(tab, l_row, f"Demo {it+1} length (secs)", self.app.demos_lengths[it], 1, 47)



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


    def UI_string_row(self, parent, row, ui_text, var, browse_text=""):
        ctk.CTkLabel(parent, text=ui_text).grid(row=row, column=0, padx=5, pady=5, sticky='w')
        ctk.CTkEntry(parent, textvariable=var, width=400).grid(row=row, column=1, padx=5, pady=5)


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

    def UI_h_bar(self, parent, row, pady=20, height=2, width=450):
        separator = ctk.CTkFrame(parent, height=height, width=width, fg_color="gray")
        separator.grid(row=row, column=0, columnspan=3, pady=pady, padx=5, sticky='ew')


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



class DatasetConfig:
    def __init__(self, app):
        self.app = app

    def apply_dataset_config(self):
        with open(self.app.dataset_config_path.get(), 'r') as file:
            data = json.load(file)

        # Modify the "path" value
        for dataset in data['datasets']:
            if 'path' in dataset:
                dataset['path'] = self.app.dataset_path.get()
            if 'default_prompt' in dataset:
                dataset['default_prompt'] = self.app.default_prompt.get()

        with open(self.app.dataset_config_path.get(), 'w') as file:
            json.dump(data, file, indent=4)


    def apply_demo_config(self):
        with open(self.app.model_config.get(), 'r') as file:
            data = json.load(file)

        demo_settings = data['training']['demo']
    
        if 'demo_every' in demo_settings:
            demo_settings['demo_every'] = self.app.demos_every.get()
        if 'demo_steps' in demo_settings:
            demo_settings['demo_steps'] = self.app.demos_steps.get()
        if 'num_demos' in demo_settings:
            demo_settings['num_demos'] = self.app.num_demos.get()

        if 'demo_cond' in demo_settings:
            demo_settings['demo_cond'] = []  # Empty the existing demo_cond
            num_demos = self.app.num_demos.get()

            demo_conditions = []  # Initialize an empty list

            if num_demos >= 1:
                demo_conditions.append({"prompt": self.app.demos_prompts[0].get(), "seconds_start": 0, "seconds_total": self.app.demos_lengths[0].get()})
            if num_demos >= 2:
                demo_conditions.append({"prompt": self.app.demos_prompts[1].get(), "seconds_start": 0, "seconds_total": self.app.demos_lengths[1].get()})
            if num_demos >= 3:
                demo_conditions.append({"prompt": self.app.demos_prompts[2].get(), "seconds_start": 0, "seconds_total": self.app.demos_lengths[2].get()})

            # Assign the demo_conditions to demo_settings['demo_cond']
            demo_settings['demo_cond'] = demo_conditions

        if 'demo_cfg_scales' in demo_settings:
            demo_settings['demo_cfg_scales'] = [self.app.demos_cfgs.get()]

        with open(self.app.model_config.get(), 'w') as file:
            json.dump(data, file, indent=4)



class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.data = DatasetConfig(self)

        # Pre-filled fields
        self.working_dir = os.getcwd()
        self.dataset_config_path = ctk.StringVar(value=os.path.join(self.working_dir, 'config', 'dataset_config.json'))
        self.pretrained_ckpt_path = ctk.StringVar(value=os.path.join(self.working_dir, 'models', 'checkpoints', 'model.ckpt'))
        self.model_config = ctk.StringVar(value=os.path.join(self.working_dir, 'config', 'model_config.json'))
        self.save_dir = ctk.StringVar(value=os.path.join(self.working_dir, 'models', 'loras'))
        self.lora_dir = self.save_dir
        self.lora_ckpt_path = ctk.StringVar()
        self.dataset_path = ctk.StringVar(value=os.path.join(self.working_dir, 'datasets', 'example'))
        self.batch_size = ctk.IntVar(value=4)
        self.ckpt_every = ctk.IntVar(value=500)
        self.train_lora = ctk.StringVar(value="true")
        self.lora_rank = ctk.IntVar(value=16)
        self.lora_alpha = ctk.IntVar(value=16)
        self.learning_rate = ctk.DoubleVar(value=1e-4)
        self.default_prompt = ctk.StringVar(value="MyLoRATrigger")
        self.model_half = ctk.StringVar(value="false")
        self.precision = ctk.StringVar(value="16-mixed")
        self.num_demos = ctk.IntVar(value="3")
        self.demos_every = ctk.IntVar(value="100")
        self.demos_steps = ctk.IntVar(value="100")
        self.demos_cfgs = ctk.IntVar(value="7")
        self.demos_prompts = [
            ctk.StringVar(value="Prompt for demo 1"),
            ctk.StringVar(value="Prompt for demo 2"),
            ctk.StringVar(value="Prompt for demo 3")
        ]
        self.demos_lengths = [
            ctk.IntVar(value=47),
            ctk.IntVar(value=47),
            ctk.IntVar(value=47)
        ]

        # Configure window
        self.title("LoRAW UI v0.5")
        width = 792
        height = 560
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"+{x}+{y}")

        # Create tabs
        self.tabs = Tabs(self, app=self)
        self.tabs.grid(row=0, column=0, padx=20, pady=20)


    def launch(self, training=False):
        self.data.apply_dataset_config()
        self.data.apply_demo_config()

        # Set PYTHONPATH to include current directory and loraw subfolder
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.working_dir}{os.pathsep}{os.path.join(self.working_dir, 'loraw')}"

        # Determine the path to the Python executable inside the virtual environment
        if sys.platform == "win32":
            python_executable = os.path.join(self.working_dir, 'env', 'Scripts', 'python.exe')
        else:
            python_executable = os.path.join(self.working_dir, 'env', 'bin', 'python')

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
                f'--dataset-config={self.dataset_config_path.get()}',
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