# LoRAW UI
Low Rank Adaptation for Waveforms, with auto-installer and GUI to set up [Stable Audio 1.0 open](https://huggingface.co/stabilityai/stable-audio-open-1.0) LoRA training.

Based on [LoRAW](https://github.com/NeuralNotW0rk/LoRAW) and [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)

# Installation
**Automatic installation:**
Clone this repo ('git clone https://github.com/Big-Onche/LoRAW.git') and run the install script based on your OS.

**Manual installation:**
- Clone the repository `git clone https://github.com/Big-Onche/LoRAW.git`
- Navigate into the cloned directory `cd LoRAW`
- Set up a virtual environment `python -m venv env`
- Activate the new environment:
  - Windows: `env\scripts\activate`
  - Linux/Mac: `source env/bin/activate`
- Install torch: pip install torch==2.4.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
- Run setup.py: `pip install .\loraw`

# Inference
- Download [Stable Audio 1.0 checkpoint](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/model.ckpt) and put it in 'LoRAW/models/checkpoints'
- Launch the Gradio interface using the run script based on your OS.

# LoRA Training
- Setup your dataset
  - Create a folder with your audio and text files.
  - The text files should contain the prompts based on the audio sample.
  - Tweak 'config.json', and 'metadata.py' from the folder datasets/example according to your dataset.
  - Supported audio formats: flac, wav, mp3, m4a, ogg, opus.
- Launch the GUI using the run script based on your OS.
- Select your dataset's config.json file in 'Dataset Config'
- Adjust learning rate, network dimension, and network alpha if needed.
- Launch the training.

##  First impressions on LoRA training
| Type of sound       | Steps  | Learning rate  | Network dim/alpha | Comment                                                                                   |
|---------------------|--------|----------------|-------------------|-------------------------------------------------------------------------------------------|
| Single sound effect | 200-300 | 0.0001 | 16/16 | With a single sound effect and some speed and pitch variations, like in the example dataset, you can achieve a good convergence to create "natural" variations of the same sound. |
| Multiple sound effects | / | / | / | Not tested. |
| Music instruments    | 200-300 | 0.0001 | 16/16 | The convergence seems fast as the sound effects, but a wider dataset will be better for melody diversity. |
| Drone/ambient        | 400-500 | 0.0001 | 16/16 | Can easily replicate drone and ambient styles. |
| Music style          | 1000+   | 0.0001 or higher? | May require higher than 16 in neural dim | It seems to require many steps to get something, separating percussion, bass, melodies, etc. in the dataset appears to help. |
| Melody               | / | / | / | Not tested |
| Voice                | / | / | / | Not tested |

## Configure model
```JSON
"lora": {
    "component_whitelist": ["transformer"],
    "multiplier": 1.0,
    "rank": 16,
    "alpha": 16,
    "dropout": 0,
    "module_dropout": 0,
    "lr": 1e-4
}
```

A full example config that works with stable audio open can be found [here](https://github.com/NeuralNotW0rk/LoRAW/blob/main/examples/model_config.json)

## Set additional args
Then run the modified `train.py` as you would in [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) with the following command line arguments as needed:
- `--use-lora`
    - Set to true to enable lora usage
    - *Default*: false
- `--lora-ckpt-path`
    - A pre-trained lora continue from
- `--relora-every`
    - Enables ReLoRA training if set
    - The number of steps between full-rank updates
    - *Default*: 0
- `--quantize`
    - CURRENTLY BROKEN
    - Set to true to enable 4-bit quantization of base model for QLoRA training
    - *Default*: false



# Usage (manual)

## Construction
Create a loraw using the LoRAWrapper class. For example using a conditional diffusion model for which we only want to target the transformer component:
```Python
from loraw.network import LoRAWrapper

lora = LoRAWrapper(
    target_model,
    component_whitelist=["transformer"],
    lora_dim=16,
    alpha=16,
    dropout=None,
    multiplier=1.0
)
```
If using stable-audio-tools, you can create a LoRA based on your model config:
```Python
from loraw.network import create_lora_from_config

lora = create_lora_from_config(model_config, target_model)
```

## Activation
If you want to load weights into the target model, be sure to do so first as activation will alter the structure and confuse state_dict copying
```Python
lora.activate()
```

## Loading and saving weights
`lora.load_weights(path)` and `lora.save_weights(path)` are for simple file IO. `lora.merge_weights(path)` can be used to add more checkpoints without overwriting the current state.

## Training
With stable-audio-tools, after activation, you can simply call
```Python
lora.prepare_for_training(training_wrapper)
```

For training to work manually, you need to:
- Set all original weights to `requires_grad = False`
- Set lora weights set to `requires_grad = True` (easily accessed with `lora.residual_modules.parameters()`)
- Update the optimizer to use the lora parameters (the same parameters as the previous step)

# References
- https://github.com/NeuralNotW0rk/LoRAW
- https://github.com/cloneofsimo/lora
- https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
