# LoRAW
Low Rank Adaptation for Waveforms

Designed to be used with Stable Audio Tools

Highly experimental still

# Installation
**Automatic installation:**
Clone this repo and run the install script based on your OS.

**Manual installation:**
- Clone the repository `git clone https://github.com/Big-Onche/LoRAW.git`
- Navigate into the cloned directory `cd LoRAW`
- Set up a virtual environment `python -m venv env`
- Activate the new environment:
  - Windows: `env\scripts\activate`
  - Linux/Mac: `source env/bin/activate`
- Install torch: pip install torch==2.4.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
- Run setup.py: `pip install .\loraw`

## Inference
- Download [Stable Audio 1.0 checkpoint](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/model.ckpt) and put it in 'LoRAW/models/checkpoints'
- Launch Gradio using the run script based on your OS.

# LoRA Training
ToDO

## Configure model
Add a `lora` section to your model config i.e.:

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
- https://github.com/cloneofsimo/lora
- https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
