#  🔊 LoRAW UI
Low Rank Adaptation for Waveforms, with auto-installer and GUI to set up [Stable Audio 1.0 open](https://huggingface.co/stabilityai/stable-audio-open-1.0) LoRA training and model finetunes.

Based on [LoRAW](https://github.com/NeuralNotW0rk/LoRAW) and [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools)

## ⚙️Installation
### **Automatic installation:**
Clone this repo (`git clone https://github.com/Big-Onche/LoRAW.git`) and run the install script based on your OS.

### **Manual installation:**
- Clone the repository `git clone https://github.com/Big-Onche/LoRAW.git`
- Navigate into the cloned directory `cd LoRAW`
- Set up a virtual environment `python -m venv env`
- Activate the new environment:
  - Windows: `env\scripts\activate`
  - Linux/Mac: `source env/bin/activate`
- Install torch: `pip install torch==2.4.1+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- Run setup.py: `pip install .\loraw`

## 🎶 Inference
**VRAM requirement: 8GB at full precision, 6GB with half precision.**

- Download [Stable Audio 1.0 checkpoint](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/model.ckpt) and put it in the folder `LoRAW/models/checkpoints` , you can also put your checkpoint anywhere and select your custom path using the GUI.
- Launch the Gradio interface using the run script based on your OS.

## 🏋🏼 LoRA Training
**VRAM requirement: 8GB**

- Setup your dataset:
  - Create a folder with your audio files.
  - For each audio sample, create a .txt file with the same name and in the same folder, the content of the text file should contain the prompts based on the audio sample.
  - Supported audio formats: flac, wav, mp3, m4a, ogg, opus.
- Launch the GUI using the run script based on your OS.
- Go to the tab 'Train a LoRA' and select the folder of your dataset
- Adjust learning rate, network dimension, and network alpha if needed.
- In the tab 'Demos settings' adjust the demo created during training as needed.
- Launch the training.

## 📝 First impressions on LoRA training
- With a Learning Rate of 0.0001, 200-300 steps seem to be the sweet spot in most cases, for music styles or drum loops/melodies, more steps should be needed.
- You can get pretty good results even with small datasets (single sound effect with slight pitch and speed variations)
- Network rank and alpha of 16 is ok, maybe higher if you want to train on a specific music style.
- Overfitting sign: The outputs are peppered with short audio glitches, especially for constant sounds like rain, wind, ambient drone music, etc. 

| Type of sound  | Comment     |
|-------------------|-----------------------------------------------------------------------|
| Single sound effect  | With a single sound effect and some speed and pitch variations, like in the example dataset, you can achieve a good convergence to create "natural" variations of the same sound. |
| Multiple sound effects | Not tested. |
| Music instruments |  The convergence seems fast as the sound effects, but a wider dataset will be better for melody diversity. |
| Drone/ambient |  Can easily replicate drone and ambient styles. |
| Music style |  It seems to require many steps to get something, separating percussion, bass, melodies, etc. in the dataset appears to help. |
| Melody | Not tested |
| Voice |  Like RVC training, a total dataset of 5 - 10 minutes is ok, but expect only gibberish when using it. |

## 🏋🏼 Pre-trained model fine-tuning 
**VRAM requirement: 12GB with 16-mixed, 8GB with 16-true**
Not really tested, would take days to get something on my setup.

## 🔗 References
- https://github.com/NeuralNotW0rk/LoRAW
- https://github.com/cloneofsimo/lora
- https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
