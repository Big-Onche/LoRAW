from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
import random
import warnings

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

from loraw.network import create_lora_from_config
from loraw.callbacks import LoRAModelCheckpoint, ReLoRAModelCheckpoint
from pytorch_lightning.plugins import BitsandbytesPrecisionPlugin

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():

    args = get_all_args()

    seed = args.seed
    checkpoint_dir = args.save_dir

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    if args.use_lora == 'true':
        if args.lora_ckpt_path:
            training_mode = "Pre-trained LoRA training"
        else:
            training_mode = "Fresh LoRA training"
    else:
        training_mode = "Model fine tuning"

    print(f"Training mode: {training_mode}")

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        state_dict = load_ckpt_state_dict(args.pretrained_ckpt_path)
        model.load_state_dict(state_dict)

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    # LORA: Create and activate
    if args.use_lora == 'true':
        lora = create_lora_from_config(model_config, model, args.lora_rank, args.lora_alpha)
        if args.lora_ckpt_path:
            lora_weights = torch.load(args.lora_ckpt_path, map_location="cpu")["state_dict"]
            lora.load_weights(lora_weights)
        lora.activate()

    training_wrapper = create_training_wrapper_from_config(model_config, model)
    print(f"Checkpoint every {args.checkpoint_every} steps, saved in {args.save_dir}")

    # LORA: Prepare training
    if args.use_lora == 'true':
        lora.prepare_for_training(training_wrapper)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(training_wrapper)

    print(f"W&B mode: {wandb_logger.experiment.mode}")

    callbacks = []

    callbacks.append(ExceptionCallback())
  
    if checkpoint_dir is not None:
        if args.save_dir and isinstance(wandb_logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints")
    else:
        print("Checkpoint directory is not set, saving to models/loras/")
        checkpoint_dir = os.path.join('.', 'models', 'loras')

    # LORA: Custom checkpoint callback
    if args.use_lora  == 'true':
        if args.relora_every == 0:
            callbacks.append(LoRAModelCheckpoint(lora=lora, every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1))
        else:
            callbacks.append(ReLoRAModelCheckpoint(lora=lora, every_n_train_steps=args.relora_every, dirpath=checkpoint_dir, save_top_k=-1, checkpoint_every_n_updates=args.checkpoint_every // args.relora_every))
    else:
        callbacks.append(pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1))

    callbacks.append(ModelConfigEmbedderCallback(model_config))

    callbacks.append(create_demo_callback_from_config(model_config, demo_dl=train_dl))

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2, 
                                        contiguous_gradients=True, 
                                        overlap_comm=True, 
                                        reduce_scatter=True, 
                                        reduce_bucket_size=5e8, 
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True
                                        )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 

    try:
        if args.quantize == 'true':
            plugins = BitsandbytesPrecisionPlugin(mode="nf4", dtype=torch.float16, ignore_modules={*lora.residual_modules})
            trainer = pl.Trainer(
                devices=args.num_gpus,
                accelerator="gpu",
                num_nodes = args.num_nodes,
                strategy=strategy,
                plugins=plugins,
                accumulate_grad_batches=args.accum_batches, 
                callbacks=callbacks,
                logger=wandb_logger,
                log_every_n_steps=1,
                max_epochs=10000000,
                default_root_dir=args.save_dir,
                gradient_clip_val=args.gradient_clip_val,
                reload_dataloaders_every_n_epochs = 0,
            )
        else:
            trainer = pl.Trainer(
                devices=args.num_gpus,
                accelerator="gpu",
                num_nodes = args.num_nodes,
                strategy=strategy,
                precision=args.precision,
                accumulate_grad_batches=args.accum_batches, 
                callbacks=callbacks,
                logger=wandb_logger,
                log_every_n_steps=1,
                max_epochs=10000000,
                default_root_dir=args.save_dir,
                gradient_clip_val=args.gradient_clip_val,
                reload_dataloaders_every_n_epochs = 0,
            )
    except Exception as e:
        print(f"Failed to initialize trainer: {e}")

    try:
        print("Starting training...")
        trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == '__main__':
    main()