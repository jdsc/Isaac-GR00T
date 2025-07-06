# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from transformers import TrainingArguments
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import ConfigGenerator
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.utils.peft import get_lora_model
from torch.utils.data import random_split


@dataclass
class Config:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: str
    """Path to the training dataset directory."""

    validation_dataset_path: str | None = None
    """Optional path to a separate validation dataset."""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP."""

    num_arms: int = 1
    """Number of arms to use for training. Should be greater or equal to 1"""

    num_cams: int = 1
    """Number of cameras to use for training. Should be greater or equal to 1"""

    # Training parameters
    batch_size: int = 16
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_epochs: int = 10
    """Number of epochs to train for."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 500
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1-2B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False # True
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = False #True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: str = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: str = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: str = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    train_test_split: float = 1
    """Percentage of data for training. Example: 1 means you train on 100% of your data"""


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def final_eval(runner: TrainRunner) -> float:
    """Run a final evaluation with the trainer and return the loss."""
    metrics = runner.evaluate()
    return metrics.get("eval_loss", 0.0)


def compute_metrics(eval_pred):
    """
    Args
    ----
    eval_pred: transformers.EvalPrediction
        .predictions: numpy array of shape (N, H, D)
        .label_ids: numpy array of same shape

    Returns
    -------
    dict: any number of scalar metrics keyed by name
    """
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    mse = np.mean((preds - labels) ** 2, dtype=np.float64)
    return {"mse": mse}


#####################################################################################
# main training function
#####################################################################################


def main(config: Config):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # # 1.1 modality configs and transforms
    data_config_cls = ConfigGenerator(num_arms=config.num_arms, num_cams=config.num_cams,
                                      video_keys=config.video_keys, state_keys=config.state_keys, 
                                      action_keys = config.action_keys,
                                      )
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader for training dataset
    full_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
    )

    # Validation dataset logic
    eval_dataset = None
    if config.validation_dataset_path is not None:
        eval_dataset = LeRobotSingleDataset(
            dataset_path=config.validation_dataset_path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )
        train_dataset = full_dataset
    elif config.train_test_split < 1:
        train_size = int(config.train_test_split * len(full_dataset))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    else:
        train_dataset = full_dataset

    # ------------ step 2: load model ------------
    model = GR00T_N1.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=config.num_epochs,
        save_strategy="steps",
        save_steps=config.save_steps,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=eval_dataset is not None,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=config.resume,
        compute_metrics=compute_metrics,
    )

    # 2.3 run experiment
    experiment.train()

    # Evaluate the model on the validation set
    # if eval_dataset is not None:
    #     print("### EVALUATION RESULTS ###")
    #     metrics = experiment.evaluate(eval_dataset=eval_dataset, ignore_keys=["state"])
    #     print(metrics)

    # if eval_dataset is not None:
    #     # Final evaluation and log in wandb
    #     eval_loss = final_eval(experiment)

    #     import wandb

    #     wandb.log({"eval/loss": eval_loss})


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(Config)

    # SO100のを追加
    config.video_keys = ["video.image_cam_0", "video.image_cam_1"]
    config.state_keys = ["state.arm_0"]
    config.action_keys = ["action.arm_0"]

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert config.num_gpus <= available_gpus, (
        f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    )
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")
                    cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
