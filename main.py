import torch
import torch.nn.parallel

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import random

import numpy as np

import wandb


from src.training.training_utils import create_optimizer, adjust_learning_rate
from src.utils.calc import compute_num_tokens
from tqdm.auto import tqdm

from accelerate import (
    DistributedDataParallelKwargs,
    Accelerator,
    DistributedType,
)
import logging


import subprocess
from omegaconf import OmegaConf
import argparse
import webdataset as wds


logging.basicConfig(level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description="GPT* Training")

    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
    )

    parser.add_argument("--cfg", default="conf/config.yaml", type=str)

    parser.add_argument("--keep", default=False, action="store_true")

    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    return args


def check_args(cfg):
    assert cfg.data.corpus in ["webtextnouveau"]
    assert cfg.model.type in ["GPT2"]


def main():

    global cfg, global_step, id, args, best_val

    args = parse()

    print(args)

    # A bit hacky but this is a way to control the global step number of the model
    global_step = 0
    best_val = 1e6

    cfg = OmegaConf.load(args.cfg)

    if torch.cuda.is_available():
        logging.info("Defaulting to CUDA.")

    check_args(cfg)

    if cfg.data.seq_len != cfg.data.train_len:
        logging.info(f"ALiBi with {cfg.data.train_len}")

        # All sequences have been preprocessed into chunks of 1024 BPE tokens. If
        # we want to train with ALiBi and a shorter ctx, just have to reshape
        def preprocess(batch):
            x = batch["input_id.pth"][: cfg.data.seq_len].reshape(
                -1, cfg.data.train_len
            )
            return x.long()

    else:

        def preprocess(batch):
            x = batch["input_id.pth"][: cfg.data.seq_len]
            return x.long()

    logging.info("Loading Configuration file.")

    # ---------- Model Creation ---------- #
    if cfg.model.type == "GPT2":
        from src.models.GPT2 import model_getter

        model = model_getter(
            cfg.model.size,
            vocab_size=50257,
            num_ctx=cfg.data.train_len,
            **{
                "fused_residuals": cfg.model.fused_residuals,
                "num_head": cfg.model.num_head,
                "use_alibi": cfg.model.ALiBi,
            },
        )

    if cfg.model.type != "GPT2":
        # Required if using gMLP/aMLP
        model.prepare()

    # Accelerate code
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    from braceexpand import braceexpand

    # We have two separate datasets so we just combine them together and perform a large shuffle at beginning of training
    original_urls_train = "data/processed/train/webtextplusplus_no_resample_train-{000000..000211}.tar.gz"
    new_urls_train = (
        "data/processed/train/WebTextNouveau_train-{000000..000050}.tar.gz"
    )

    original_urls_validation = "data/processed/train/webtextplusplus_no_resample_validation-{000000..000018}.tar.gz"
    new_urls_validation = (
        "data/processed/train/WebTextNouveau_validation-{000000..000006}.tar.gz"
    )

    train_urls = list(braceexpand(new_urls_train)) + list(
        braceexpand(original_urls_train)
    )

    validation_urls = list(braceexpand(new_urls_validation)) + list(
        braceexpand(original_urls_validation)
    )

    # Since we are combining multiple lists of Shards, to properly shuffle,
    # we need to set the buffer/initial values to be quite large.
    # This incurs a performance hit at the beginning of the run, but its a one-off.
    # On a 32 core Xeon, shuffling 10M records takes around 10 minutes
    train_dataset = wds.DataPipeline(
        wds.SimpleShardList(train_urls),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1e7, initial=1e7, rng=random.Random(23)),
        wds.decode(),
        wds.map(preprocess),
    )

    validation_dataset = wds.DataPipeline(
        wds.SimpleShardList(validation_urls),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1e5, initial=1e5, rng=random.Random(23)),
        wds.decode(),
        wds.map(preprocess),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = create_optimizer(
        model,
        cfg.training.weight_decay,
        cfg.training.max_lr,
    )

    # Load model states here
    if cfg.training.use_pretrained_weights:
        state_dict = torch.load(
            f"completed/GPT2_small.pth.tar",
            map_location="cpu",
        )
        model.load_state_dict(state_dict)

    # Resume training from checkpoint
    resume_step = None
    if args.resume:

        state_dict = torch.load(
            f"{cfg.training.training_checkpoint_file}",
            map_location="cpu",
        )

        # Load model, optimizer dict
        model.load_state_dict(state_dict["state_dict"])

        optimizer = create_optimizer(
            model, cfg.training.weight_decay, 0, use_bnb=False
        )

        optimizer.load_state_dict(state_dict["optimizer"])

        if args.keep:
            id = state_dict["run_id"]
            global_step = state_dict["step"]
            resume_step = state_dict["step"]

            best_val = state_dict["best_val"]
            validation_loss = state_dict["validation_loss"]

            scaler = state_dict["scaler"]

            if validation_loss < best_val:
                best_val = validation_loss

        else:
            global_step = 0

        logging.info(f"Reloading from checkpoint: Global Step: {global_step}")

    model.to(accelerator.device)

    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )

    if args.resume:
        logging.info("Restoring fp16 scaler")
        accelerator.scaler.load_state_dict(scaler)

    if args.resume == False or args.keep == False:
        id = wandb.util.generate_id()

    if accelerator.is_local_main_process:

        n_devices = torch.cuda.device_count()

        num_tokens = compute_num_tokens(
            n_devices,
            batch_per_gpu=cfg.training.batch_size,
            accum_steps=cfg.training.gradient_accumulation_steps,
            n_ctx=cfg.data.seq_len,
            num_gradient_updates=cfg.training.steps,
        )

        wandb.init(id=id, resume="allow", project="Sprinkle")
        # Training Setup
        if args.resume == False or args.keep == False:
            # No need to change this if resuming from a run
            wandb.config.max_epochs = cfg.training.max_epochs
            wandb.config.steps = cfg.training.steps
            wandb.config.batch_size = cfg.training.batch_size
            wandb.config.num_tokens = num_tokens / 1e9

            # Hyperparam Setup
            wandb.config.weight_decay = cfg.training.weight_decay
            wandb.config.warmup = cfg.training.warmup
            wandb.config.accum_steps = cfg.training.gradient_accumulation_steps
            wandb.config.seq_len = cfg.data.seq_len
            wandb.config.model = cfg.model.size

            # Model setup
            wandb.config.corpus = cfg.data.corpus

    for i in range(cfg.training.max_epochs):

        accelerator.wait_for_everyone()
        try:
            _ = train(
                train_loader,
                model,
                optimizer,
                epoch=i,
                eval_loader=eval_loader,
                accelerator=accelerator,
                resume_step=resume_step,
            )

            # This may be needed if we exit a train epoch mid grad accum step.
            optimizer.zero_grad()

        except GetOutOfLoop:
            logging.info("Training has completed.")
            accelerator.wait_for_everyone()

            unwrapped_model = accelerator.unwrap_model(model)

            accelerator.save(
                {
                    "epoch": -1,
                    "step": global_step,
                    "run_id": id,
                    "validation_loss": -1,
                    "best_val": -1,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": unwrapped_model.state_dict(),
                    "scaler": accelerator.scaler.state_dict(),
                },
                f=f"checkpoints/training_params.pth.tar",
            )

            if accelerator.is_local_main_process:
                try:
                    directory = (
                        "openwebtxtbf"
                        if cfg.data.corpus == "webtextnouveau"
                        else "bookcorpusbf"
                    )
                    path = f"checkpoint.tar.gz"
                    prefix = cfg.training.prefix + "_final"
                    subprocess.Popen(
                        [
                            "python",
                            f"data_utils/upload_checkpoint_CLI.py",
                            "--dir",
                            directory,
                            "--prefix",
                            prefix,
                            "--path",
                            path,
                        ]
                    )
                except Exception as e:
                    logging.warning("Model checkpointing failed.")
            return True


class GetOutOfLoop(Exception):
    pass


def train(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    eval_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    resume_step: int = None,
) -> float:
    losses = AverageMeter()

    # switch to train mode
    model.train()
    running_loss = 0

    for i, text in enumerate(
        tqdm(train_loader, disable=not accelerator.is_local_main_process)
    ):

        if i == 0:
            if accelerator.is_local_main_process:
                logging.info(
                    "Performing shard shuffle. Depending on your buffer size this may take a while"
                )

        if (
            epoch == 0
            and resume_step != None
            and i <= cfg.training.gradient_accumulation_steps * resume_step
        ):
            continue

        # Control global steps for model training. This is what the lr decay is tied to.
        global global_step, best_val

        if cfg.data.seq_len != cfg.data.train_len:
            text = text.reshape(-1, cfg.data.train_len)
            # https://gist.github.com/fattorib/6e43a8c3d8696a6a92fd063773b144cb

        if accelerator.distributed_type == DistributedType.TPU:
            lm_logits, loss = model(text, text)

        else:
            with accelerator.autocast():
                lm_logits, loss = model(text, text)

        loss = loss / cfg.training.gradient_accumulation_steps

        accelerator.backward(loss)

        running_loss += loss.item()

        if (i + 1) % cfg.training.gradient_accumulation_steps == 0:

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step = global_step + 1

            if global_step < cfg.training.steps:
                pass
            else:
                raise GetOutOfLoop

            reduced_loss = (
                cfg.training.gradient_accumulation_steps
                * accelerator.gather(loss)
            )

            if not accelerator.optimizer_step_was_skipped:
                lr = adjust_learning_rate(optimizer, global_step, cfg)

            else:
                lr = 0

            if (global_step - 1) % cfg.training.log_freq == 0:
                eval_loss = validate(
                    eval_loader,
                    model,
                    num_eval_steps=cfg.training.eval_steps,
                    accelerator=accelerator,
                )

                model.train()

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                # Deepspeed throws a keyerror for some reason...?
                accelerator.save(
                    {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "run_id": id,
                        "validation_loss": eval_loss,
                        "best_val": best_val,
                        "optimizer": optimizer.state_dict(),
                        "state_dict": unwrapped_model.state_dict(),
                        "scaler": accelerator.scaler.state_dict(),
                    },
                    f=f"checkpoints/training_params.pth.tar",
                )

                if accelerator.is_local_main_process:

                    is_best = eval_loss < best_val

                    if eval_loss < best_val:
                        best_val = eval_loss

                    try:
                        directory = (
                            "openwebtxtbf"
                            if cfg.data.corpus == "webtextnouveau"
                            else "bookcorpusbf"
                        )
                        path = f"checkpoint.tar.gz"
                        prefix = (
                            cfg.training.prefix + "_best"
                            if is_best == True
                            else cfg.training.prefix + "_latest"
                        )
                        subprocess.Popen(
                            [
                                "python",
                                f"upload_checkpoint_CLI.py",
                                "--dir",
                                directory,
                                "--prefix",
                                prefix,
                                "--path",
                                path,
                            ]
                        )

                        logging.info("Model checkpoint uploaded")
                    except Exception as e:
                        logging.warning(
                            "Model checkpointing failed. Training will continue."
                        )
                    logging.info(
                        f" Loss: {losses.test:.10f} ({losses.avg:.4f})\t Global Step: {global_step}"
                    )

                    # Logging weights

                    wandb.log(
                        {
                            "Train LM Loss": running_loss,
                            "Validation LM Loss": eval_loss,
                            "Learning Rate": lr,
                            "Last Train LM Perplexity": np.exp(running_loss),
                            "Validation LM Perplexity": np.exp(eval_loss),
                        }
                    )

                    running_loss = 0
            else:
                if accelerator.is_local_main_process:
                    wandb.log(
                        {
                            "Train LM Loss": running_loss,
                            "Learning Rate": lr,
                            "Last Train LM Perplexity": np.exp(running_loss),
                        }
                    )
                running_loss = 0

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.test = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, test, n=1):
        self.test = test
        self.sum += test * n
        self.count += n
        self.avg = self.sum / self.count


def validate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    num_eval_steps: int,
    accelerator: Accelerator,
):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, text in enumerate(
        tqdm(loader, disable=not accelerator.is_local_main_process)
    ):
        if i < num_eval_steps:

            if cfg.data.seq_len != cfg.data.train_len:
                # https://gist.github.com/fattorib/6e43a8c3d8696a6a92fd063773b144cb
                text = text.reshape(-1, cfg.data.train_len)
            if accelerator.distributed_type == DistributedType.TPU:
                with torch.no_grad():
                    _, loss = model(text, text)
            else:
                with accelerator.autocast():
                    with torch.no_grad():
                        _, loss = model(text, text)
            reduced_loss = accelerator.gather(loss)

            losses.update((reduced_loss.mean().item()), text.size(0))
        else:
            break

    return losses.avg


if __name__ == "__main__":
    main()
