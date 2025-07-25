import argparse
import logging
import os
from typing import Any, Dict, Tuple
import tqdm, wandb
from model import net_dict

from datasets import collate_fcs,SeqeuncesMotionDataset
import pickle

import numpy as np
import pypose as pp
import torch
import torch.utils.data as Data

import yaml
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.losses import get_motion_loss, get_motion_RMSE
from utils import (cat_state, move_to, save_ckpt, save_state,
                   write_wandb)
import copy

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def train(network: torch.nn.Module, 
          loader: Data.DataLoader, 
          confs: Dict[str, Any], 
          epoch: int, 
          optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """
    Train network for one epoch using a specified data loader.

    Args:
        network (torch.nn.Module): The neural network model to train.
        loader (Data.DataLoader): The data loader for training data.
        confs (Dict[str, Any]): A dictionary of configurations.
        epoch (int): The current epoch number.
        optimizer (torch.optim.Optimizer): The optimizer for training.

    Returns:
        Dict[str, float]: A dictionary containing the average loss and covariance.
    """
    network.train()
    losses, pred_cov = 0.0, 0.0

    t_range = tqdm.tqdm(loader)
    for i, (data,_, label) in enumerate(t_range):
        data, label = move_to([data, label], confs.device)
        rot = label['gt_rot'][:,:-1,:].Log().tensor()
        inte_state = network(data, rot)
        gt_label = network.get_label(label['gt_vel'])
        loss_state = get_motion_loss(inte_state, gt_label, confs)

        # statistics
        losses += loss_state["loss"].item()

        if confs.propcov:
            pred_cov += loss_state["cov_loss"].mean().item()

        t_range.set_description(
            f"training epoch: %03d,losses: %.06f" % (epoch, loss_state["loss"])
        )

        t_range.refresh()
        optimizer.zero_grad()
        loss_state["loss"].backward()
        optimizer.step()

    return {"loss": (losses / (i + 1)), "cov": (pred_cov / (i + 1))}


def test(network: torch.nn.Module, 
         loader: Data.DataLoader, 
         confs: Dict[str, Any]) -> Dict[str, float]:
    """
    Test the network using a specified data loader.

    Args:
        network (torch.nn.Module): The neural network model to test.
        loader (Data.DataLoader): The data loader for test data.
        confs (Dict[str, Any]): A dictionary of configurations.

    Returns:
        Dict[str, float]: A dictionary containing test results.
    """
    network.eval() 
    with torch.no_grad():
        losses, pred_cov = 0.0, 0.0
        pearson_r_sum = torch.zeros(3, device=confs.device)  # [x,y,z] components
        n_batches = 0

        t_range = tqdm.tqdm(loader)
        for i, (data, _, label) in enumerate(t_range):
            data,label = move_to([data, label], confs.device)
            rot = label['gt_rot'][:,:-1,:].Log().tensor()
            inte_state = network(data,rot)
            gt_label = network.get_label(label['gt_vel'])
            loss_state = get_motion_RMSE(inte_state, gt_label,confs)
            # statistics
            losses += loss_state["loss"].item()
            pearson_r_sum += loss_state["pearson_r"]
            n_batches = i + 1

            if confs.propcov:
                pred_cov += loss_state["cov_loss"].mean().item()
                cov_loss_value = torch.sqrt(loss_state['cov_loss'])
            else:   
                cov_loss_value = 0
                
            # Calculate average Pearson R
            pearson_r_avg = pearson_r_sum / n_batches
            
            t_range.set_description(
                "testing loss: %.06f, cov: %.06f, error: %.06f, R[xyz]: %.03f,%.03f,%.03f" % (
                    losses / n_batches, 
                    cov_loss_value, 
                    loss_state['dist'],
                    pearson_r_avg[0],
                    pearson_r_avg[1],
                    pearson_r_avg[2]
                )
            )

            t_range.refresh()
            
    return {
        "loss": (losses / n_batches), 
        "cov": (pred_cov / n_batches),
        "pearson_r": pearson_r_sum / n_batches
    }


def evaluate(network: torch.nn.Module, 
             loader: Data.DataLoader, 
             confs: Dict[str, Any], 
             silent_tqdm: bool = False) -> Dict[str, Any]:
    """
    Evaluate the network using a specified data loader.

    Args:
        network (torch.nn.Module): The neural network model to evaluate.
        loader (Data.DataLoader): The data loader for evaluation data.
        confs (Dict[str, Any]): A dictionary of configurations.
        silent_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation results, including states,
                        covariance, losses, and labels.
    """
    network.eval()
    evaluate_states, loss_states, labels = {}, {}, {}
    pred_cov = []
    skip_key = None
    pearson_r_sum = torch.zeros(3, device=confs.device)  # [x,y,z] components
    n_batches = 0

    with torch.no_grad():
        inte_state = None
        for i, (data,_, label) in enumerate(tqdm.tqdm(loader)):
            
            data,label = move_to([data, label], confs.device)
            rot = label['gt_rot'][:,:-1,:].Log().tensor()
            inte_state = network(data,rot)
            gt_label = network.get_label(label['gt_vel'])
            loss_state = get_motion_RMSE(inte_state, gt_label, confs)

            save_state(loss_states, loss_state)
            save_state(evaluate_states, inte_state)
            save_state(labels, label)
            pearson_r_sum += loss_state["pearson_r"]
            n_batches = i + 1

            if "cov" in inte_state and inte_state["cov"] is not None:
                pred_cov.append(inte_state["cov"])
        
        if confs.propcov:
            if inte_state is None:
                raise ValueError("Error: evaluate dataset is too small to compute.")
            else:
                cov = torch.cat(pred_cov, dim=-2)
        else:
            cov = torch.tensor(0.0, device=confs.device)
            skip_key = "cov_loss"

        for k, v in loss_states.items():
            if skip_key and k == skip_key:
                continue
            loss_states[k] = torch.stack(v, dim=0)

        cat_state(evaluate_states)
        cat_state(labels)
        
        pearson_r_avg = pearson_r_sum / n_batches

        logger.info("evaluating: vel losses %.06f, evaluation cov %.06f, R[xyz]: %.03f,%.03f,%.03f" % (
            loss_states['loss'].mean(), 
            cov.mean(),
            pearson_r_avg[0],
            pearson_r_avg[1],
            pearson_r_avg[2]
        ))

    return {
        "evaluate": evaluate_states,
        "evaluate_cov": cov,
        "loss": loss_states,
        "labels": labels,
        "pearson_r": pearson_r_avg
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/EuRoC/motion_body.conf",
        help="config file path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0"
    )
    parser.add_argument(
        "--load_ckpt",
        default=False,
        action="store_true",
        help="If True, try to load the newest.ckpt in the \
                                                                                exp_dir specificed in our config file.",
    )
    parser.add_argument(
        "--log",
        default=True,
        action="store_false",
        help="if True, save the meta data with wandb",
    )

    args = parser.parse_args()
    logger.info(args)
    
    # Process config to merge with defaults
    from model.code import process_config
    conf = ConfigFactory.parse_file(args.config)
    conf = process_config(conf)

    conf.train.device = args.device
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf["general"]["exp_dir"] = os.path.join(conf.general.exp_dir, conf_name)
    if "gravity" in conf.dataset.train:
        gravity = conf.dataset.train.gravity
        conf.train.put("gravity", conf.dataset.train.gravity)
    else:
        gravity = 9.81007

    train_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.eval)

    if "collate" in conf.dataset.keys():
        collate_fn_train, collate_fn_test = collate_fcs[conf.dataset.collate.type], collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn_train, collate_fn_test = collate_fcs["base"], collate_fcs["base"]

    
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=conf.train.batch_size,
        shuffle = True,
        collate_fn=collate_fn_train,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=conf.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
    )
    eval_loader = Data.DataLoader(
        dataset=eval_dataset,
        batch_size=conf.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        drop_last=True,
    )


    os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))

    if not args.log:
        wandb.disabled = True
        logger.info("wandb is disabled")
    else:
        wandb.init(
            project="AirIO" + exp_folder,
            config=conf.train,
            group=conf.train.network,
            name=conf_name,
        )

    ## optimizer and network
    network = net_dict[conf.model["network"]](
        input_dim=conf.model["n_features"],
        feature_names=conf.model["feature_names"],
        hidden_channels=conf.model["feature_channels"],
        kernel_sizes=conf.model["kernel_sizes"],
        strides=conf.model["strides"],
        padding_num=conf.model["padding_num"],
        propcov=conf.model["propcov"],
    ).to(device=args.device, dtype=train_dataset.get_dtype())

    # # Validate that datasets provide all features required by the model
    # train_dataset.validate_features(feature_names)
    # test_dataset.validate_features(feature_names)
    # eval_dataset.validate_features(feature_names)

    optimizer = torch.optim.Adam(
        network.parameters(), lr=conf.train.lr, weight_decay=conf.train.weight_decay
    )  # to use with ViTs
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=conf.train.factor,
        patience=conf.train.patience,
        min_lr=conf.train.min_lr,
    )
    best_loss = np.inf
    epoch = 0

    ## load the chkp if there exist
    if args.load_ckpt:
        if os.path.isfile(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")):
            checkpoint = torch.load(
                os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"),
                map_location=args.device,
                weights_only=True
            )
            network.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            logger.info(
                "loaded state dict %s best_loss %f"
                % (os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"), best_loss)
            )
        else:
            logger.info("Can't find the checkpoint")

    for epoch_i in range(epoch, conf.train.max_epoches):
        train_loss = train(network, train_loader, conf.train, epoch_i, optimizer)
        test_loss = test(network, test_loader, conf.train)
        logger.info("train loss: %f test loss: %f" % (train_loss["loss"], test_loss["loss"]))

        # save the training meta information
        if not args.no_wandb:
            write_wandb("train", train_loss, epoch_i)
            write_wandb("test", test_loss, epoch_i)
            # Log Pearson R for test
            write_wandb("test/pearson_r_x", test_loss["pearson_r"][0], epoch_i)
            write_wandb("test/pearson_r_y", test_loss["pearson_r"][1], epoch_i)
            write_wandb("test/pearson_r_z", test_loss["pearson_r"][2], epoch_i)
            write_wandb("lr", scheduler.optimizer.param_groups[0]["lr"], epoch_i)

        if epoch_i % conf.train.eval_freq == conf.train.eval_freq - 1:
            eval_state = evaluate(network=network, loader=eval_loader, confs=conf.train)
            if not args.no_wandb:
                write_wandb('eval/loss', eval_state['loss']['loss'].mean(), epoch_i)
                write_wandb('eval/dist', eval_state['loss']['dist'].mean(), epoch_i)
                # Log Pearson R for eval
                write_wandb('eval/pearson_r_x', eval_state['pearson_r'][0], epoch_i)
                write_wandb('eval/pearson_r_y', eval_state['pearson_r'][1], epoch_i)
                write_wandb('eval/pearson_r_z', eval_state['pearson_r'][2], epoch_i)
            if "supervise_pos" in conf.train:
                logger.info("eval pos: %f "%(eval_state['loss']['loss'].mean()))
            else:
                logger.info("eval vel: %f "%(eval_state['loss']['loss'].mean()))

        scheduler.step(test_loss["loss"])
        if test_loss["loss"] < best_loss:
            best_loss = test_loss["loss"]
            save_best = True
        else:
            save_best = False

        save_ckpt(
            network,
            optimizer,
            scheduler,
            epoch_i,
            best_loss,
            conf,
            save_best=save_best,
        )

    wandb.finish()
