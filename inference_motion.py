import os
import sys
import torch
from typing import Any, Dict

import torch.utils.data as Data
import argparse
import pickle

import tqdm
from utils import move_to, save_state
from pyhocon import ConfigFactory

from datasets import collate_fcs, SeqeuncesMotionDataset
from model import net_dict
from utils import *
from model.code import get_feature_names_from_config, get_feature_dim_from_config


def inference(network: torch.nn.Module, 
              loader: Data.DataLoader, 
              confs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    '''
    Correction inference.

    Saves the corrections generated from the network.

    Args:
        network (torch.nn.Module): The neural network model for inference.
        loader (Data.DataLoader): The data loader for inference data.
        confs (Dict[str, Any]): A dictionary of configurations.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the evaluated states.
    '''
    network.eval()
    evaluate_states: Dict[str, Any] = {}
    with torch.no_grad():
        inte_state = None
        for data, _, label in tqdm.tqdm(loader):
            data, label = move_to([data, label],  confs.device)
            rot = label['gt_rot'][:,:-1,:].Log().tensor()
            inte_state = network.forward(data, rot)
            inte_state['ts'] = network.get_label(data['ts'][...,None])[0]
            save_state(evaluate_states, inte_state)
           
        for k, v in evaluate_states.items():    
            evaluate_states[k] = torch.cat(v,  dim=-2)
    return evaluate_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/EuRoC/motion_body.conf', help='config file path')
    parser.add_argument('--load', type=str, default=None, help='path for specific model check point, Default is the best model')
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--seqlen', type=int, default=1000, help='window size.')
    parser.add_argument('--whole', default=True, action="store_true", help='estimate the whole seq')


    args = parser.parse_args(); print(args)
    
    # Process config to merge with defaults
    from model.code import process_config
    conf = ConfigFactory.parse_file(args.config)
    conf = process_config(conf)
    
    conf.train.device = args.device
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
    conf['device'] = args.device
    dataset_conf = conf.dataset.inference
    
    
    # Get feature information from model config
    feature_names = get_feature_names_from_config(conf.model)
    network = net_dict[conf.model.network](
        input_dim=conf.model.n_features,
        feature_names=dataset_conf.feature_names,
        hidden_channels=conf.model.feature_channels,
        kernel_sizes=conf.model.kernel_sizes if hasattr(conf.model, 'kernel_sizes') else [7, 7],
        strides=conf.model.strides if hasattr(conf.model, 'strides') else [3, 3],
        padding_num=conf.model.padding_num if hasattr(conf.model, 'padding_num') else 3,
        propcov=conf.model.propcov if hasattr(conf.model, 'propcov') else True,
    ).to(args.device).double()

    # Validate that dataset provides all features required by the model
    dataset_conf.validate_features(dataset_conf.feature_names)
    save_folder = os.path.join(conf.general.exp_dir, "evaluate")
    os.makedirs(save_folder, exist_ok=True)

    if args.load is None:
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
    elif os.path.exists(os.path.join(conf.general.exp_dir, "ckpt", args.load)):
        ckpt_path = os.path.join(conf.general.exp_dir, "ckpt", args.load)
    elif os.path.exists(args.load):
        ckpt_path = args.load
    else:
        raise KeyError(f"No model loaded {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device),weights_only=True)
    print("loaded state dict %s in epoch %i"%(ckpt_path, checkpoint["epoch"]))
    network.load_state_dict(checkpoint["model_state_dict"])
        
    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn = collate_fcs['base']

    cov_result, rmse = [], []
    net_out_result = {}
    evals = {}
    dataset_conf.data_list[0]["window_size"] = args.seqlen
    dataset_conf.data_list[0]["step_size"] = args.seqlen
    for data_conf in dataset_conf.data_list:
        for path in data_conf.data_drive:
            if args.whole:
                dataset_conf["mode"] = "inference"
            else:
                dataset_conf["mode"] = "infevaluate"
            dataset_conf["exp_dir"] = conf.general.exp_dir
            eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
            eval_loader = Data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, 
                                            shuffle=False, collate_fn=collate_fn, drop_last = False)
            inference_state = inference(network=network, loader = eval_loader, confs=conf.train)    
            if not "cov" in inference_state.keys():
                    inference_state["cov"] = torch.zeros_like(inference_state["net_vel"])         
            inference_state['ts'] = inference_state['ts']
            inference_state['net_vel'] = inference_state['net_vel'][0] #TODO: batch size != 1
            net_out_result[path] = inference_state

    net_result_path = os.path.join(conf.general.exp_dir, 'net_output.pickle')
    print("save netout, ", net_result_path)
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
