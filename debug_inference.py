import os
import sys
import torch
import argparse
from pyhocon import ConfigFactory

print("Step 1: Importing modules...")
from datasets import collate_fcs, SeqeuncesMotionDataset
from model import net_dict
from utils import *

print("Step 2: Parsing arguments...")
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/Bavovna/motion_body_rot.conf', help='config file path')
parser.add_argument('--load', type=str, default=None, help='path for specific model check point, Default is the best model')
parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
parser.add_argument('--seqlen', type=int, default=1000, help='window size.')
parser.add_argument('--whole', default=True, action="store_true", help='estimate the whole seq')

args = parser.parse_args()
print(f"Args: {args}")

print("Step 3: Loading config...")
conf = ConfigFactory.parse_file(args.config)
print(f"Config loaded: {conf}")

print("Step 4: Setting up device and paths...")
conf.train.device = args.device
conf_name = os.path.split(args.config)[-1].split(".")[0]
conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
conf['device'] = args.device
print(f"Exp dir: {conf.general.exp_dir}")

print("Step 5: Getting dataset config...")
dataset_conf = conf.dataset.inference
print(f"Dataset config: {dataset_conf}")

print("Step 6: Creating network...")
network = net_dict[conf.train.network](conf.train).to(args.device).double()
print("Network created successfully")

print("Step 7: Loading checkpoint...")
if args.load is None:
    ckpt_path = os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt")
else:
    ckpt_path = os.path.join(conf.general.exp_dir, "ckpt", args.load)

print(f"Looking for checkpoint at: {ckpt_path}")
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device), weights_only=True)
    print(f"Loaded state dict {ckpt_path} in epoch {checkpoint['epoch']}")
    network.load_state_dict(checkpoint["model_state_dict"])
    print("Checkpoint loaded successfully")
else:
    print(f"ERROR: No model found at {ckpt_path}")
    sys.exit(1)

print("Step 8: Setting up collate function...")
if 'collate' in conf.dataset.keys():
    collate_fn = collate_fcs[conf.dataset.collate.type]
else:
    collate_fn = collate_fcs['base']
print(f"Collate function: {collate_fn}")

print("Step 9: Creating dataset...")
dataset_conf.data_list[0]["window_size"] = args.seqlen
dataset_conf.data_list[0]["step_size"] = args.seqlen
print("Dataset config updated")

for data_conf in dataset_conf.data_list:
    for path in data_conf.data_drive:
        print(f"Processing path: {path}")
        if args.whole:
            dataset_conf["mode"] = "inference"
        else:
            dataset_conf["mode"] = "infevaluate"
        dataset_conf["exp_dir"] = conf.general.exp_dir
        
        print("Creating SeqeuncesMotionDataset...")
        eval_dataset = SeqeuncesMotionDataset(data_set_config=dataset_conf, data_path=path, data_root=data_conf["data_root"])
        print("Dataset created successfully")
        
        print("Creating DataLoader...")
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, 
                                                shuffle=False, collate_fn=collate_fn, drop_last=False)
        print("DataLoader created successfully")
        
        print("Starting inference...")
        # Test with just one batch
        for i, (data, _, label) in enumerate(eval_loader):
            print(f"Processing batch {i}")
            if i >= 1:  # Just process first batch for testing
                break
        print("Inference test completed")

print("All steps completed successfully!") 