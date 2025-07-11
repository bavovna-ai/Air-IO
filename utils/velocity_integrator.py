import torch
from torch import nn
from typing import Dict, Optional, Any

from utils import move_to


class Velocity_Integrator(nn.Module):
    """
    Integrates velocity over time to compute position.
    """
    def __init__(self, pos: torch.Tensor = torch.zeros(3)):
        """
        Initializes the Velocity_Integrator.

        Args:
            pos (torch.Tensor, optional): The initial position. Defaults to zeros.
        """
        super().__init__()
        self.register_buffer('pos',self._check(pos).clone(), persistent=False)
        
    def _check(self, obj: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Checks and adjusts the shape of the input tensor.

        Args:
            obj (Optional[torch.Tensor]): The input tensor.

        Returns:
            Optional[torch.Tensor]: The reshaped tensor.
        """
        if obj is not None:
            if len(obj.shape) == 2:
                obj = obj[None, ...]

            elif len(obj.shape) == 1:
                obj = obj[None, None, ...]
        return obj

    def forward(self, dt: torch.Tensor, 
                vel: torch.Tensor, 
                init_state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the integrator.

        Args:
            dt (torch.Tensor): The time intervals.
            vel (torch.Tensor): The velocities.
            init_state (Optional[Dict[str, torch.Tensor]], optional): The initial state. 
                                                                    Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: The resulting state dictionary.
        """
        dt = self._check(dt)
        if dt is None:
            raise ValueError("dt cannot be None")
        B = dt.shape[0]

        if init_state is None:
            init_state = {'pos': self.pos}
          
        predict = self.integrate(dt,vel,init_state)
        
        self.pos = predict['pos'][...,-1,:]
        
        return {**predict}
    
    def integrate(self, dt: torch.Tensor, 
                  vel: torch.Tensor, 
                  init_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Integrates velocity to get position increments.

        Args:
            dt (torch.Tensor): The time intervals.
            vel (torch.Tensor): The velocities.
            init_state (Dict[str, torch.Tensor]): The initial state.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the final position.
        """
        B, F = dt.shape[:2]
        
        dp = torch.zeros(B,1,3,dtype = dt.dtype, device = dt.device)
        # dp = torch.cat([dp, 0.5 * (vel[:,:F]+vel[:,1:])*dt ],dim=1)
       
        dp = torch.cat([dp, vel[:,:F]*dt],dim=1)
        incre_p = torch.cumsum(dp,dim=1)
              
        return {'pos': init_state['pos'] + incre_p[:,1:,:]}
   
 
# #################TEST#############
def integrate_pos(integrator: Velocity_Integrator, 
                  datainte: Dict[str, torch.Tensor], 
                  init: Dict[str, torch.Tensor],
                  dataset: Any, 
                  device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Integrates position using the Velocity_Integrator.

    Args:
        integrator (Velocity_Integrator): The velocity integrator instance.
        datainte (Dict[str, torch.Tensor]): A dictionary containing 'dt' and 'vel' tensors.
        init (Dict[str, torch.Tensor]): The initial state.
        dataset (Any): The dataset object.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the output state and error metrics.
    """
    out_state: Dict[str, torch.Tensor] = dict()
    vel_gt, poses_gt = [init['vel'][None,:]],[init['pos'][None,:]]
    state = integrator(
            dt=datainte['dt'][...,None],vel=datainte["vel"][None,...]
        )
    out_state['poses'] = state["pos"]
    out_state['net_vel'] = datainte["vel"][None,...]
    out_state['vel_gt'] = dataset.data['velocity']
    out_state['poses_gt'] = dataset.data['gt_translation']
    out_state['pos_dist'] = (out_state['poses'] -out_state['poses_gt'][1:,:]).norm(dim=-1)
    out_state['vel_dist'] = (datainte["vel"] -  out_state['vel_gt']).norm(dim=-1)
    out_state['vel_mag_dist'] = torch.abs(datainte["vel"].norm(dim=-1) - out_state['vel_gt'].norm(dim=-1))
    out_state['vel_error'] = (datainte["vel"] -  out_state['vel_gt'])
    return out_state  


if __name__ == "__main__":
    import argparse
    import os

    import torch
    import torch.utils.data as Data
    import tqdm
    from pyhocon import ConfigFactory

    from datasets import SeqDataset, imu_seq_collate
    from utils import CPU_Unpickler, move_to

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu", help="cuda or cpu, Default is cuda:0"
    )
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument(
        "--seqlen", type=int, default="200", help="the length of the segment"
    )
    parser.add_argument(
        "--dataconf",
        type=str,
        default="configs/datasets/SubTDataset/SubT_UGV1_final_half.conf",
        help="the configuration of the dataset",
    )

    args = parser.parse_args()
    print(("\n" * 3) + str(args) + ("\n" * 3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference

    net_result_path = os.path.join(args.exp, "net_output.pickle")
    if os.path.isfile(net_result_path):
        with open(net_result_path, "rb") as handle:
            inference_state_load = CPU_Unpickler(handle).load()
        for data_conf in dataset_conf.data_list:
            for data_name in data_conf.data_drive:
                dataset = SeqDataset(
                    data_conf.data_root,
                    data_name,
                    args.device,
                    name=data_conf.name,
                    duration=args.seqlen,
                    step_size=args.seqlen,
                    drop_last=False,
                    conf=dataset_conf,
                )
                loader = Data.DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    collate_fn=imu_seq_collate,
                    shuffle=False,
                    drop_last=False,
                )
                init = dataset.get_init_value()

                inference_state = inference_state_load[data_name]

                integrator = Velocity_Integrator(
                                    init['pos']).to(args.device).double()
                                
                outstate =integrate_pos(
                                    integrator, data_inte, init, loader,
                                    device=args.device)
                relative_outstate = calculate_rte(outstate, args.seqlen,args.seqlen)
                    
                print("==============Integration==============")
                print("outstate:")
                print("pos_err: ", outstate['pos_dist'].mean())
                print("vel_err: ", outstate['vel_dist'].mean())
                    
                print("relative_state:")
                print("pos_err: ", relative_outstate['pos_dist'].mean())
                   
                    
                    
