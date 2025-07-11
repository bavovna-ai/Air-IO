import os, argparse
import numpy as np
from numpy.typing import NDArray

def interp_xyz(time: NDArray[np.float64], 
               opt_time: NDArray[np.float64], 
               xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Interpolates XYZ coordinates to a new time base.

    Args:
        time (NDArray[np.float64]): The time array to interpolate to.
        opt_time (NDArray[np.float64]): The time array of the original XYZ data.
        xyz (NDArray[np.float64]): The XYZ coordinates to interpolate.

    Returns:
        NDArray[np.float64]: The interpolated XYZ coordinates.
    """
    intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
    intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
    intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])

    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return inte_xyz

def gradientvelo(xyz: NDArray[np.float64], 
                 imu_time: NDArray[np.float64], 
                 time: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculates velocity by differentiating position data.

    Args:
        xyz (NDArray[np.float64]): The XYZ coordinates.
        imu_time (NDArray[np.float64]): The IMU timestamps.
        time (NDArray[np.float64]): The timestamps for the XYZ data.

    Returns:
        NDArray[np.float64]: The calculated velocity.
    """
    inte_xyz = interp_xyz(imu_time, time, xyz)
    time_interval = imu_time[1:] - imu_time[:-1]
    time_interval = np.append(time_interval, time_interval.mean())
    velo_d = np.einsum('nd, n -> nd', np.gradient(inte_xyz, axis=0), 1/time_interval)

    return velo_d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/data/datasets/yuhengq/tumvio')
    parser.add_argument('--seq', nargs='+', default=["dataset-room1_512_16", "dataset-room2_512_16", "dataset-room3_512_16", 
                                                    "dataset-room4_512_16", "dataset-room5_512_16", "dataset-room6_512_16"])
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument('--load_ckpt', default=False, action="store_true")

    args = parser.parse_args(); print(args)

    for seq in args.seq:
        print(seq)
        gt_data = np.loadtxt(os.path.join(args.root, seq, "mav0/mocap0/data.csv"), dtype=float, delimiter=',')
        imu_data = np.loadtxt(os.path.join(args.root, seq, "mav0/imu0/data.csv"), dtype=float, delimiter=',')

        gt_time = gt_data[:,0]*1e-9
        xyz     = gt_data[:,1:4]

        imu_time = imu_data[:, 0]*1e-9
        acc      = imu_data[:, 4:]

        gt_velo = gradientvelo(xyz, imu_time, gt_time)
        to_save = np.concatenate([imu_time[:,None], gt_velo], 1)
        print("saving to ", os.path.join(args.root, seq, "mav0/mocap0/grad_velo.txt"))
        np.savetxt(os.path.join(args.root, seq, "mav0/mocap0/grad_velo.txt"), to_save)

        