import os
import numpy as np
from pathlib import Path
import torch
import smplx
from tqdm import tqdm

MGPT_SMPL_MODEL_PATH = "../IMUPoser/src/imuposer/smpl"
SMPL_NEUTRAL_MODEL_PATH = MGPT_SMPL_MODEL_PATH + "/SMPL_NEUTRAL.pkl"

def convert_to_imuposer_format(input_path, output_dir=None):
    mgpt_data = np.load(input_path)     # load MotionGPT data   
    motion_data_3d_joints = mgpt_data['data'][0]    # extract 3D joints from the MotionGPT data

    N = motion_data_3d_joints.shape[0]   # N = number of frames
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # load SMPL model
    smpl_model = smplx.SMPL(
        model_path=MGPT_SMPL_MODEL_PATH, # path to SMPL model
        num_betas=10,               # number of betas, 10 shape parameters
        flat_hand_mean=True,        # for simplified hand
        num_pca_comps=0,            # no PCA for hands if you don't have them
    ).to(device)
    
    smpl_poses = torch.zeros(N, 24, 3, device=device, requires_grad=True) # 3D rotations for 24 joints (23 body joints + 1 root)
    smpl_betas = torch.zeros(10, device=device, requires_grad=True) # 10 shape parameters   
    smpl_global_orient = torch.zeros(N, 1, 3, device=device, requires_grad=True) # 3D rotation for global orientation
    smpl_transl = torch.zeros(N, 3, device=device, requires_grad=True) # 3D translation for root joint

    motion_data_torch = torch.from_numpy(motion_data_3d_joints).float().to(device) # convert to torch tensor
    optimizer = torch.optim.Adam([smpl_poses, smpl_betas, smpl_global_orient, smpl_transl], lr=0.01) # optimizer using Adam to minimize error between SMPL joints and input joints

    num_iterations = 200 # optimize for 200 iterations
    for i in tqdm(range(num_iterations), desc="Fitting SMPL poses"):
        optimizer.zero_grad() # clear gradients

        full_smpl_poses_input = torch.cat([smpl_global_orient, smpl_poses[:, 1:]], dim=1) # concatenate global orientation and body pose (23 joints excluding root)

        # forward pass through SMPL model to get predicted joints
        smpl_output = smpl_model(
            betas=smpl_betas.unsqueeze(0).repeat(N, 1), # shape (N, 10)
            body_pose=full_smpl_poses_input[:, 1:],   # body pose (23 joints excluding root)
            global_orient=full_smpl_poses_input[:, :1], # global orientation (root)
            transl=smpl_transl, # translation for root joint
            return_joints=True # return joints
        )
        
        predicted_joints = smpl_output.joints 
        loss = torch.nn.functional.mse_loss(predicted_joints[:, :22, :], motion_data_torch) # mean squared error between predicted and ground truth joints
        pose_prior_loss = torch.mean(smpl_poses[:, 1:] ** 2) * 0.001 # L2 on non-root poses
        loss += pose_prior_loss # add pose prior loss to total loss

        loss.backward() # backpropagate loss
        optimizer.step() # update parameters

        if (i+1) % 20 == 0: # print loss every 20 iterations
            tqdm.write(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.4f}")

    # detach tensors from conputation graph and move to CPU
    final_smpl_poses_axis_angle = smpl_poses.detach().cpu().numpy() 
    final_smpl_global_orient_axis_angle = smpl_global_orient.detach().cpu().numpy() 
    final_smpl_betas = smpl_betas.detach().cpu().numpy() 
    final_smpl_transl = smpl_transl.detach().cpu().numpy()

    full_pose = np.concatenate([final_smpl_global_orient_axis_angle, final_smpl_poses_axis_angle[:, 1:]], axis=1) # concatenate global orientation and body pose

    # prepare 52-dimensional pose for IMUPoser
    pose_52 = np.zeros((N, 52, 3), dtype=np.float32) # (N, 24, 3) axis angle representation for the SMPL format
    pose_52[:, :24, :] = full_pose # match the SMPL structure which gets truncated to 24 joints in IMUPoser
    pose_52[:, 37, :] = pose_52[:, 23, :] # copies the axis angle of joint 23 (right hand) to joint 37 (left hand) to match the SMPL format

    # prepare data for IMUPoser
    imuposer_data = {
        'trans': [final_smpl_transl.astype(np.float32)], # translation for root joint
        'gender': 'neutral', # gender
        'mocap_framerate': 60.0, # frame rate
        'betas': [final_smpl_betas.astype(np.float32)], # shape parameters
        'dmpls': [np.zeros((N, 8), dtype=np.float32)], # DMPLs
        'poses': [pose_52.astype(np.float32)], # pose
    }

    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    input_filename = os.path.basename(input_path)
    base_name = Path(input_filename).stem
    output_filename = f"imuposer_{base_name}_poses.npz"
    output_path = os.path.join(output_dir, output_filename)

    for filename in os.listdir(output_dir):
        if (
            filename.startswith(input_filename.replace(".npz", "")) # Check against base name
            and not filename.startswith("imuposer_")
            and filename.endswith((".npz", ".npy", ".mp4", ".gif"))
        ):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    np.savez(output_path, **imuposer_data)

    return output_path

def convert_directory(input_dir, output_dir=None):
    """
    Convert all MotionGPT .npz files in a directory to IMUPoser format
    
    Args:
        input_dir: Directory containing MotionGPT .npz files
        output_dir: Directory to save converted files. If None, saves in same directory
        
    Returns:
        list: Paths to all converted files
    """
    input_dir = Path(input_dir)
    converted_files = []
    
    for npz_file in input_dir.glob("*.npz"):
        if not npz_file.name.startswith("imuposer_"):  # Skip already converted files
            output_path = convert_to_imuposer_format(str(npz_file), output_dir)
            converted_files.append(output_path)
            print(f"Converted {npz_file.name} -> {os.path.basename(output_path)}")
    
    return converted_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MotionGPT .npz files to IMUPoser format")
    parser.add_argument("input", help="Input .npz file or directory")
    parser.add_argument("--output-dir", help="Output directory (optional)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if input_path.is_file():
        output_path = convert_to_imuposer_format(str(input_path), args.output_dir)
        print(f"Converted {input_path.name} -> {os.path.basename(output_path)}")
    else:
        converted_files = convert_directory(input_path, args.output_dir)
        print(f"\nConverted {len(converted_files)} files") 
