"""
Convert MotionGPT motion data to DIP-IMU .pkl format

This module provides functions to convert MotionGPT motion data into the exact
format used by DIP-IMU datasets for fine-tuning IMUPoser models.
"""

import os
import torch
import numpy as np
import pickle
from pathlib import Path


def create_synthetic_imu_data(motion_joints, num_imus=17):
    """Create synthetic IMU data from joint positions
    
    Args:
        motion_joints: Joint positions tensor (frames, joints, 3)
        num_imus: Number of IMU sensors to simulate (default 17 for DIP-IMU)
        
    Returns:
        tuple: (acceleration_data, orientation_data)
    """
    frames, joints, _ = motion_joints.shape
    
    # Create synthetic accelerometer data
    # Use finite differences to approximate acceleration from position
    if frames > 2:
        # Second derivative approximation
        vel = torch.diff(motion_joints, dim=0)  # (frames-1, joints, 3)
        acc = torch.diff(vel, dim=0)  # (frames-2, joints, 3)
        
        # Pad to original length
        acc = torch.cat([acc[0:1], acc, acc[-1:]], dim=0)  # (frames, joints, 3)
    else:
        acc = torch.zeros_like(motion_joints)
    
    # Map joints to IMU locations for DIP-IMU format
    # DIP-IMU typically uses 17 IMU sensors placed on various body parts
    available_joints = min(joints, num_imus)
    
    # Create mapping from available joints to IMU sensors
    if available_joints >= num_imus:
        # Use first num_imus joints
        imu_joint_mapping = list(range(num_imus))
    else:
        # Repeat joints to fill num_imus sensors
        imu_joint_mapping = []
        for i in range(num_imus):
            imu_joint_mapping.append(i % available_joints)
    
    # Extract accelerations for IMU locations
    acc_data = torch.zeros(frames, num_imus, 3)
    for i, joint_idx in enumerate(imu_joint_mapping):
        acc_data[:, i, :] = acc[:, joint_idx, :]
    
    # Add some noise to make it more realistic
    noise = torch.randn_like(acc_data) * 0.1
    acc_data += noise
    
    # Create synthetic orientation data (rotation matrices)
    # Use joint velocities to estimate orientation
    if frames > 1:
        velocities = torch.diff(motion_joints, dim=0)
        velocities = torch.cat([velocities[0:1], velocities], dim=0)  # Pad to original length
        
        # Normalize velocities to get direction vectors
        vel_norm = torch.norm(velocities, dim=-1, keepdim=True)
        vel_norm = torch.clamp(vel_norm, min=1e-6)  # Avoid division by zero
        directions = velocities / vel_norm
        
        # Create rotation matrices from direction vectors (simplified)
        ori_data = torch.zeros(frames, num_imus, 3, 3)
        for i, joint_idx in enumerate(imu_joint_mapping):
            # Use direction as x-axis, create orthogonal y and z axes
            x_axis = directions[:, joint_idx, :]
            
            # Create a reference up vector
            up = torch.tensor([0.0, 1.0, 0.0]).expand_as(x_axis)
            
            # Create orthogonal vectors
            z_axis = torch.cross(x_axis, up, dim=-1)
            z_norm = torch.norm(z_axis, dim=-1, keepdim=True)
            z_norm = torch.clamp(z_norm, min=1e-6)
            z_axis = z_axis / z_norm
            
            y_axis = torch.cross(z_axis, x_axis, dim=-1)
            
            # Stack to form rotation matrices
            ori_data[:, i, :, 0] = x_axis
            ori_data[:, i, :, 1] = y_axis
            ori_data[:, i, :, 2] = z_axis
    else:
        ori_data = torch.eye(3).expand(frames, num_imus, 3, 3)
    
    return acc_data, ori_data


def convert_motion_to_dipimu_format(motion_joints, subject_id="synthetic"):
    """Convert motion joint positions to DIP-IMU raw format
    
    Args:
        motion_joints: Motion data as numpy array (frames, joints, 3)
        subject_id: Subject identifier for the synthetic data
        
    Returns:
        dict: Data in DIP-IMU raw format with keys ['gt', 'imu_acc', 'imu_ori', 'sop', 'sip', 'imu']
    """
    # Convert to torch tensor for processing
    motion_tensor = torch.from_numpy(motion_joints).float()
    frames, joints, _ = motion_tensor.shape
    
    print(f"Converting motion with shape: {motion_tensor.shape}")
    
    # 1. Ground truth pose data (gt)
    # Convert joint positions to pose parameters (72-dimensional)
    # For simplicity, flatten the joint positions and pad/truncate to 72 dimensions
    joint_flat = motion_tensor.reshape(frames, -1)  # (frames, joints*3)
    
    if joint_flat.shape[1] < 72:
        # Pad with zeros if we have fewer dimensions
        padding = torch.zeros(frames, 72 - joint_flat.shape[1])
        gt_data = torch.cat([joint_flat, padding], dim=1)
    else:
        # Truncate if we have more dimensions
        gt_data = joint_flat[:, :72]
    
    # 2. Create synthetic IMU data with 17 sensors (matching DIP-IMU format)
    num_imus = 17
    acc_data, ori_data = create_synthetic_imu_data(motion_tensor, num_imus=num_imus)
    
    # 3. Create sop and sip data (same as gt for now)
    sop_data = gt_data.clone()
    sip_data = gt_data.clone()
    
    # 4. Create combined IMU data (imu field)
    # This typically contains both acceleration and orientation data
    # Shape: (frames, num_imus, 12) - 3 for acc + 9 for rotation matrix
    imu_combined = torch.zeros(frames, num_imus, 12)
    imu_combined[:, :, :3] = acc_data  # First 3 components: acceleration
    imu_combined[:, :, 3:] = ori_data.reshape(frames, num_imus, 9)  # Next 9: flattened rotation matrix
    
    # Create the DIP-IMU raw format structure (numpy arrays, not tensors)
    dipimu_data = {
        'gt': gt_data.numpy(),           # Ground truth pose (frames, 72)
        'imu_acc': acc_data.numpy(),     # IMU accelerations (frames, 17, 3)
        'imu_ori': ori_data.numpy(),     # IMU orientations (frames, 17, 3, 3)
        'sop': sop_data.numpy(),         # SOP data (frames, 72)
        'sip': sip_data.numpy(),         # SIP data (frames, 72)
        'imu': imu_combined.numpy()      # Combined IMU data (frames, 17, 12)
    }
    
    return dipimu_data


def save_motion_as_dipimu_pkl(motion_data, output_path, subject_id="synthetic"):
    """Save motion data as a DIP-IMU format .pkl file
    
    Args:
        motion_data: Motion data as numpy array (frames, joints, 3)
        output_path: Path where to save the .pkl file
        subject_id: Subject identifier for the synthetic data
        
    Returns:
        str: Path to the saved .pkl file
    """
    # Check for empty motion data
    if motion_data.size == 0:
        print(f"Warning: Empty motion data provided for {subject_id}. Skipping file creation.")
        return None
    
    if len(motion_data.shape) != 3 or motion_data.shape[0] == 0:
        print(f"Warning: Invalid motion data shape {motion_data.shape} for {subject_id}. Expected (frames, joints, 3). Skipping file creation.")
        return None
    
    # Convert to DIP-IMU raw format
    dipimu_data = convert_motion_to_dipimu_format(motion_data, subject_id)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(dipimu_data, f)
    
    print(f"Saved DIP-IMU format data to: {output_path}")
    print(f"Data structure:")
    for key, value in dipimu_data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {type(value)} shape: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    return output_path
