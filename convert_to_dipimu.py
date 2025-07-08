import numpy as np
import pickle
import os
import re
from scipy.spatial.transform import Rotation

def convert_totalcapture_to_dipimu(bone_file, sensors_file, output_dir):
    """Convert TotalCapture format to DIP-IMU compatible .pkl format
    
    Args:
        bone_file: Path to *_calib_imu_bone.txt file (orientation data)
        sensors_file: Path to *.sensors file (acceleration data)
        output_dir: Directory to save output pkl file
    
    Returns:
        Path to the created output file
    """
    
    # Define body part mapping to DIP-IMU indices
    # DIP-IMU uses specific sensor indices:
    # 0: left wrist, 1: right wrist, 2: left thigh, 3: right thigh, 4: head, 5: pelvis
    body_part_mapping = {
        'L_LowArm': 0,  # Left wrist
        'R_LowArm': 1,  # Right wrist
        'L_UpLeg': 2,   # Left thigh
        'R_UpLeg': 3,   # Right thigh
        'Head': 4,      # Head
        'Pelvis': 5,    # Pelvis
    }
    
    accelerations = parse_sensors_file(sensors_file)
    num_frames = len(accelerations)
    
    # Parse orientation data WITH frame count
    orientations = parse_bone_file(bone_file, num_frames)
    
    # Verify we have the same number of frames from both files
    if len(orientations) != len(accelerations):
        print(f"Warning: Mismatched frame counts - bone file: {len(orientations)}, sensors file: {len(accelerations)}")
        # Use the smaller count
        num_frames = min(len(orientations), len(accelerations))
        orientations = orientations[:num_frames]
        accelerations = accelerations[:num_frames]
    else:
        num_frames = len(orientations)
    
    # Create arrays for DIP-IMU format
    num_imus = 17  # DIP-IMU standard
    imu_acc = np.zeros((num_frames, num_imus, 3), dtype=np.float64)
    imu_ori = np.zeros((num_frames, num_imus, 3, 3), dtype=np.float64)
    
    # Fill arrays with available data
    for i in range(num_frames):
        for body_part, idx in body_part_mapping.items():
            # Add orientation if available
            if body_part in orientations[i]:
                quat = orientations[i][body_part]
                rot_matrix = Rotation.from_quat(quat).as_matrix()
                imu_ori[i, idx] = rot_matrix
            
            # Add acceleration if available
            if body_part in accelerations[i]:
                imu_acc[i, idx] = accelerations[i][body_part]
    
    # Create dummy ground truth data (placeholder)
    gt = np.zeros((num_frames, 72), dtype=np.float64)  # 72 = 24 joints × 3 DoF
    
    # Create combined IMU array (acc + flattened rotation matrix)
    imu_combined = np.zeros((num_frames, num_imus, 12), dtype=np.float64)
    for frame in range(num_frames):
        for imu_idx in range(num_imus):
            imu_combined[frame, imu_idx, :3] = imu_acc[frame, imu_idx]
            imu_combined[frame, imu_idx, 3:] = imu_ori[frame, imu_idx].flatten()
    
    # Create DIP-IMU compatible data structure
    dip_data = {
        'gt': gt,
        'imu_acc': imu_acc,
        'imu_ori': imu_ori,
        'imu': imu_combined,
        'sop': gt.copy(),
        'sip': gt.copy()
    }
    
    # Create output filename
    base_name = os.path.basename(bone_file).split('_calib')[0]
    output_path = os.path.join(output_dir, f"{base_name}.pkl")
    
    # Save as pickle file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(dip_data, f)
    
    print(f"Converted {bone_file} and {sensors_file} to {output_path}")
    return output_path

def parse_bone_file(bone_file, num_frames):
    """Parse TotalCapture bone file (.txt) containing static calibration quaternion data"""
    # Read the file
    with open(bone_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the single frame of orientation data
    sensor_data = {}
    
    # Skip the first line (number of sensors)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 5:  # Body part + 4 quaternion values
            body_part = parts[0]
            # Format is [qx, qy, qz, qw]
            quat = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            sensor_data[body_part] = quat
    
    # Replicate the static orientation data for all frames
    orientations = [sensor_data.copy() for _ in range(num_frames)]
    
    return orientations

def parse_sensors_file(sensors_file):
    """Parse TotalCapture sensors file (.sensors) containing quaternion + acceleration data"""
    accelerations = []
    
    with open(sensors_file, 'r') as f:
        lines = f.readlines()
    
    current_frame = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a frame number
        if re.match(r'^\d+$', line):
            if current_frame:
                accelerations.append(current_frame)
                current_frame = {}
            continue
        
        # Parse sensor data
        parts = line.split()
        if len(parts) >= 8:  # Body part + 4 quaternion + 3 acceleration values
            body_part = parts[0]
            acc = [float(parts[5]), float(parts[6]), float(parts[7])]
            current_frame[body_part] = acc
    
    # Add the last frame
    if current_frame:
        accelerations.append(current_frame)
    
    return accelerations

def batch_convert_totalcapture(input_dir, output_dir):
    """Convert all TotalCapture files in a directory"""
    # Find all bone files
    bone_files = [f for f in os.listdir(input_dir) if f.endswith('_calib_imu_bone.txt')]
    
    for bone_file in bone_files:
        # Find corresponding sensors file
        base_name = bone_file.split('_calib')[0]
        sensors_file = f"{base_name}_Xsens.sensors"
        
        if not os.path.exists(os.path.join(input_dir, sensors_file)):
            print(f"Warning: No matching sensors file for {bone_file}")
            continue
        
        # Convert the pair
        convert_totalcapture_to_dipimu(
            os.path.join(input_dir, bone_file),
            os.path.join(input_dir, sensors_file),
            output_dir
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert TotalCapture dataset to DIP-IMU format')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing TotalCapture files')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output pkl files')
    
    args = parser.parse_args()
    
    batch_convert_totalcapture(args.input_dir, args.output_dir)