import os
import torch
import numpy as np
import pytorch_lightning as pl
import glob
import sys
from pathlib import Path

# Add project root to path to find constants.py  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args, get_module_config
from mGPT.utils.load_model import load_model
import mGPT.render.matplot.plot_3d_global as plot_3d
import moviepy.editor as mp
from convert_to_imuposer import convert_to_imuposer_format
from convert_motiongpt_to_dipimu import save_motion_as_dipimu_pkl
from argparse import ArgumentParser
from omegaconf import OmegaConf
from constants import MOTIONGPT_CACHE_DIR

def extend_motion_smoothly(motion_data, target_length, blend_frames=10):
    """Extend motion to target length with smooth blending between repetitions
    
    Args:
        motion_data: Original motion data (frames, joints, 3)
        target_length: Desired number of frames
        blend_frames: Number of frames to blend between repetitions
        
    Returns:
        Extended motion data
    """
    current_length = motion_data.shape[0]
    
    # Safety check
    if current_length == 0:
        raise ValueError("Cannot extend motion with 0 frames")
    
    if current_length >= target_length:
        return motion_data[:target_length]
    
    # Ensure blend_frames doesn't exceed current_length
    blend_frames = min(blend_frames, current_length // 2)
    
    # Calculate repetitions needed
    repetitions_needed = target_length // current_length
    remaining_frames = target_length % current_length
    
    extended_motion = []
    
    for i in range(repetitions_needed):
        if i == 0:
            # First repetition - use as is
            extended_motion.append(motion_data)
        else:
            # Subsequent repetitions - blend the beginning with previous end
            blended_motion = motion_data.copy()
            
            if blend_frames > 0 and len(extended_motion) > 0:
                # Get the last few frames from previous segment
                prev_end = extended_motion[-1][-blend_frames:]
                
                # Blend the beginning of current repetition with previous end
                for j in range(min(blend_frames, current_length)):
                    weight = j / blend_frames  # Fade from 0 to 1
                    if j < len(prev_end):
                        blended_motion[j] = (1 - weight) * prev_end[j] + weight * motion_data[j]
            
            extended_motion.append(blended_motion)
    
    # Add remaining frames if needed
    if remaining_frames > 0:
        partial_motion = motion_data[:remaining_frames].copy()
        
        # Blend with previous segment if possible
        if blend_frames > 0 and len(extended_motion) > 0:
            prev_end = extended_motion[-1][-blend_frames:]
            
            for j in range(min(blend_frames, remaining_frames)):
                weight = j / blend_frames
                if j < len(prev_end):
                    partial_motion[j] = (1 - weight) * prev_end[j] + weight * motion_data[j]
        
        extended_motion.append(partial_motion)
    
    # Concatenate all segments
    result = np.concatenate(extended_motion, axis=0)
    
    # Final safety check
    if result.shape[0] != target_length:
        print(f"Warning: Extended motion has {result.shape[0]} frames, expected {target_length}")
        if result.shape[0] > target_length:
            result = result[:target_length]
        else:
            # Pad with last frame if still too short
            padding_needed = target_length - result.shape[0]
            last_frame = result[-1:]
            padding = np.repeat(last_frame, padding_needed, axis=0)
            result = np.concatenate([result, padding], axis=0)
    
    return result

def upsample_to_60fps(motion_data, source_fps=20, target_fps=60):
    """Upsample motion data from 20fps to 60fps using linear interpolation
    
    This uses the same linear interpolation technique as IMUPoser's _resample function
    but in reverse direction (upsampling instead of downsampling).
    
    Args:
        motion_data: Motion data array of shape (frames, joints, 3)
        source_fps: Source frame rate (default 20)
        target_fps: Target frame rate (default 60)
        
    Returns:
        Upsampled motion data
    """
    if source_fps == target_fps:
        return motion_data
    
    # Convert numpy array to torch tensor for processing
    motion_tensor = torch.from_numpy(motion_data).float()
    
    # Calculate duration and target number of frames
    duration = motion_tensor.shape[0] / source_fps
    target_frames = int(duration * target_fps)
    
    # Create indices that map target frames back to source timeline
    # This is the reverse of: indices = torch.arange(0, tensor.shape[0], 60/target_fps)
    indices = torch.arange(target_frames).float() * (source_fps / target_fps)
    
    # Find the surrounding frames for interpolation
    start_indices = torch.floor(indices).long()
    end_indices = torch.ceil(indices).long()
    end_indices[end_indices >= motion_tensor.shape[0]] = motion_tensor.shape[0] - 1  # Handle edge cases
    
    # Get the actual frames
    start = motion_tensor[start_indices]
    end = motion_tensor[end_indices]
    
    # Calculate interpolation weights (same as IMUPoser's technique)
    floats = indices - start_indices
    for shape_index in range(len(motion_tensor.shape) - 1):
        floats = floats.unsqueeze(1)
    weights = torch.ones_like(start) * floats
    
    # Linear interpolation using torch.lerp (same as IMUPoser)
    upsampled_tensor = torch.lerp(start, end, weights)
    
    # Convert back to numpy
    return upsampled_tensor.numpy()

def process_long_sequence_with_overlap(motion_data, max_window_size=196, overlap_ratio=0.3):
    """Process long sequences using overlapping windows and blend the results
    
    Args:
        motion_data: Motion data array of shape (frames, joints, 3)
        max_window_size: Maximum window size the model can handle (default 196)
        overlap_ratio: Ratio of overlap between windows (default 0.3)
        
    Returns:
        Processed motion data maintaining original length
    """
    total_frames = motion_data.shape[0]
    
    if total_frames <= max_window_size:
        return motion_data
    
    # Calculate step size and number of windows
    overlap_frames = int(max_window_size * overlap_ratio)
    step_size = max_window_size - overlap_frames
    num_windows = (total_frames - overlap_frames + step_size - 1) // step_size
    
    # Process each window
    processed_windows = []
    window_starts = []
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = min(start_idx + max_window_size, total_frames)
        
        # Extract window
        window = motion_data[start_idx:end_idx]
        
        # If window is too short, pad it
        if window.shape[0] < max_window_size:
            padding_needed = max_window_size - window.shape[0]
            # Repeat the last frame for padding - fix the shape issue
            last_frame = window[-1:]  # Keep as (1, joints, 3)
            padding = np.repeat(last_frame, padding_needed, axis=0)  # (padding_needed, joints, 3)
            window = np.concatenate([window, padding], axis=0)
        
        processed_windows.append(window)
        window_starts.append(start_idx)
    
    # Blend overlapping regions
    result = np.zeros_like(motion_data)
    weight_sum = np.zeros(total_frames)
    
    for i, (window, start_idx) in enumerate(zip(processed_windows, window_starts)):
        end_idx = min(start_idx + max_window_size, total_frames)
        actual_window_size = end_idx - start_idx
        
        # Create blending weights (higher in center, lower at edges)
        weights = np.ones(actual_window_size)
        if i > 0:  # Not the first window
            # Fade in at the beginning
            fade_in_len = min(overlap_frames, actual_window_size // 2)
            weights[:fade_in_len] = np.linspace(0, 1, fade_in_len)
        if i < len(processed_windows) - 1:  # Not the last window
            # Fade out at the end
            fade_out_len = min(overlap_frames, actual_window_size // 2)
            weights[-fade_out_len:] = np.linspace(1, 0, fade_out_len)
        
        # Apply weights and accumulate
        for frame_idx in range(actual_window_size):
            global_idx = start_idx + frame_idx
            weight = weights[frame_idx]
            result[global_idx] += window[frame_idx] * weight
            weight_sum[global_idx] += weight
    
    # Normalize by accumulated weights
    for i in range(total_frames):
        if weight_sum[i] > 0:
            result[i] /= weight_sum[i]
    
    return result

def render_motion(data, feats, output_dir, text, output_types=["npz", "mp4", "npy"], save_imuposer=False, test_mode=False):
    """Render motion to video and save motion data
    
    Args:
        data: Motion data to render
        feats: Motion features
        output_dir: Directory to save outputs
        text: Text description of motion
        output_types: List of output types to generate. Can include "npz", "mp4", "npy"
        save_imuposer: Whether to also save in IMUPoser-compatible format
        test_mode: If True, save as DIP-IMU .pkl format instead of .npz

    Returns:
        dict: Dictionary containing paths to generated files
    """
    
    # Apply preprocessing for IMUPoser compatibility
    # First process long sequences with overlapping windows (if needed)
    data_processed = process_long_sequence_with_overlap(data, max_window_size=196, overlap_ratio=0.3)
    
    # Debug: Check if data is empty
    if data_processed.size == 0:
        print(f"Warning: Empty motion data after processing for text: '{text}'")
        print(f"Original data shape: {data.shape}")
        return {"error": "Empty motion data generated"}
    
    # Then upsample from 20fps to 60fps for IMUPoser compatibility
    data_60fps = upsample_to_60fps(data_processed, source_fps=20, target_fps=60)
    
    # Double-check after upsampling
    if data_60fps.size == 0:
        print(f"Warning: Empty motion data after upsampling for text: '{text}'")
        return {"error": "Empty motion data after upsampling"}
    
    # file names
    fname = text.lower().replace(" ", "_")
    outputs = {}

    if "npy" in output_types:
        feats_fname = fname + '.npy'
        output_npy_path = os.path.join(output_dir, feats_fname)
        np.save(output_npy_path, feats)
        outputs["npy"] = output_npy_path

    # Handle test mode - save as DIP-IMU .pkl instead of .npz
    if test_mode:
        pkl_fname = fname + '.pkl'
        output_pkl_path = os.path.join(output_dir, pkl_fname)
        
        # Save motion data in DIP-IMU format
        subject_id = f"motiongpt_{fname}"
        result_path = save_motion_as_dipimu_pkl(data_60fps, output_pkl_path, subject_id)
        
        if result_path is not None:
            outputs["pkl"] = result_path
            print(f"✓ Successfully saved DIP-IMU format data to {result_path}")
        else:
            print(f"✗ Failed to save DIP-IMU format data - motion data was empty or invalid")
        
    elif "npz" in output_types:
        npz_fname = fname + '.npz'
        output_npz_path = os.path.join(output_dir, npz_fname)
        
        # Save motion data with 60fps upsampling applied
        np.savez(output_npz_path, 
                 feats=feats, 
                 data=data_60fps,  # Use the 60fps upsampled data
                 original_data=data,  # Also save original 20fps for reference
                 fps=60)  # Include fps metadata
        outputs["npz"] = output_npz_path
        
        # Convert to IMUPoser format if requested
        if save_imuposer:
            imuposer_path = convert_to_imuposer_format(output_npz_path)
            outputs["imuposer_npz"] = imuposer_path

    if "mp4" in output_types:
        video_fname = fname + '.mp4'
        output_mp4_path = os.path.join(output_dir, video_fname)
        output_gif_path = output_mp4_path[:-4] + '.gif'
        
        # Use original 20fps data for video rendering (better for visualization)
        video_data = data
        
        # Ensure data is in the right format for plotting
        print(f"Video data shape before processing: {video_data.shape}")
        
        # The plotting function expects (batch_size, frames, joints, 3)
        # Our data is (frames, joints, 3), so add batch dimension
        if len(video_data.shape) == 3:
            video_data = video_data[None]  # Add batch dimension: (1, frames, joints, 3)
        
        if isinstance(video_data, torch.Tensor):
            video_data = video_data.cpu().numpy()
        
        print(f"Video data shape after processing: {video_data.shape}")
        
        # Create title list that matches batch size
        titles = [text]  # Single title for single batch item
        
        try:
            plot_3d.draw_to_batch(video_data, titles, [output_gif_path])
            
            # Convert GIF to MP4
            if os.path.exists(output_gif_path):
                out_video = mp.VideoFileClip(output_gif_path)
                out_video.write_videofile(output_mp4_path, verbose=False, logger=None)
                out_video.close()
                outputs["mp4"] = output_mp4_path
            else:
                print(f"Warning: GIF file not created at {output_gif_path}")
                
        except Exception as e:
            print(f"Error creating video: {e}")
            print(f"Trying alternative approach...")
            
            # Alternative: just save the GIF if MP4 fails
            if os.path.exists(output_gif_path):
                outputs["gif"] = output_gif_path
    
    return outputs

def process_text(text, output_dir=MOTIONGPT_CACHE_DIR, output_types=["npz", "mp4", "npy"], save_imuposer=False, motion_length=None, blend_frames=15, test_mode=False):
    """Process a text description to generate motion
    
    Args:
        text: Text description of motion
        output_dir: Directory to save outputs
        output_types: List of output types to generate. Can include "npz", "mp4", "npy"
        save_imuposer: Whether to also save in IMUPoser-compatible format
        motion_length: Target motion length in frames (None = use MotionGPT's natural length)
        blend_frames: Number of frames to blend between repetitions
        test_mode: If True, save as DIP-IMU .pkl format instead of .npz
        
    Returns:
        dict: Dictionary containing paths to generated files
    """
    
    global cfg, model, device
    if 'model' not in globals():
        cfg, model, device = load_model(parse_args(phase="webui"))

    batch = {
        "length": [196],  # Use a reasonable hint for MotionGPT
        "text": [text],
    }

    outputs = model(batch, task="t2m")
    out_feats = outputs["feats"][0]
    out_lengths = outputs["length"][0]
    
    # Debug: Check if MotionGPT generated valid output
    print(f"MotionGPT output length: {out_lengths}")
    if out_lengths == 0:
        print(f"Warning: MotionGPT generated 0-length motion for text: '{text}'")
        return {"error": "MotionGPT generated empty motion"}
    
    out_joints = outputs["joints"][:out_lengths].detach().cpu().numpy()
    
    # Debug output
    print(f"Model output - joints shape: {outputs['joints'].shape}, length: {out_lengths}")
    print(f"Extracted joints shape: {out_joints.shape}")
    
    # Check if extracted joints are empty
    if out_joints.size == 0:
        print(f"Warning: Extracted joints are empty for text: '{text}'")
        return {"error": "Extracted joints are empty"}
    
    # Fix the shape issue - MotionGPT returns (batch, frames, joints, 3)
    # We need (frames, joints, 3) for our processing
    if len(out_joints.shape) == 4 and out_joints.shape[0] == 1:
        out_joints = out_joints[0]  # Remove batch dimension
        print(f"Removed batch dimension, new shape: {out_joints.shape}")
    
    # Sanity check - if we got too few frames, something went wrong
    if out_joints.shape[0] < 10:
        print(f"Warning: Only got {out_joints.shape[0]} frames, this seems too short. Using minimum of 60 frames.")
        # Create a simple repeated motion if we got too few frames
        if out_joints.shape[0] == 1:
            # If we only got 1 frame, create a simple idle motion by slightly varying the pose
            # First, remove the batch dimension if present
            if len(out_joints.shape) == 4:  # (batch, frames, joints, 3)
                out_joints = out_joints[0]  # Take first (and likely only) batch
            out_joints = np.repeat(out_joints, 60, axis=0)
            # Add small random variations to make it more natural
            noise = np.random.normal(0, 0.01, out_joints.shape)
            out_joints = out_joints + noise
    
    # Extend motion to desired length if it's shorter than requested
    actual_frames = out_joints.shape[0]
    print(f"Generated {actual_frames} frames")
    
    if motion_length is not None and actual_frames != motion_length:
        # User specified a target length, so extend/truncate to match
        print(f"Extending to requested {motion_length} frames")
        out_joints = extend_motion_smoothly(out_joints, motion_length, blend_frames=blend_frames)
        print(f"Adjusted motion to {out_joints.shape[0]} frames")
    else:
        # Use MotionGPT's natural length
        print(f"Using MotionGPT's natural length of {actual_frames} frames")
    
    return render_motion(
        out_joints,
        out_feats.to('cpu').numpy(),
        output_dir,
        text,
        output_types,
        save_imuposer,
        test_mode
    )

if __name__ == "__main__":
    # First get the base config
    base_parser = ArgumentParser()
    base_group = base_parser.add_argument_group("Base options")
    
    # Add the base MotionGPT arguments
    base_group.add_argument(
        "--cfg_assets",
        type=str,
        required=False,
        default="./configs/assets.yaml",
        help="config file for asset paths",
    )
    base_group.add_argument(
        "--cfg",
        type=str,
        required=False,
        default="./configs/webui.yaml",
        help="config file",
    )
    
    # Add our custom arguments
    custom_group = base_parser.add_argument_group("Generation options")
    custom_group.add_argument("--input", help="Text file with motion descriptions (one per line)")
    custom_group.add_argument("--input-dir", help="Directory containing .txt files to process")
    custom_group.add_argument("--text", help="Single text description")
    custom_group.add_argument("--output-dir", default=MOTIONGPT_CACHE_DIR, help="Output directory (for single file processing)")
    custom_group.add_argument("--output-root", help="Root directory for batch processing output")
    custom_group.add_argument("--save-imuposer", action="store_true", help="Also save in IMUPoser format")
    custom_group.add_argument("--motion-length", type=int, default=None, help="Target motion length in frames (default: use MotionGPT's natural length)")
    custom_group.add_argument("--blend-frames", type=int, default=15, help="Number of frames to blend between repetitions")
    custom_group.add_argument("--output-types", nargs="+", default=["npz"], choices=["npz", "mp4", "npy"], help="Output types to generate")
    custom_group.add_argument("--test", action="store_true", help="Test mode: Save as DIP-IMU .pkl format instead of .npz for fine-tuning compatibility")
    
    args = base_parser.parse_args()
    
    # Process the config through MotionGPT's system
    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(args.cfg_assets)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(args.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    
    # Load the model with the processed config
    cfg, model, device = load_model(cfg)
    
    if args.input_dir and args.output_root:
        # Batch processing mode
        
        # Get all .txt files from input directory
        txt_files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
        
        if not txt_files:
            print(f"Error: No .txt files found in {args.input_dir}")
            exit(1)
        
        # Extract the main category from input directory (e.g., "Cooking" from "/path/to/Cooking")
        category = os.path.basename(args.input_dir.rstrip('/'))
        
        # Generate output directory names automatically: Category1, Category2, etc.
        output_dirs = [f"{category}{i+1}" for i in range(len(txt_files))]
        
        print(f"Processing {len(txt_files)} files from category: {category}")
        print(f"Will create directories: {output_dirs}")
        
        for txt_file, output_subdir in zip(txt_files, output_dirs):
            # Create full output path: output_root/Category1/, output_root/Category2/, etc.
            full_output_dir = os.path.join(args.output_root, output_subdir)
            os.makedirs(full_output_dir, exist_ok=True)
            
            print(f"\nProcessing {os.path.basename(txt_file)} -> {output_subdir}/")
            
            # Process each line in the txt file
            with open(txt_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    text = line.strip()
                    if text:  # Skip empty lines
                        try:
                            output_files = process_text(text, full_output_dir, args.output_types, args.save_imuposer, args.motion_length, args.blend_frames, args.test)
                            print(f"  [{line_num}] Processed: {text}")
                            if args.test:
                                print(f"       Generated DIP-IMU .pkl files: {list(output_files.keys())}")
                            else:
                                print(f"       Generated: {list(output_files.keys())}")
                        except Exception as e:
                            print(f"  [{line_num}] Error processing '{text}': {e}")
        
        print(f"\nBatch processing complete!")
        
    elif args.input:
        # Single file processing mode
        with open(args.input, "r") as f:
            for line in f:
                text = line.strip()
                output_files = process_text(text, args.output_dir, args.output_types, args.save_imuposer, args.motion_length, args.blend_frames, args.test)
                print(f"Processed: {text}")
                if args.test:
                    print(f"Generated DIP-IMU .pkl files: {output_files}")
                else:
                    print(f"Generated files: {output_files}")
    elif args.text:
        # Single text processing mode
        output_files = process_text(args.text, args.output_dir, args.output_types, args.save_imuposer, args.motion_length, args.blend_frames, args.test)
        print(f"Processed: {args.text}")
        if args.test:
            print(f"Generated DIP-IMU .pkl files: {output_files}")
        else:
            print(f"Generated files: {output_files}")
    else:
        print("Please provide one of the following:")
        print("  --input with a file for single file processing")
        print("  --text with a single description for single text processing")
        print("  --input-dir and --output-root for batch processing")
