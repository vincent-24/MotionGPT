#!/usr/bin/env python
"""
Validation script to compare generated .pkl files with reference DIP_IMU files.

This script compares the data structure, shapes, and format of generated .pkl files
against known DIP_IMU reference files to ensure compatibility.

Usage:
    python validate_pkl_format.py --reference /path/to/reference.pkl --generated /path/to/generated.pkl
    python validate_pkl_format.py --reference-dir /path/to/dip_imu/s_09 --generated-dir /path/to/generated/s_11
"""

import pickle
import numpy as np
import os
import argparse
import glob
from pathlib import Path

def load_pkl_file(file_path):
    """Load a pickle file and return its data."""
    try:
        # Try loading with default encoding first
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data, None
    except UnicodeDecodeError:
        try:
            # If that fails, try with latin1 encoding (common for Python 2 pickles)
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            # Convert any byte strings to regular strings
            data = convert_bytes_to_strings(data)
            return data, None
        except Exception as e:
            try:
                # Last resort: try with bytes encoding
                with open(file_path, 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                # Convert any byte strings to regular strings
                data = convert_bytes_to_strings(data)
                return data, None
            except Exception as e2:
                return None, f"Failed to load with multiple encodings. Last error: {str(e2)}"
    except Exception as e:
        return None, str(e)

def convert_bytes_to_strings(data):
    """Convert byte strings to regular strings for Python 2 compatibility."""
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Convert byte string keys to regular strings
            if isinstance(key, bytes):
                key = key.decode('utf-8', errors='ignore')
            
            # Recursively convert values
            new_dict[key] = convert_bytes_to_strings(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_bytes_to_strings(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_bytes_to_strings(item) for item in data)
    elif isinstance(data, bytes):
        # Try to decode bytes to string, but preserve numpy arrays
        try:
            return data.decode('utf-8', errors='ignore')
        except:
            return data
    else:
        return data

def compare_data_structures(ref_data, gen_data, ref_file, gen_file):
    """Compare the data structures of two pickle files."""
    print(f"\n{'='*80}")
    print(f"COMPARING DATA STRUCTURES")
    print(f"{'='*80}")
    print(f"Reference file: {os.path.basename(ref_file)}")
    print(f"Generated file:  {os.path.basename(gen_file)}")
    print(f"{'='*80}")
    
    # Compare keys
    ref_keys = set(ref_data.keys()) if isinstance(ref_data, dict) else set()
    gen_keys = set(gen_data.keys()) if isinstance(gen_data, dict) else set()
    
    print(f"\nKEY COMPARISON:")
    print(f"Reference keys: {sorted(ref_keys)}")
    print(f"Generated keys: {sorted(gen_keys)}")
    
    # Check for missing or extra keys
    missing_keys = ref_keys - gen_keys
    extra_keys = gen_keys - ref_keys
    common_keys = ref_keys & gen_keys
    
    if missing_keys:
        print(f"❌ Missing keys in generated file: {sorted(missing_keys)}")
    if extra_keys:
        print(f"➕ Extra keys in generated file: {sorted(extra_keys)}")
    if common_keys:
        print(f"✅ Common keys: {sorted(common_keys)}")
    
    # Compare data shapes and types for common keys
    print(f"\nDATA COMPARISON:")
    issues = []
    
    for key in sorted(common_keys):
        ref_val = ref_data[key]
        gen_val = gen_data[key]
        
        print(f"\n--- Key: '{key}' ---")
        
        # Compare types
        ref_type = type(ref_val)
        gen_type = type(gen_val)
        print(f"Reference type: {ref_type}")
        print(f"Generated type:  {gen_type}")
        
        if ref_type != gen_type:
            issues.append(f"Type mismatch for '{key}': {ref_type} vs {gen_type}")
            continue
        
        # Compare shapes for numpy arrays
        if hasattr(ref_val, 'shape'):
            print(f"Reference shape: {ref_val.shape}")
            print(f"Generated shape:  {gen_val.shape}")
            
            if ref_val.shape != gen_val.shape:
                issues.append(f"Shape mismatch for '{key}': {ref_val.shape} vs {gen_val.shape}")
            else:
                print("✅ Shapes match")
            
            # Compare dtypes
            if hasattr(ref_val, 'dtype') and hasattr(gen_val, 'dtype'):
                print(f"Reference dtype: {ref_val.dtype}")
                print(f"Generated dtype:  {gen_val.dtype}")
                
                if ref_val.dtype != gen_val.dtype:
                    issues.append(f"Dtype mismatch for '{key}': {ref_val.dtype} vs {gen_val.dtype}")
                else:
                    print("✅ Dtypes match")
                    
            # Show data ranges for numerical data
            if np.issubdtype(ref_val.dtype, np.number) and np.issubdtype(gen_val.dtype, np.number):
                print(f"Reference range: [{np.min(ref_val):.6f}, {np.max(ref_val):.6f}]")
                print(f"Generated range:  [{np.min(gen_val):.6f}, {np.max(gen_val):.6f}]")
        
        # Compare string values
        elif isinstance(ref_val, str):
            print(f"Reference value: '{ref_val}'")
            print(f"Generated value:  '{gen_val}'")
            
            if ref_val != gen_val:
                print(f"⚠️  String values differ")
            else:
                print("✅ String values match")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    if not issues:
        print("✅ ALL CHECKS PASSED - Files have compatible structure!")
    else:
        print(f"❌ FOUND {len(issues)} ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    return len(issues) == 0

def validate_dipimu_format(pkl_path):
    """Validate that a file matches expected DIP_IMU format."""
    print(f"\nValidating DIP_IMU format for: {os.path.basename(pkl_path)}")
    
    data, error = load_pkl_file(pkl_path)
    if error:
        print(f"❌ Error loading file: {error}")
        return False
    
    # Check required keys
    required_keys = ['imu_acc', 'imu_ori', 'gt']
    optional_keys = ['description']
    
    missing_required = []
    for key in required_keys:
        if key not in data:
            missing_required.append(key)
    
    if missing_required:
        print(f"❌ Missing required keys: {missing_required}")
        return False
    
    print("✅ All required keys present")
    
    # Check data shapes and types
    issues = []
    
    # Check imu_acc
    imu_acc = data['imu_acc']
    if not isinstance(imu_acc, np.ndarray):
        issues.append(f"imu_acc should be numpy array, got {type(imu_acc)}")
    elif len(imu_acc.shape) != 3:
        issues.append(f"imu_acc should have 3 dimensions, got {len(imu_acc.shape)}")
    elif imu_acc.shape[2] != 3:
        issues.append(f"imu_acc last dimension should be 3, got {imu_acc.shape[2]}")
    else:
        print(f"✅ imu_acc shape: {imu_acc.shape} (frames={imu_acc.shape[0]}, imus={imu_acc.shape[1]}, xyz=3)")
    
    # Check imu_ori
    imu_ori = data['imu_ori']
    if not isinstance(imu_ori, np.ndarray):
        issues.append(f"imu_ori should be numpy array, got {type(imu_ori)}")
    elif len(imu_ori.shape) != 4:
        issues.append(f"imu_ori should have 4 dimensions, got {len(imu_ori.shape)}")
    elif imu_ori.shape[2:] != (3, 3):
        issues.append(f"imu_ori last two dimensions should be (3,3), got {imu_ori.shape[2:]}")
    else:
        print(f"✅ imu_ori shape: {imu_ori.shape} (frames={imu_ori.shape[0]}, imus={imu_ori.shape[1]}, 3x3 rotation)")
    
    # Check gt
    gt = data['gt']
    if not isinstance(gt, np.ndarray):
        issues.append(f"gt should be numpy array, got {type(gt)}")
    else:
        print(f"✅ gt shape: {gt.shape}")
    
    # Check frame consistency
    if 'imu_acc' in data and 'imu_ori' in data:
        if data['imu_acc'].shape[0] != data['imu_ori'].shape[0]:
            issues.append(f"Frame count mismatch: imu_acc={data['imu_acc'].shape[0]}, imu_ori={data['imu_ori'].shape[0]}")
        
        if data['imu_acc'].shape[1] != data['imu_ori'].shape[1]:
            issues.append(f"IMU count mismatch: imu_acc={data['imu_acc'].shape[1]}, imu_ori={data['imu_ori'].shape[1]}")
    
    # Check optional description
    if 'description' in data:
        desc = data['description']
        if isinstance(desc, str):
            print(f"✅ description: '{desc}' (length: {len(desc)})")
        else:
            print(f"⚠️  description present but not a string: {type(desc)}")
    else:
        print("ℹ️  No description field (optional)")
    
    if issues:
        print(f"❌ Found {len(issues)} format issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ File passes DIP_IMU format validation!")
        return True

def find_reference_files(reference_dir):
    """Find reference DIP_IMU files in a directory."""
    pkl_files = glob.glob(os.path.join(reference_dir, "*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in reference directory: {reference_dir}")
        return []
    return sorted(pkl_files)

def debug_pkl_contents(data, file_path):
    """Debug function to show the contents of a pickle file."""
    print(f"\nDEBUG: Contents of {os.path.basename(file_path)}:")
    print(f"  Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"    {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"      Shape: {value.shape}")
                print(f"      Dtype: {value.dtype}")
    elif hasattr(data, 'shape'):
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
    else:
        print(f"  Value: {str(data)[:100]}...")

def main():
    parser = argparse.ArgumentParser(description='Validate generated .pkl files against DIP_IMU format')
    parser.add_argument('--reference', type=str, help='Path to reference DIP_IMU .pkl file')
    parser.add_argument('--generated', type=str, help='Path to generated .pkl file')
    parser.add_argument('--reference-dir', type=str, help='Directory containing reference DIP_IMU files')
    parser.add_argument('--generated-dir', type=str, help='Directory containing generated files')
    parser.add_argument('--validate-only', action='store_true', help='Only validate format, skip comparison')
    parser.add_argument('--debug', action='store_true', help='Show debug information about file contents')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Just validate the generated files
        if args.generated:
            validate_dipimu_format(args.generated)
        elif args.generated_dir:
            gen_files = glob.glob(os.path.join(args.generated_dir, "*.pkl"))
            if not gen_files:
                print(f"No .pkl files found in: {args.generated_dir}")
                return
            
            print(f"Found {len(gen_files)} files to validate")
            passed = 0
            for gen_file in sorted(gen_files):
                if validate_dipimu_format(gen_file):
                    passed += 1
            
            print(f"\n{'='*80}")
            print(f"VALIDATION COMPLETE: {passed}/{len(gen_files)} files passed")
            print(f"{'='*80}")
        return
    
    # Compare reference vs generated
    if args.reference and args.generated:
        # Single file comparison
        print("Loading reference file...")
        ref_data, ref_error = load_pkl_file(args.reference)
        if ref_error:
            print(f"Error loading reference file: {ref_error}")
            return
        
        if args.debug:
            debug_pkl_contents(ref_data, args.reference)
        
        print("Loading generated file...")
        gen_data, gen_error = load_pkl_file(args.generated)
        if gen_error:
            print(f"Error loading generated file: {gen_error}")
            return
        
        if args.debug:
            debug_pkl_contents(gen_data, args.generated)
        
        compare_data_structures(ref_data, gen_data, args.reference, args.generated)
        
    elif args.reference_dir and args.generated_dir:
        # Directory comparison
        ref_files = find_reference_files(args.reference_dir)
        gen_files = glob.glob(os.path.join(args.generated_dir, "*.pkl"))
        
        if not ref_files:
            print(f"No reference files found in: {args.reference_dir}")
            return
        
        if not gen_files:
            print(f"No generated files found in: {args.generated_dir}")
            return
        
        print(f"Found {len(ref_files)} reference files and {len(gen_files)} generated files")
        
        # Use first reference file as template
        ref_file = ref_files[0]
        print(f"Using reference template: {os.path.basename(ref_file)}")
        
        ref_data, ref_error = load_pkl_file(ref_file)
        if ref_error:
            print(f"Error loading reference file: {ref_error}")
            return
        
        # Compare all generated files against reference
        passed = 0
        for i, gen_file in enumerate(sorted(gen_files), 1):
            print(f"\n[{i}/{len(gen_files)}] Checking: {os.path.basename(gen_file)}")
            
            gen_data, gen_error = load_pkl_file(gen_file)
            if gen_error:
                print(f"❌ Error loading: {gen_error}")
                continue
            
            if compare_data_structures(ref_data, gen_data, ref_file, gen_file):
                passed += 1
        
        print(f"\n{'='*80}")
        print(f"BATCH VALIDATION COMPLETE: {passed}/{len(gen_files)} files passed")
        print(f"{'='*80}")
    
    else:
        print("Please provide either:")
        print("  --reference and --generated for single file comparison")
        print("  --reference-dir and --generated-dir for batch comparison")
        print("  --generated --validate-only for format validation only")

if __name__ == "__main__":
    main()
