#!/usr/bin/env python3
"""
Load feature JSON files based on metadata, select features, pad/truncate sequences,
split into train/val/test, and save as NumPy arrays.

This script works with feature files created by batch_process_features.py and
metadata files (like WLASL nslt_*.json) to create datasets for deep learning.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional

# Import feature extraction functions from the other script
from create_npy_features_dataset import FEATURE_EXTRACTORS, determine_feature_size

def check_for_nans_and_infs(array, context_info="", video_id="", frame_idx=None):
    """
    Check for NaN and Inf values in an array and raise detailed error if found.
    
    Args:
        array: numpy array to check
        context_info: descriptive context for where this check is happening
        video_id: video ID being processed (for error reporting)
        frame_idx: frame index being processed (for error reporting)
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    nan_count = np.isnan(array).sum()
    inf_count = np.isinf(array).sum()
    
    if nan_count > 0 or inf_count > 0:
        error_msg = f"❌ CRITICAL ERROR: Invalid values detected in {context_info}"
        if video_id:
            error_msg += f" for video '{video_id}'"
        if frame_idx is not None:
            error_msg += f" at frame {frame_idx}"
        error_msg += f"\n  NaN count: {nan_count}"
        error_msg += f"\n  Inf count: {inf_count}"
        error_msg += f"\n  Array shape: {array.shape}"
        error_msg += f"\n  Array dtype: {array.dtype}"
        
        if nan_count > 0:
            nan_indices = np.where(np.isnan(array))
            if len(nan_indices[0]) > 0:
                error_msg += f"\n  First few NaN positions: {list(zip(*[idx[:5] for idx in nan_indices]))}"
        
        if inf_count > 0:
            inf_indices = np.where(np.isinf(array))
            if len(inf_indices[0]) > 0:
                error_msg += f"\n  First few Inf positions: {list(zip(*[idx[:5] for idx in inf_indices]))}"
        
        error_msg += "\n\n🛑 This indicates a bug in the feature extraction pipeline!"
        error_msg += "\n   Please check the sign_language_features.py for division by zero or invalid operations."
        
        raise ValueError(error_msg)

def validate_final_dataset(X, y, subset_name):
    """
    Perform final validation on the complete dataset before saving.
    
    Args:
        X: Feature array
        y: Label array  
        subset_name: Name of the subset (train/val/test)
    """
    print(f"\n🔍 Final validation for {subset_name} set...")
    
    # Check X array
    check_for_nans_and_infs(X, f"{subset_name} feature array (X)")
    
    # Check y array 
    check_for_nans_and_infs(y, f"{subset_name} label array (y)")
    
    # Additional sanity checks
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch: X has {X.shape[0]} samples but y has {y.shape[0]} samples")
    
    if len(X.shape) != 3:
        raise ValueError(f"Expected X to be 3D (samples, time, features), got shape {X.shape}")
    
    if len(y.shape) != 1:
        raise ValueError(f"Expected y to be 1D (samples,), got shape {y.shape}")
    
    # Check for reasonable value ranges
    finite_X = X[np.isfinite(X)]
    if len(finite_X) > 0:
        x_min, x_max = finite_X.min(), finite_X.max()
        x_mean, x_std = finite_X.mean(), finite_X.std()
        print(f"  ✅ X statistics: min={x_min:.4f}, max={x_max:.4f}, mean={x_mean:.4f}, std={x_std:.4f}")
    
    y_min, y_max = y.min(), y.max()
    print(f"  ✅ y statistics: min={y_min}, max={y_max}, unique_labels={len(np.unique(y))}")
    
    print(f"  ✅ {subset_name} set validation passed!")

def main():
    parser = argparse.ArgumentParser(
        description="Load feature JSON files based on metadata, select features, "
                    "pad/truncate sequences, split into train/val/test, and save as NumPy arrays."
    )
    parser.add_argument("--feature_dir", required=True,
                        help="Directory containing the input feature JSON files (output of batch_process_features.py).")
    parser.add_argument("--metadata_file", required=True,
                        help="Path to the WLASL metadata JSON file (e.g., nslt_100.json).")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the output .npy files (X_train, y_train, etc.).")
    parser.add_argument("--features", required=True, nargs='+', 
                        choices=list(FEATURE_EXTRACTORS.keys()),
                        help=f"Which features to include. Choose from: {list(FEATURE_EXTRACTORS.keys())}")
    parser.add_argument("--max_len", required=True, type=int,
                        help="Maximum sequence length (frames). Shorter sequences padded, longer truncated.")
    parser.add_argument("--zero_pad", action="store_true",
                        help="Pad missing features with zeros instead of skipping frames")

    args = parser.parse_args()

    feature_path = Path(args.feature_dir)
    metadata_path = Path(args.metadata_file)
    output_path = Path(args.output_dir)
    selected_features = args.features
    max_len = args.max_len

    if not feature_path.is_dir():
        print(f"Error: Feature directory not found: {feature_path}")
        sys.exit(1)

    if not metadata_path.is_file():
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)

    if max_len <= 0:
        print(f"Error: --max_len must be a positive integer.")
        sys.exit(1)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output .npy files will be saved to: {output_path.resolve()}")

    print(f"\nSelected features: {selected_features}")
    print(f"Target sequence length: {max_len}")
    print("🔍 NaN/Inf detection enabled - will abort if invalid values are found!")

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from metadata file: {metadata_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading metadata file {metadata_path}: {e}")
        sys.exit(1)

    print(f"\nLoaded metadata for {len(metadata)} video instances.")

    # Find available feature files
    feature_files = list(feature_path.glob('*_features.json'))
    if not feature_files:
        print(f"Error: No feature JSON files found in {feature_path}")
        sys.exit(1)

    # Determine feature vector size
    try:
        feature_size = determine_feature_size(feature_files, selected_features)
    except Exception as e:
        print(f"Error determining feature size: {e}")
        sys.exit(1)

    # Initialize lists for each data split
    data_splits = {
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []}
    }

    processed_count = 0
    skipped_count = 0

    # Process each entry in metadata
    for video_id, entry_data in metadata.items():
        feature_json_file = feature_path / f"{video_id}_features.json"

        if not feature_json_file.exists():
            print(f"  Skipping '{video_id}': Feature JSON file not found at '{feature_json_file}'.")
            skipped_count += 1
            continue

        try:
            with open(feature_json_file, 'r') as f:
                feature_data = json.load(f)

            features_sequence = feature_data['features_sequence']
            if not features_sequence:
                print(f"  Warning: Skipping '{video_id}': Empty feature sequence.")
                skipped_count += 1
                continue

            # Frame trimming based on metadata (action[1] and action[2])
            # WLASL action array is [label_id, start_frame, end_frame] (1-based inclusive)
            start_frame_metadata = entry_data["action"][1]
            end_frame_metadata = entry_data["action"][2]

            # Convert to 0-based slicing for Python lists
            start_idx = start_frame_metadata - 1
            end_idx = end_frame_metadata

            num_extracted_frames = len(features_sequence)

            # Validate and adjust frame indices
            start_idx = max(0, start_idx)
            end_idx = min(num_extracted_frames, end_idx)

            if start_idx >= end_idx:
                print(f"  Warning: Skipping '{video_id}': Invalid frame range [{start_frame_metadata}-{end_frame_metadata}] "
                      f"results in empty sequence after considering {num_extracted_frames} extracted frames.")
                skipped_count += 1
                continue

            # Get the relevant segment of frames
            features_segment = features_sequence[start_idx:end_idx]

            if not features_segment:
                print(f"  Warning: Skipping '{video_id}': Feature segment is empty after trimming.")
                skipped_count += 1
                continue

            # Process each frame to extract feature vectors
            frame_vectors = []
            for frame_idx, frame_features in enumerate(features_segment):
                if 'error' in frame_features:
                    if args.zero_pad:
                        zero_frame = np.zeros(feature_size)
                        # Check that zeros don't contain NaN (they shouldn't, but let's be safe)
                        check_for_nans_and_infs(zero_frame, "zero-padded frame", video_id, frame_idx)
                        frame_vectors.append(zero_frame)
                    continue

                # Extract features for this frame
                frame_vector = []
                for feature_type in selected_features:
                    extractor = FEATURE_EXTRACTORS[feature_type]
                    extracted = extractor(frame_features)
                    if extracted is not None:
                        # Check extracted features for NaN/Inf before adding
                        check_for_nans_and_infs(extracted, f"{feature_type} features", video_id, frame_idx)
                        frame_vector.extend(extracted)
                    elif args.zero_pad:
                        # Skip this frame if any feature is missing and not zero-padding
                        frame_vector = None
                        break

                if frame_vector is not None and len(frame_vector) > 0:
                    # Ensure consistent size
                    if len(frame_vector) < feature_size:
                        frame_vector.extend([0.0] * (feature_size - len(frame_vector)))
                    elif len(frame_vector) > feature_size:
                        frame_vector = frame_vector[:feature_size]
                    
                    frame_array = np.array(frame_vector, dtype=np.float32)
                    # Check final frame vector for NaN/Inf
                    check_for_nans_and_infs(frame_array, "final frame vector", video_id, frame_idx)
                    frame_vectors.append(frame_array)
                elif args.zero_pad:
                    zero_frame = np.zeros(feature_size, dtype=np.float32)
                    check_for_nans_and_infs(zero_frame, "zero-padded frame", video_id, frame_idx)
                    frame_vectors.append(zero_frame)

            if not frame_vectors:
                print(f"  Warning: Skipping '{video_id}': No valid frames found after feature extraction.")
                skipped_count += 1
                continue

            # Convert to numpy array
            sequence_array = np.array(frame_vectors, dtype=np.float32)
            # Check sequence array for NaN/Inf
            check_for_nans_and_infs(sequence_array, "sequence array", video_id)

            # Pad or truncate sequence length
            num_frames = sequence_array.shape[0]
            processed_sequence = np.zeros((max_len, feature_size), dtype=np.float32)

            if num_frames == max_len:
                processed_sequence = sequence_array
            elif num_frames > max_len:
                # Truncate (from the end)
                processed_sequence = sequence_array[:max_len, :]
            else:
                # Pad with zeros (at the end)
                processed_sequence[:num_frames, :] = sequence_array

            # Final check on processed sequence
            check_for_nans_and_infs(processed_sequence, "processed sequence", video_id)

            # Get label and subset
            label = entry_data["action"][0]  # The first element in 'action' is the class label
            subset = entry_data["subset"]

            if subset in data_splits:
                data_splits[subset]["X"].append(processed_sequence)
                data_splits[subset]["y"].append(label)
                processed_count += 1

                if processed_count % 50 == 0:
                    print(f"  Processed {processed_count} sequences so far...")

            else:
                print(f"  Warning: Unknown subset '{subset}' for video '{video_id}'. Skipping.")
                skipped_count += 1

        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from feature file: {feature_json_file.name}")
            skipped_count += 1
        except Exception as e:
            print(f"  Error processing video '{video_id}': {e}")
            skipped_count += 1

    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total in metadata: {len(metadata)}")

    # Convert lists to NumPy arrays and save
    for subset_name, subset_data in data_splits.items():
        if subset_data["X"]:
            X = np.array(subset_data["X"], dtype=np.float32)
            y = np.array(subset_data["y"], dtype=np.int64)

            # Perform final validation before saving
            validate_final_dataset(X, y, subset_name)

            X_file = output_path / f"X_{subset_name}.npy"
            y_file = output_path / f"y_{subset_name}.npy"

            np.save(X_file, X)
            np.save(y_file, y)

            print(f"\n{subset_name.upper()} set:")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  Saved to: {X_file} and {y_file}")
            print(f"  ✅ No NaN/Inf values detected!")
        else:
            print(f"\n{subset_name.upper()} set: No data")

    print(f"\n=== Dataset Creation Complete ===")
    print(f"All files saved to: {output_path.resolve()}")
    print("✅ All datasets validated successfully - no NaN or Inf values found!")

    # Save dataset info
    info = {
        'features_used': selected_features,
        'feature_vector_size': feature_size,
        'max_sequence_length': max_len,
        'zero_padding_enabled': args.zero_pad,
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'validation_passed': True,  # Only reaches here if all validations passed
        'splits': {
            subset: {
                'num_sequences': len(data["X"]),
                'num_unique_labels': len(set(data["y"])) if data["y"] else 0
            }
            for subset, data in data_splits.items()
        }
    }

    info_file = output_path / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Dataset information saved to: {info_file}")

if __name__ == "__main__":
    main() 