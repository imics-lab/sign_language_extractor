import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os

# --- Landmark Constants (MUST match batch_process_videos.py and create_npy_dataset.py) ---
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21

POSE_FEATURES = NUM_POSE_LANDMARKS * 3  # 99 (x, y, z)
FACE_FEATURES = NUM_FACE_LANDMARKS * 3  # 1404
HAND_FEATURES = NUM_HAND_LANDMARKS * 3  # 63

# Calculate start and end indices for each landmark type in the flat array
# Order: Pose, Face, Left Hand, Right Hand (as per batch_process_videos.py)
POSE_START, POSE_END = 0, POSE_FEATURES
FACE_START, FACE_END = POSE_END, POSE_END + FACE_FEATURES
LH_START, LH_END = FACE_END, FACE_END + HAND_FEATURES
RH_START, RH_END = LH_END, LH_END + HAND_FEATURES

TOTAL_FEATURES_EXPECTED = RH_END # Should be 1629 (from batch_process_videos.py)

# Dictionary mapping landmark type names to their slice indices and feature count
LANDMARK_INFO = {
    "Pose": {"slice": slice(POSE_START, POSE_END), "features": POSE_FEATURES},
    "Face": {"slice": slice(FACE_START, FACE_END), "features": FACE_FEATURES},
    "LeftHand": {"slice": slice(LH_START, LH_END), "features": HAND_FEATURES},
    "RightHand": {"slice": slice(RH_START, RH_END), "features": HAND_FEATURES},
}

def main():
    parser = argparse.ArgumentParser(
        description="Load landmark JSON files based on metadata, select features, "
                    "pad/truncate sequences, split into train/val/test, and save as NumPy arrays."
    )
    parser.add_argument("--landmark_dir", required=True,
                        help="Directory containing the input JSON landmark files (output of batch_process_videos.py).")
    parser.add_argument("--metadata_file", required=True,
                        help="Path to the WLASL metadata JSON file (e.g., nslt_100.json).")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the output .npy files (X_train, y_train, etc.).")
    parser.add_argument("--landmarks", required=True, nargs='+', choices=LANDMARK_INFO.keys(),
                        help=f"Which landmarks to include. Choose one or more from: {list(LANDMARK_INFO.keys())}")
    parser.add_argument("--max_len", required=True, type=int,
                        help="Maximum sequence length (number of frames). Shorter sequences padded, longer ones truncated.")

    args = parser.parse_args()

    landmark_path = Path(args.landmark_dir)
    metadata_path = Path(args.metadata_file)
    output_path = Path(args.output_dir)
    selected_landmark_names = args.landmarks
    max_len = args.max_len

    if not landmark_path.is_dir():
        print(f"Error: Landmark directory not found: {landmark_path}")
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

    # --- Determine selected features (copied from create_npy_dataset.py) ---
    selected_slices = []
    total_selected_features = 0
    print("\nSelected Landmarks:")
    for name in selected_landmark_names:
        info = LANDMARK_INFO[name]
        selected_slices.append(info["slice"])
        total_selected_features += info["features"]
        print(f"- {name} ({info['features']} features, indices {info['slice'].start}-{info['slice'].stop-1})")

    if total_selected_features == 0:
        print("Error: No features selected based on landmark choices.")
        sys.exit(1)
    print(f"Total selected features per frame: {total_selected_features}")
    print(f"Target sequence length (max_len): {max_len}")

    # --- Load Metadata ---
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

    # --- Initialize lists for each data split ---
    data_splits = {
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []}
    }

    processed_count = 0
    skipped_count = 0

    # --- Process each entry in metadata ---
    for video_id, entry_data in metadata.items():
        landmark_json_file = landmark_path / f"{video_id}.json"

        if not landmark_json_file.exists():
            print(f"  Skipping '{video_id}': Landmark JSON file not found at '{landmark_json_file}'.")
            skipped_count += 1
            continue

        try:
            with open(landmark_json_file, 'r') as f:
                full_sequence_data = json.load(f) # List of frame_landmark_lists

            if not isinstance(full_sequence_data, list) or not full_sequence_data:
                print(f"  Warning: Skipping '{video_id}': Empty or invalid JSON landmark file.")
                skipped_count += 1
                continue

            # --- Frame Trimming based on metadata (action[1] and action[2]) ---
            # WLASL action array is [label_id, start_frame, end_frame] (1-based inclusive)
            start_frame_metadata = entry_data["action"][1]
            end_frame_metadata = entry_data["action"][2]

            # Convert to 0-based slicing for Python lists
            # start_idx is inclusive, end_idx is exclusive for Python slice
            start_idx = start_frame_metadata - 1
            end_idx = end_frame_metadata # Python slice up to, but not including, end_idx

            num_extracted_frames = len(full_sequence_data)

            # Validate and adjust frame indices against actual extracted frames
            start_idx = max(0, start_idx) # Ensure start_idx is not negative
            end_idx = min(num_extracted_frames, end_idx) # Ensure end_idx is within bounds

            if start_idx >= end_idx:
                print(f"  Warning: Skipping '{video_id}': Invalid frame range [{start_frame_metadata}-{end_frame_metadata}] "
                      f"results in empty sequence after considering {num_extracted_frames} extracted frames "
                      f"(effective range [{start_idx+1}-{end_idx}]).")
                skipped_count += 1
                continue
            
            # Get the relevant segment of frames
            sequence_data_segment = full_sequence_data[start_idx:end_idx]

            if not sequence_data_segment:
                print(f"  Warning: Skipping '{video_id}': Frame segment is empty after trimming "
                      f"([{start_frame_metadata}-{end_frame_metadata}] from {num_extracted_frames} frames).")
                skipped_count += 1
                continue
            
            # Convert to NumPy array for easier slicing
            sequence_array = np.array(sequence_data_segment, dtype=np.float32)

            # Verify feature count in the first frame
            if sequence_array.shape[1] != TOTAL_FEATURES_EXPECTED:
                print(f"  Warning: Skipping '{video_id}'. Unexpected number of features per frame. "
                      f"Expected {TOTAL_FEATURES_EXPECTED}, found {sequence_array.shape[1]}.")
                skipped_count += 1
                continue

            # Select the specified landmark features (columns)
            selected_features_sequence = np.concatenate(
                [sequence_array[:, s] for s in selected_slices], axis=1
            )
            assert selected_features_sequence.shape[1] == total_selected_features, \
                f"Feature selection error for {video_id}"

            # --- Pad or Truncate Sequence Length ---
            num_frames = selected_features_sequence.shape[0]
            processed_sequence = np.zeros((max_len, total_selected_features), dtype=np.float32)

            if num_frames == max_len:
                processed_sequence = selected_features_sequence
            elif num_frames > max_len:
                # Truncate (from the end)
                processed_sequence = selected_features_sequence[:max_len, :]
            else: # num_frames < max_len
                # Pad with zeros (at the end)
                processed_sequence[:num_frames, :] = selected_features_sequence
            
            # Get label and subset
            label = entry_data["action"][0] # The first element in 'action' is the class label
            subset = entry_data["subset"]

            if subset in data_splits:
                data_splits[subset]["X"].append(processed_sequence)
                data_splits[subset]["y"].append(label)
                processed_count += 1
            else:
                print(f"  Warning: Unknown subset '{subset}' for video_id '{video_id}'. Skipping.")
                skipped_count +=1

        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from landmark file: {landmark_json_file.name}")
            skipped_count += 1
        except Exception as e:
            print(f"  Error processing file {landmark_json_file.name}: {e}")
            skipped_count += 1

    print(f"\n--- Processing Summary ---")
    print(f"Successfully processed sequences: {processed_count}")
    print(f"Skipped sequences (missing files/errors/empty): {skipped_count}")

    # --- Stack sequences and save datasets ---
    for split_name, data in data_splits.items():
        num_samples = len(data["X"])
        print(f"\nProcessing '{split_name}' split with {num_samples} samples.")

        if num_samples > 0:
            X_data = np.stack(data["X"], axis=0)
            y_data = np.array(data["y"], dtype=np.int32) # Labels are integers

            print(f"  {split_name} X shape: {X_data.shape}")
            print(f"  {split_name} y shape: {y_data.shape}")

            try:
                X_output_file = output_path / f"X_{split_name}.npy"
                y_output_file = output_path / f"y_{split_name}.npy"
                np.save(X_output_file, X_data)
                np.save(y_output_file, y_data)
                print(f"  Saved {X_output_file.name} and {y_output_file.name}")
            except Exception as e:
                print(f"  Error saving NumPy arrays for {split_name} split to {output_path}: {e}")
        else:
            print(f"  No samples for '{split_name}' split. No .npy files will be created.")

    print("\nDataset creation complete.")

if __name__ == "__main__":
    main()

# Note: This script is designed to be run from the command line.
# Example usage:
# python create_split_dataset.py \
#     --landmark_dir data/WLASL_extracted_landmarks_100/ \
#     --metadata_file nslt_100.json \
#     --output_dir data/WLASL_npy_dataset_100_split/ \
#     --landmarks Pose LeftHand RightHand \
#     --max_len 80