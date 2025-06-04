#!/usr/bin/env python3
"""
Load feature JSON files, select specific feature components, pad/truncate sequences, 
and save as a NumPy array dataset for deep learning models.

This script works with the feature files created by batch_process_features.py,
which contain extracted invariant features instead of raw landmarks.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional

def extract_pose_features(features: Dict) -> Optional[np.ndarray]:
    """Extract pose-related features from the features dictionary."""
    if not features.get('pose_available', False):
        return None
    
    feature_vector = []
    
    # Add normalized landmarks if available
    if 'landmarks_normalized' in features:
        landmarks = np.array(features['landmarks_normalized']).flatten()
        feature_vector.extend(landmarks)
    
    # Add joint angles if available
    if 'joint_angles' in features:
        feature_vector.extend(features['joint_angles'])
    
    # Add limb lengths if available  
    if 'limb_lengths' in features:
        feature_vector.extend(features['limb_lengths'])
    
    return np.array(feature_vector) if feature_vector else None

def extract_hand_features(hand_data: Dict) -> Optional[np.ndarray]:
    """Extract hand-related features from the hand data dictionary."""
    if not isinstance(hand_data, dict) or not hand_data.get('landmarks_normalized'):
        return None
    
    feature_vector = []
    
    # Add normalized hand landmarks
    landmarks = np.array(hand_data['landmarks_normalized']).flatten()
    feature_vector.extend(landmarks)
    
    # Add finger angles if available
    if 'finger_angles' in hand_data:
        feature_vector.extend(hand_data['finger_angles'])
    
    # Add finger distances if available
    if 'finger_distances' in hand_data:
        feature_vector.extend(hand_data['finger_distances'])
    
    # Add hand shape features if available
    if 'hand_shape' in hand_data:
        feature_vector.extend(hand_data['hand_shape'])
    
    return np.array(feature_vector) if feature_vector else None

def extract_face_features(face_data: Dict) -> Optional[np.ndarray]:
    """Extract face-related features from the face data dictionary."""
    if not isinstance(face_data, dict) or not face_data.get('landmarks_normalized'):
        return None
    
    feature_vector = []
    mode = face_data.get('mode', 'compact')  # Default to compact for backward compatibility
    
    # Add normalized landmarks
    landmarks = np.array(face_data['landmarks_normalized']).flatten()
    feature_vector.extend(landmarks)
    
    # Add semantic features if available (common to both modes)
    if 'semantic_features' in face_data:
        semantic = face_data['semantic_features']
        feature_vector.extend([
            semantic.get('mouth_openness', 0.0),
            semantic.get('mouth_width', 0.0), 
            semantic.get('mouth_relative_x', 0.0),
            semantic.get('mouth_relative_y', 0.0),
            semantic.get('eyebrow_raise', 0.0),
            semantic.get('eyebrow_asymmetry', 0.0),
            semantic.get('eye_openness', 0.0),
            semantic.get('eye_asymmetry', 0.0)
        ])
    
    # Add traditional features if available (only in full mode)
    if mode == 'full' and 'traditional_features' in face_data:
        traditional = face_data['traditional_features']
        feature_vector.extend([
            traditional.get('eyebrow_position', 0.0),
            traditional.get('mouth_width', 0.0),
            traditional.get('mouth_height', 0.0)
        ])
    
    return np.array(feature_vector) if feature_vector else None

def extract_relationship_features(relationships: Dict) -> Optional[np.ndarray]:
    """Extract relationship features between body parts."""
    if not isinstance(relationships, dict):
        return None
    
    feature_vector = []
    
    # Add various relationship measurements
    for key, value in relationships.items():
        if isinstance(value, (int, float)):
            feature_vector.append(value)
        elif isinstance(value, list):
            feature_vector.extend(value)
    
    return np.array(feature_vector) if feature_vector else None

def extract_metadata_features(metadata: Dict) -> Optional[np.ndarray]:
    """Extract metadata features like completeness scores."""
    if not isinstance(metadata, dict):
        return None
    
    feature_vector = []
    
    # Add boolean flags as 0/1
    feature_vector.append(1.0 if metadata.get('pose_complete', False) else 0.0)
    feature_vector.append(1.0 if metadata.get('left_hand_complete', False) else 0.0)
    feature_vector.append(1.0 if metadata.get('right_hand_complete', False) else 0.0)
    feature_vector.append(1.0 if metadata.get('face_complete', False) else 0.0)
    feature_vector.append(1.0 if metadata.get('has_full_torso', False) else 0.0)
    
    # Add completeness score
    feature_vector.append(metadata.get('completeness_score', 0.0))
    
    return np.array(feature_vector)

# Feature type mapping
FEATURE_EXTRACTORS = {
    "Pose": lambda f: extract_pose_features(f.get('body', {})),
    "LeftHand": lambda f: extract_hand_features(f.get('left_hand', {})),
    "RightHand": lambda f: extract_hand_features(f.get('right_hand', {})),
    "Face": lambda f: extract_face_features(f.get('face', {})),
    "Relationships": lambda f: extract_relationship_features(f.get('relationships', {})),
    "Metadata": lambda f: extract_metadata_features(f.get('feature_metadata', {}))
}

def determine_feature_size(feature_files: List[Path], selected_features: List[str]) -> int:
    """
    Determine the size of feature vectors by processing a sample of files.
    This is needed because feature sizes can vary based on what's available.
    """
    print("Determining feature vector size from sample files...")
    
    sample_size = min(10, len(feature_files))
    total_size = 0
    valid_samples = 0
    
    for i, feature_file in enumerate(feature_files[:sample_size]):
        try:
            with open(feature_file, 'r') as f:
                feature_data = json.load(f)
            
            features_sequence = feature_data['features_sequence']
            if not features_sequence:
                continue
            
            # Get the first valid frame
            for frame_features in features_sequence:
                if 'error' in frame_features:
                    continue
                
                # Extract features for this frame
                frame_vector = []
                for feature_type in selected_features:
                    extractor = FEATURE_EXTRACTORS[feature_type]
                    extracted = extractor(frame_features)
                    if extracted is not None:
                        frame_vector.extend(extracted)
                
                if frame_vector:
                    current_size = len(frame_vector)
                    total_size += current_size
                    valid_samples += 1
                    print(f"  Sample {i+1}: {current_size} features")
                    break
                    
        except Exception as e:
            print(f"  Warning: Could not process sample file {feature_file.name}: {e}")
            continue
    
    if valid_samples == 0:
        raise ValueError("Could not determine feature size from any sample files")
    
    # Use the most common size (in case of slight variations)
    avg_size = int(total_size / valid_samples)
    print(f"Determined feature vector size: {avg_size} (from {valid_samples} samples)")
    return avg_size

def main():
    parser = argparse.ArgumentParser(
        description="Load feature JSON files, select features, pad/truncate sequences, and save as NumPy array."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing the input feature JSON files.")
    parser.add_argument("--output_file", required=True,
                        help="Path to save the output .npy file.")
    parser.add_argument("--features", required=True, nargs='+', 
                        choices=list(FEATURE_EXTRACTORS.keys()),
                        help=f"Which features to include. Choose from: {list(FEATURE_EXTRACTORS.keys())}")
    parser.add_argument("--max_len", required=True, type=int,
                        help="Maximum sequence length (frames). Shorter sequences padded, longer truncated.")
    parser.add_argument("--zero_pad", action="store_true",
                        help="Pad missing features with zeros instead of skipping frames")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_file)
    selected_features = args.features
    max_len = args.max_len

    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)

    if max_len <= 0:
        print(f"Error: --max_len must be a positive integer.")
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Selected features: {selected_features}")
    print(f"Target sequence length: {max_len}")

    # Find feature JSON files
    feature_files = sorted(list(input_path.glob('*_features.json')))
    if not feature_files:
        print(f"Error: No feature JSON files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(feature_files)} feature files to process")

    # Determine feature vector size
    try:
        feature_size = determine_feature_size(feature_files, selected_features)
    except Exception as e:
        print(f"Error determining feature size: {e}")
        sys.exit(1)

    all_processed_sequences = []
    processed_count = 0
    skipped_count = 0

    for i, feature_file in enumerate(feature_files):
        print(f"Processing ({i+1}/{len(feature_files)}): {feature_file.name}...")
        
        try:
            with open(feature_file, 'r') as f:
                feature_data = json.load(f)
            
            features_sequence = feature_data['features_sequence']
            if not features_sequence:
                print(f"  Warning: Skipping empty feature sequence")
                skipped_count += 1
                continue
            
            # Process each frame
            frame_vectors = []
            for frame_idx, frame_features in enumerate(features_sequence):
                if 'error' in frame_features:
                    if args.zero_pad:
                        frame_vectors.append(np.zeros(feature_size))
                    continue
                
                # Extract features for this frame
                frame_vector = []
                for feature_type in selected_features:
                    extractor = FEATURE_EXTRACTORS[feature_type]
                    extracted = extractor(frame_features)
                    if extracted is not None:
                        frame_vector.extend(extracted)
                    elif args.zero_pad:
                        # We need to know the expected size for this feature type
                        # For now, we'll skip this frame if any feature is missing
                        frame_vector = None
                        break
                
                if frame_vector is not None and len(frame_vector) > 0:
                    # Ensure consistent size (pad or truncate if needed)
                    if len(frame_vector) < feature_size:
                        frame_vector.extend([0.0] * (feature_size - len(frame_vector)))
                    elif len(frame_vector) > feature_size:
                        frame_vector = frame_vector[:feature_size]
                    
                    frame_vectors.append(np.array(frame_vector))
                elif args.zero_pad:
                    frame_vectors.append(np.zeros(feature_size))
            
            if not frame_vectors:
                print(f"  Warning: No valid frames found")
                skipped_count += 1
                continue
            
            # Convert to numpy array
            sequence_array = np.array(frame_vectors)
            
            # Pad or truncate sequence
            num_frames = sequence_array.shape[0]
            processed_sequence = np.zeros((max_len, feature_size), dtype=np.float32)
            
            if num_frames == max_len:
                processed_sequence = sequence_array
                print(f"  Sequence length matches max_len ({num_frames} frames)")
            elif num_frames > max_len:
                processed_sequence = sequence_array[:max_len, :]
                print(f"  Truncated sequence from {num_frames} to {max_len} frames")
            else:
                processed_sequence[:num_frames, :] = sequence_array
                print(f"  Padded sequence from {num_frames} to {max_len} frames")
            
            all_processed_sequences.append(processed_sequence)
            processed_count += 1
            
        except Exception as e:
            print(f"  Error processing file {feature_file.name}: {e}")
            skipped_count += 1

    if not all_processed_sequences:
        print("\nError: No valid sequences were processed")
        sys.exit(1)

    # Stack sequences into 3D array
    final_dataset = np.stack(all_processed_sequences, axis=0)

    print(f"\nSuccessfully processed {processed_count} sequences")
    print(f"Skipped {skipped_count} sequences")
    print(f"Final dataset shape: {final_dataset.shape}")

    # Save dataset
    try:
        np.save(output_path, final_dataset)
        print(f"Dataset saved successfully to: {output_path.resolve()}")
    except Exception as e:
        print(f"Error saving dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 