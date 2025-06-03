#!/usr/bin/env python3
"""
Batch process raw landmark JSON files to extract invariant features.
This script processes all JSON files in an input directory and saves the extracted features.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial

# Import the feature extractor
from sign_language_features import SignLanguageFeatureExtractor, flat_array_to_landmarks_dict

def process_single_file(input_file: Path, output_dir: Path, extractor: SignLanguageFeatureExtractor) -> bool:
    """
    Process a single JSON file to extract features.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save output features file
        extractor: Configured SignLanguageFeatureExtractor instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load raw landmark data
        with open(input_file, 'r') as f:
            raw_sequence = json.load(f)
        
        if not isinstance(raw_sequence, list):
            print(f"  Error: {input_file.name} - Invalid format (not a list)")
            return False
        
        # Create output filename
        output_file = output_dir / f"{input_file.stem}_features.json"
        
        # Process the file using the configured extractor
        from sign_language_features import process_json_to_features_with_mode
        face_mode = 'compact' if extractor.use_compact_face else 'full'
        
        print(f"  Processing {input_file.name} -> {output_file.name} (face mode: {face_mode})")
        
        # Process frames
        features_sequence = []
        transform_params_sequence = []
        
        for frame_idx, flat_frame in enumerate(raw_sequence):
            try:
                if len(flat_frame) != 1629:  # Expected total features
                    continue
                
                # Convert flat array to landmarks dict
                from sign_language_features import flat_array_to_landmarks_dict
                landmarks_dict = flat_array_to_landmarks_dict(flat_frame)
                
                # Extract features
                features, transform_params = extractor.extract_invariant_features(landmarks_dict)
                
                features_sequence.append(features)
                transform_params_sequence.append(transform_params)
                
            except Exception as e:
                # Add dummy entry to maintain sequence
                features_sequence.append({'error': f'Frame {frame_idx} failed: {str(e)}'})
                transform_params_sequence.append({'error': True})
                continue
        
        # Save results
        output_data = {
            'features_sequence': features_sequence,
            'transform_params_sequence': transform_params_sequence,
            'metadata': {
                'num_frames': len(features_sequence),
                'original_num_frames': len(raw_sequence),
                'input_file': str(input_file),
                'face_mode': face_mode,
                'feature_extractor_version': '1.0'
            }
        }
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif hasattr(obj, 'item'):
                return obj.item()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=convert_numpy)
        
        return True
        
    except Exception as e:
        print(f"  Error processing {input_file.name}: {e}")
        return False

def process_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    input_file, output_dir = args
    extractor = SignLanguageFeatureExtractor()
    return process_single_file(input_file, output_dir, extractor)

def main():
    parser = argparse.ArgumentParser(
        description="Batch convert raw landmark JSON files to feature JSON files using invariant feature extraction."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing input JSON files with raw landmarks")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save extracted feature JSON files")
    parser.add_argument("--face_mode", choices=['compact', 'full'], default='compact',
                        help="Face feature extraction mode: compact (28 landmarks, ~92 features) or full (468 landmarks, ~1407 features)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--file_pattern", default="*.json",
                        help="File pattern to match input files (default: *.json)")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    input_files = sorted(list(input_path.glob(args.file_pattern)))
    if not input_files:
        print(f"Error: No files matching pattern '{args.file_pattern}' found in {input_path}")
        sys.exit(1)

    print(f"Found {len(input_files)} files to process")
    print(f"Face mode: {args.face_mode}")
    print(f"Using {args.num_workers} worker processes")
    print(f"Output directory: {output_path}")

    # Create feature extractor with specified face mode
    use_compact_face = (args.face_mode == 'compact')
    extractor = SignLanguageFeatureExtractor(use_compact_face=use_compact_face)

    # Process files
    if args.num_workers == 1:
        # Sequential processing
        successful = 0
        for i, input_file in enumerate(input_files):
            print(f"Processing ({i+1}/{len(input_files)}): {input_file.name}")
            if process_single_file(input_file, output_path, extractor):
                successful += 1
    else:
        # Parallel processing
        process_func = partial(process_single_file, output_dir=output_path, extractor=extractor)
        
        successful = 0
        with mp.Pool(args.num_workers) as pool:
            results = []
            for i, input_file in enumerate(input_files):
                result = pool.apply_async(process_func, (input_file,))
                results.append((input_file, result))
            
            for input_file, result in results:
                try:
                    if result.get(timeout=60):  # 60 second timeout per file
                        successful += 1
                        print(f"✓ Completed: {input_file.name}")
                    else:
                        print(f"✗ Failed: {input_file.name}")
                except Exception as e:
                    print(f"✗ Error: {input_file.name} - {e}")

    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {successful}/{len(input_files)} files")
    print(f"Feature files saved to: {output_path}")

if __name__ == "__main__":
    main() 