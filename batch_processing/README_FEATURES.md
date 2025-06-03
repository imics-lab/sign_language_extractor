# Sign Language Features Pipeline

This directory contains scripts for processing sign language videos using **invariant features** extracted from MediaPipe landmarks, instead of using raw landmark coordinates directly.

## Overview

The feature-based pipeline provides several advantages over using raw landmarks:

1. **Scale invariant**: Features are normalized to remove dependency on person size or distance from camera
2. **Position invariant**: Features are relative to body center, removing absolute position dependency
3. **Rotation resilient**: Some features are rotation-invariant or use relative angles
4. **More compact**: Extracted features can be more compact than full landmark sets
5. **Semantically meaningful**: Features capture relationships between body parts, joint angles, etc.

## Face Feature Modes

The pipeline now supports two face feature extraction modes:

### **Compact Face Mode (Default)**
- **Features**: 28 key facial landmarks (~92 total face features)
- **Landmarks**: Only the most important facial points for sign language
- **Advantages**: Much smaller feature vectors, faster processing, focuses on essential facial expressions
- **Best for**: Large vocabularies with limited training data (like 2000 words × 10 examples)

### **Full Face Mode**
- **Features**: All 468 facial landmarks (~1407 total face features)  
- **Landmarks**: Complete MediaPipe face mesh
- **Advantages**: Maximum facial detail, captures subtle expressions
- **Best for**: Smaller vocabularies with abundant training data

## Updated Feature Counts

### **Compact Face Mode**

| Feature Type | Components | Feature Count |
|--------------|------------|---------------|
| **Pose** | landmarks_normalized (33×3) + joint_angles (4) + limb_lengths (4) | **107 features** |
| **LeftHand** | landmarks_normalized (21×3) + finger_angles (15) + finger_distances (10) + hand_shape (2) | **90 features** |
| **RightHand** | landmarks_normalized (21×3) + finger_angles (15) + finger_distances (10) + hand_shape (2) | **90 features** |
| **CompactFace** | key_landmarks (28×3) + semantic_features (8) | **92 features** |
| **Relationships** | Variable: 1-8 features depending on detected parts | **~6 features** |
| **Metadata** | Completeness flags and scores | **6 features** |

**Total: ~391 features**

### **Full Face Mode**

| Feature Type | Components | Feature Count |
|--------------|------------|---------------|
| **Pose** | landmarks_normalized (33×3) + joint_angles (4) + limb_lengths (4) | **107 features** |
| **LeftHand** | landmarks_normalized (21×3) + finger_angles (15) + finger_distances (10) + hand_shape (2) | **90 features** |
| **RightHand** | landmarks_normalized (21×3) + finger_angles (15) + finger_distances (10) + hand_shape (2) | **90 features** |
| **FullFace** | landmarks_normalized (468×3) + semantic_features (8) + traditional_features (3) | **1415 features** |
| **Relationships** | Variable: 1-8 features depending on detected parts | **~6 features** |
| **Metadata** | Completeness flags and scores | **6 features** |

**Total: ~1714 features**

## Usage Examples

### **1. Batch Process Features (with face mode selection)**

```bash
# Compact face mode (recommended for your dataset)
python batch_process_features.py \
    --input_dir /path/to/raw/landmarks \
    --output_dir /path/to/features \
    --face_mode compact \
    --num_workers 4

# Full face mode (for datasets with abundant training data)
python batch_process_features.py \
    --input_dir /path/to/raw/landmarks \
    --output_dir /path/to/features \
    --face_mode full \
    --num_workers 4
```

### **2. Create NumPy Dataset from Features**

```bash
# For compact face mode - recommended feature combinations
python create_npy_features_dataset.py \
    --input_dir /path/to/features \
    --output_file datasets/compact_features.npy \
    --features Pose LeftHand RightHand CompactFace Metadata \
    --max_len 64

# For full face mode
python create_npy_features_dataset.py \
    --input_dir /path/to/features \
    --output_file datasets/full_features.npy \
    --features Pose LeftHand RightHand FullFace Metadata \
    --max_len 64
```

### **3. Create Train/Val/Test Split**

```bash
# Using metadata file (like WLASL)
python create_split_features_dataset.py \
    --features_dir /path/to/features \
    --metadata_file /path/to/nslt_100.json \
    --output_prefix datasets/wlasl_compact \
    --features Pose LeftHand RightHand CompactFace Metadata \
    --max_len 64 \
    --val_split 0.15 \
    --test_split 0.15
```

### **4. Test Feature Extraction Pipeline**

```bash
# Test with compact face mode
python sign_language_features.py --face_mode compact \
    full_test /path/to/sample.json

# Test with full face mode  
python sign_language_features.py --face_mode full \
    full_test /path/to/sample.json
```

## Recommendations for Your Dataset

Given your dataset characteristics (2000 classes × ~10 examples each), we **strongly recommend**:

### **Option 1: Optimal for Limited Data (RECOMMENDED)**
```bash
--features Pose LeftHand RightHand CompactFace Metadata
```
- **Total**: ~391 features per frame
- **Rationale**: Captures all essential information while keeping dimensionality manageable
- **Perfect for**: 2000 classes with 10 examples each

### **Option 2: Minimal Essential Features**
```bash
--features Pose LeftHand RightHand Metadata  
```
- **Total**: ~293 features per frame
- **Rationale**: Core signing information without facial expressions
- **Use if**: The optimal option still overfits

### **Option 3: Hands + Key Face Features Only**
```bash
--features LeftHand RightHand CompactFace Metadata
```
- **Total**: ~188 features per frame
- **Rationale**: Focus on hands and facial expressions (core of sign language)
- **Use if**: Body pose is not essential for your vocabulary

## Script Descriptions

1. **`sign_language_features.py`**: Core feature extractor with face mode selection
2. **`batch_process_features.py`**: Batch convert raw JSON files to feature files
3. **`create_npy_features_dataset.py`**: Create NumPy datasets from feature files  
4. **`create_split_features_dataset.py`**: Create train/val/test splits using metadata
5. **`face_features_compact.py`**: Standalone compact face feature extractor

## Benefits for Your Transformer Model

1. **Reduced overfitting**: ~391 features vs 1629 raw landmarks
2. **Better generalization**: Invariant features work across different people/cameras
3. **Faster training**: Smaller feature vectors mean faster forward/backward passes
4. **Semantic meaning**: Features represent meaningful concepts (angles, distances, etc.)
5. **Robust to missing data**: Graceful handling of partially visible landmarks

The compact face mode is specifically designed for scenarios like yours where you have many classes but limited examples per class!

## Pipeline Structure

### Original Raw Landmark Pipeline
```
Videos → batch_process_videos.py → Raw JSON → create_npy_dataset.py → NumPy datasets
                                           ↘ create_split_dataset.py → Train/Val/Test splits
```

### New Feature-based Pipeline
```
Videos → batch_process_videos.py → Raw JSON → batch_process_features.py → Feature JSON
                                                                        ↓
                                        NumPy datasets ← create_npy_features_dataset.py
                                        Train/Val/Test ← create_split_features_dataset.py
```

## Scripts

### 1. `sign_language_features.py`
The core feature extraction module that converts raw MediaPipe landmarks to invariant features.

**Key Features Extracted:**
- **Pose features**: Normalized landmarks, joint angles, limb length ratios
- **Hand features**: Normalized hand landmarks, finger angles, finger distances, hand shape
- **Face features**: Normalized face landmarks, eyebrow position, mouth shape
- **Relationships**: Hand-to-face distances, hand-to-chest positions, hand-to-hand relationships
- **Metadata**: Feature completeness scores, availability flags

**Usage:**
```bash
# Test single file processing
python sign_language_features.py raw_to_features input.json output_features.json

# Test full pipeline (raw → features → raw → compare)
python sign_language_features.py full_test input.json --temp_dir temp_test
```

### 2. `batch_process_features.py` (NEW)
Batch processes raw landmark JSON files to extract invariant features for all files in a directory.

**Usage:**
```bash
# Process all JSON files in a directory
python batch_process_features.py \
    --input_dir ./landmark_data \
    --output_dir ./feature_data \
    --num_workers 4 \
    --overwrite
```

**Parameters:**
- `--input_dir`: Directory containing raw landmark JSON files
- `--output_dir`: Directory to save feature JSON files  
- `--num_workers`: Number of parallel processing workers (default: all CPU cores)
- `--overwrite`: Overwrite existing feature files

### 3. `create_npy_features_dataset.py` (NEW)
Creates NumPy datasets from feature JSON files, similar to `create_npy_dataset.py` but for features.

**Usage:**
```bash
# Create dataset using only hand features
python create_npy_features_dataset.py \
    --input_dir ./feature_data \
    --output_file ./datasets/hands_features_len100.npy \
    --features LeftHand RightHand \
    --max_len 100 \
    --zero_pad

# Create dataset using all available features
python create_npy_features_dataset.py \
    --input_dir ./feature_data \
    --output_file ./datasets/all_features_len150.npy \
    --features Pose LeftHand RightHand Face Relationships Metadata \
    --max_len 150
```

**Available Feature Types:**
- `Pose`: Body pose features (landmarks, joint angles, limb ratios)
- `LeftHand`: Left hand features (landmarks, finger angles, shapes)
- `RightHand`: Right hand features  
- `Face`: Face features (landmarks, expression elements)
- `Relationships`: Spatial relationships between body parts
- `Metadata`: Feature availability and completeness scores

**Parameters:**
- `--input_dir`: Directory containing feature JSON files
- `--output_file`: Path to save the output .npy file
- `--features`: Which feature types to include
- `--max_len`: Maximum sequence length (frames)
- `--zero_pad`: Pad missing features with zeros instead of skipping

### 4. `create_split_features_dataset.py` (NEW)
Creates train/validation/test splits using feature data and metadata files.

**Usage:**
```bash
# Create train/val/test splits for WLASL dataset
python create_split_features_dataset.py \
    --feature_dir ./feature_data \
    --metadata_file ./nslt_100.json \
    --output_dir ./datasets/wlasl_features_100 \
    --features Pose LeftHand RightHand Relationships \
    --max_len 100 \
    --zero_pad
```

**Parameters:**
- `--feature_dir`: Directory containing feature JSON files
- `--metadata_file`: WLASL metadata JSON file (e.g., nslt_100.json)
- `--output_dir`: Directory to save train/val/test arrays
- `--features`: Which feature types to include
- `--max_len`: Maximum sequence length
- `--zero_pad`: Handle missing features with zero padding

**Output Files:**
- `X_train.npy`, `y_train.npy`: Training data and labels
- `X_val.npy`, `y_val.npy`: Validation data and labels  
- `X_test.npy`, `y_test.npy`: Test data and labels
- `dataset_info.json`: Dataset metadata and statistics

## Complete Workflow Example

Here's a complete example of processing sign language videos with the feature-based pipeline:

### Step 1: Extract Raw Landmarks (if not done already)
```bash
python batch_process_videos.py \
    --input_dir ./videos \
    --output_dir ./landmark_data
```

### Step 2: Extract Features from Landmarks
```bash
python batch_process_features.py \
    --input_dir ./landmark_data \
    --output_dir ./feature_data \
    --num_workers 8
```

### Step 3: Create Feature Datasets
```bash
# Option A: Create simple dataset without metadata splits
python create_npy_features_dataset.py \
    --input_dir ./feature_data \
    --output_file ./datasets/hand_features.npy \
    --features LeftHand RightHand \
    --max_len 100

# Option B: Create train/val/test splits using metadata
python create_split_features_dataset.py \
    --feature_dir ./feature_data \
    --metadata_file ./nslt_100.json \
    --output_dir ./datasets/wlasl_100_features \
    --features Pose LeftHand RightHand \
    --max_len 100
```

## Feature Types Details

### Pose Features
- **Normalized landmarks**: Body landmarks relative to torso center and scaled by torso size
- **Joint angles**: Angles between connected body parts (shoulders, elbows, etc.)
- **Limb length ratios**: Relative lengths of limbs normalized by reference measurements

### Hand Features  
- **Normalized landmarks**: Hand landmarks relative to wrist position
- **Finger angles**: Angles between finger segments
- **Finger distances**: Distances between fingertips and palm
- **Hand shape**: Overall hand shape descriptors

### Face Features
- **Normalized landmarks**: Face landmarks relative to face center  
- **Eyebrow position**: Relative eyebrow height
- **Mouth shape**: Mouth opening and shape descriptors

### Relationship Features
- **Hand-to-face distance**: Distance from hands to face center
- **Hand-to-chest position**: Position of hands relative to chest
- **Hand-to-hand relationships**: Distance and relative position between hands

### Metadata Features
- **Completeness flags**: Boolean indicators for which body parts were detected
- **Completeness score**: Overall quality score (0-1) for the frame
- **Fallback indicators**: Whether fallback methods were used

## Advantages of Feature-based Approach

1. **Robustness**: Less sensitive to camera position, person size, and orientation
2. **Semantic meaning**: Features capture meaningful relationships rather than raw coordinates  
3. **Efficiency**: Often more compact than full landmark representation
4. **Better generalization**: More likely to generalize across different recording setups
5. **Interpretability**: Individual features often have clear semantic meaning

## Performance Considerations

- **Processing time**: Feature extraction adds computational overhead but is highly parallelizable
- **Memory usage**: Feature files are typically smaller than raw landmark files
- **Quality**: Features handle missing or low-quality landmarks more gracefully
- **Compatibility**: Feature datasets are compatible with standard deep learning frameworks

## Troubleshooting

### Common Issues

1. **Missing feature files**: Ensure `batch_process_features.py` completed successfully
2. **Feature size mismatches**: Feature sizes can vary based on available data; use `--zero_pad` for consistency
3. **Low completeness scores**: Check video quality and MediaPipe detection confidence
4. **Import errors**: Ensure all scripts are in the same directory or adjust Python path

### Validation

```bash
# Test feature extraction on a single file
python sign_language_features.py full_test sample_landmarks.json

# Check feature dataset statistics
python -c "import numpy as np; data=np.load('dataset.npy'); print(f'Shape: {data.shape}, Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}')"
```

## Configuration Tips

### For Hand-focused Models
```bash
--features LeftHand RightHand Relationships
```

### For Full-body Models  
```bash
--features Pose LeftHand RightHand Face Relationships Metadata
```

### For Compact Models
```bash
--features Relationships Metadata  # Most compact representation
```

### For Robust Training
```bash
--zero_pad  # Handle missing data gracefully
``` 