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

The pipeline supports two face feature extraction modes:

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

## Feature Counts

### **Compact Face Mode (Recommended)**

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
| **FullFace** | landmarks_normalized (468×3) + semantic_features (8) + traditional_features (3) | **1415 features** |
| *Other features same as compact mode* | | |

**Total: ~1714 features**

---

## Quick Start - Most Common Usage

### **Recommended Pipeline for 2000-class Dataset**

```bash
# Step 1: Extract features from raw landmarks
python ./batch_processing/batch_process_features.py \
    --input_dir ./data/landmark_data \
    --output_dir ./data/feature_data \
    --face_mode compact \
    --num_workers 8

# Step 2: Create train/val/test splits (MOST COMMON USAGE)
python ./batch_processing/create_split_features_dataset.py \
    --feature_dir ./data/feature_data \
    --metadata_file ./data/nslt_2000.json \
    --output_dir ./data/wlasl_2000_features_numpy \
    --features Pose LeftHand RightHand Face Relationships Metadata \
    --max_len 90 \
    --zero_pad
```

This creates train/val/test splits with all feature types for optimal sign language recognition performance.

---

## Complete Usage Guide

### **1. Batch Process Features**

```bash
# Compact face mode (recommended)
python batch_process_features.py \
    --input_dir /path/to/raw/landmarks \
    --output_dir /path/to/features \
    --face_mode compact \
    --num_workers 4

# Full face mode (for abundant training data)
python batch_process_features.py \
    --input_dir /path/to/raw/landmarks \
    --output_dir /path/to/features \
    --face_mode full \
    --num_workers 4
```

### **2. Create Train/Val/Test Splits (Recommended)**

```bash
# Standard WLASL dataset processing
python create_split_features_dataset.py \
    --feature_dir ./feature_data \
    --metadata_file ./nslt_100.json \
    --output_dir ./datasets/wlasl_100_features \
    --features Pose LeftHand RightHand Face Relationships Metadata \
    --max_len 90 \
    --val_split 0.15 \
    --test_split 0.15 \
    --zero_pad
```

### **3. Create Simple NumPy Dataset (Alternative)**

```bash
# All features dataset
python create_npy_features_dataset.py \
    --input_dir ./feature_data \
    --output_file ./datasets/all_features.npy \
    --features Pose LeftHand RightHand Face Relationships Metadata \
    --max_len 90 \
    --zero_pad

# Hands-only dataset
python create_npy_features_dataset.py \
    --input_dir ./feature_data \
    --output_file ./datasets/hands_only.npy \
    --features LeftHand RightHand \
    --max_len 90 \
    --zero_pad
```

### **4. Test Feature Extraction**

```bash
# Test with compact face mode
python sign_language_features.py --face_mode compact \
    full_test /path/to/sample.json

# Test with full face mode  
python sign_language_features.py --face_mode full \
    full_test /path/to/sample.json
```

---

## Pipeline Structure

### Complete Workflow
```
Videos → batch_process_videos.py → Raw JSON → batch_process_features.py → Feature JSON
                                                                        ↓
                                    NumPy arrays ← create_split_features_dataset.py
                                                 ← create_npy_features_dataset.py
```

---

## Script Reference

### **1. `sign_language_features.py`**
Core feature extraction module with face mode selection.

**Key Features:**
- Pose features: Normalized landmarks, joint angles, limb ratios
- Hand features: Finger angles, distances, hand shapes
- Face features: Key landmarks and semantic expressions
- Relationships: Spatial relationships between body parts
- Metadata: Completeness scores and quality indicators

### **2. `batch_process_features.py`**
Batch convert raw landmark JSON files to feature files.

**Parameters:**
- `--face_mode`: `compact` or `full` face feature extraction
- `--num_workers`: Parallel processing workers (default: 4)
- `--file_pattern`: File matching pattern (default: `*.json`)

### **3. `create_split_features_dataset.py`**
Create train/validation/test splits using metadata files.

**Key Parameters:**
- `--feature_dir`: Directory with feature JSON files
- `--metadata_file`: WLASL metadata file (e.g., nslt_2000.json)
- `--output_dir`: Output directory for numpy arrays
- `--features`: Feature types to include
- `--max_len`: Maximum sequence length
- `--zero_pad`: Handle missing features with zero padding

**Output Files:**
- `X_train.npy`, `y_train.npy`: Training data and labels
- `X_val.npy`, `y_val.npy`: Validation data and labels  
- `X_test.npy`, `y_test.npy`: Test data and labels
- `dataset_info.json`: Dataset metadata and statistics

### **4. `create_npy_features_dataset.py`**
Create simple NumPy datasets from feature files.

**Available Feature Types:**
- `Pose`: Body pose features
- `LeftHand` / `RightHand`: Hand features
- `Face`: Facial features
- `Relationships`: Spatial relationships
- `Metadata`: Quality indicators

---

## Recommendations by Dataset Size

### **Large Vocabulary (2000+ classes, limited examples)**
```bash
--features Pose LeftHand RightHand Face Metadata
# Total: ~385 features, optimal for limited training data
```

### **Medium Vocabulary (100-500 classes)**
```bash
--features Pose LeftHand RightHand Face Relationships Metadata
# Total: ~391 features, full feature set
```

### **Small Vocabulary (< 100 classes, abundant data)**
```bash
--features Pose LeftHand RightHand FullFace Relationships Metadata
# Total: ~1714 features, maximum detail (use --face_mode full)
```

### **Hands-focused Models**
```bash
--features LeftHand RightHand Relationships Metadata
# Total: ~188 features, focus on manual components
```

---

## Benefits of Feature-based Approach

1. **Robustness**: Less sensitive to camera position, person size, and orientation
2. **Semantic meaning**: Features capture meaningful relationships rather than raw coordinates  
3. **Efficiency**: More compact than full landmark representation
4. **Better generalization**: Works across different recording setups
5. **Interpretability**: Individual features have clear anatomical meaning

