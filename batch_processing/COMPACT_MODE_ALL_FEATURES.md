# Complete Feature Documentation: Compact Mode
## Pose + LeftHand + RightHand + CompactFace + Relationships + Metadata

This document provides detailed information about **all ~391 features** extracted when using the complete Compact Mode pipeline for sign language recognition. This documentation facilitates **geometry-aware data augmentations** that respect human anatomy, biomechanics, and sign language linguistics.

## Overview

**Total Features**: ~391
- **Pose**: 107 features (body landmarks, joint angles, limb lengths)
- **LeftHand**: 90 features (hand landmarks, finger angles, distances, shape)
- **RightHand**: 90 features (same structure as left hand)
- **CompactFace**: 92 features (key facial landmarks + semantic features)
- **Relationships**: 6 features (spatial relationships between body parts)
- **Metadata**: 6 features (completeness flags and quality scores)

**Coordinate Systems**: All landmarks normalized relative to appropriate reference frames (body center, wrist, nose) and scaled by anatomical reference measurements.

---

## FEATURE GROUP 1: POSE FEATURES (Features 0-106)
### Total: 107 features

### 1.1 Normalized Body Landmarks (Features 0-98)
**33 landmarks × 3 coordinates = 99 features**

#### Feature Order and MediaPipe Indices

| Feature Index | Landmark | MediaPipe ID | Anatomical Description |
|---------------|----------|--------------|------------------------|
| 0-2 | Nose | 0 | Tip of nose (reference point) |
| 3-5 | Left Eye Inner | 1 | Inner corner of left eye |
| 6-8 | Left Eye | 2 | Center of left eye |
| 9-11 | Left Eye Outer | 3 | Outer corner of left eye |
| 12-14 | Right Eye Inner | 4 | Inner corner of right eye |
| 15-17 | Right Eye | 5 | Center of right eye |
| 18-20 | Right Eye Outer | 6 | Outer corner of right eye |
| 21-23 | Left Ear | 7 | Left ear landmark |
| 24-26 | Right Ear | 8 | Right ear landmark |
| 27-29 | Mouth Left | 9 | Left corner of mouth |
| 30-32 | Mouth Right | 10 | Right corner of mouth |
| 33-35 | Left Shoulder | 11 | Left shoulder joint |
| 36-38 | Right Shoulder | 12 | Right shoulder joint |
| 39-41 | Left Elbow | 13 | Left elbow joint |
| 42-44 | Right Elbow | 14 | Right elbow joint |
| 45-47 | Left Wrist | 15 | Left wrist joint |
| 48-50 | Right Wrist | 16 | Right wrist joint |
| 51-53 | Left Pinky | 17 | Left hand pinky base |
| 54-56 | Right Pinky | 18 | Right hand pinky base |
| 57-59 | Left Index | 19 | Left hand index base |
| 60-62 | Right Index | 20 | Right hand index base |
| 63-65 | Left Thumb | 21 | Left hand thumb base |
| 66-68 | Right Thumb | 22 | Right hand thumb base |
| 69-71 | Left Hip | 23 | Left hip joint |
| 72-74 | Right Hip | 24 | Right hip joint |
| 75-77 | Left Knee | 25 | Left knee joint |
| 78-80 | Right Knee | 26 | Right knee joint |
| 81-83 | Left Ankle | 27 | Left ankle joint |
| 84-86 | Right Ankle | 28 | Right ankle joint |
| 87-89 | Left Heel | 29 | Left heel |
| 90-92 | Right Heel | 30 | Right heel |
| 93-95 | Left Foot Index | 31 | Left foot index toe |
| 96-98 | Right Foot Index | 32 | Right foot index toe |

#### Coordinate System
- **Origin**: Center of torso (midpoint between shoulders and hips)
- **Scale**: Normalized by shoulder width
- **Orientation**: Body-relative coordinate system (rotation-invariant)

### 1.2 Joint Angles (Features 99-102)
**4 joint angles in radians**

| Feature Index | Joint | Description | Range |
|---------------|-------|-------------|-------|
| 99 | Left Elbow Angle | Angle between upper arm and forearm | 0-π radians |
| 100 | Right Elbow Angle | Angle between upper arm and forearm | 0-π radians |
| 101 | Left Wrist Angle | Angle at wrist joint | 0-π radians |
| 102 | Right Wrist Angle | Angle at wrist joint | 0-π radians |

### 1.3 Normalized Limb Lengths (Features 103-106)
**4 limb length ratios**

| Feature Index | Limb | Description | Typical Range |
|---------------|------|-------------|---------------|
| 103 | Left Upper Arm | Shoulder to elbow length ratio | 0.8-1.2 |
| 104 | Left Forearm | Elbow to wrist length ratio | 0.8-1.2 |
| 105 | Right Upper Arm | Shoulder to elbow length ratio | 0.8-1.2 |
| 106 | Right Forearm | Elbow to wrist length ratio | 0.8-1.2 |

#### Anatomical Constraints for Pose Augmentation
- **Joint angle limits**: Elbows 0°-150°, wrists have limited rotation
- **Limb length stability**: Should vary minimally (±10%)
- **Bilateral symmetry**: Left-right limbs should be approximately symmetric
- **Shoulder width**: Critical reference measurement, minimal variation

---

## FEATURE GROUP 2: LEFT HAND FEATURES (Features 107-196)
### Total: 90 features

### 2.1 Normalized Hand Landmarks (Features 107-169)
**21 landmarks × 3 coordinates = 63 features**

#### Feature Order and MediaPipe Hand Indices

| Feature Index | Landmark | MediaPipe ID | Anatomical Description |
|---------------|----------|--------------|------------------------|
| 107-109 | Wrist | 0 | Wrist center (reference point) |
| 110-112 | Thumb CMC | 1 | Thumb carpometacarpal joint |
| 113-115 | Thumb MCP | 2 | Thumb metacarpophalangeal joint |
| 116-118 | Thumb IP | 3 | Thumb interphalangeal joint |
| 119-121 | Thumb Tip | 4 | Thumb fingertip |
| 122-124 | Index MCP | 5 | Index metacarpophalangeal joint |
| 125-127 | Index PIP | 6 | Index proximal interphalangeal joint |
| 128-130 | Index DIP | 7 | Index distal interphalangeal joint |
| 131-133 | Index Tip | 8 | Index fingertip |
| 134-136 | Middle MCP | 9 | Middle metacarpophalangeal joint |
| 137-139 | Middle PIP | 10 | Middle proximal interphalangeal joint |
| 140-142 | Middle DIP | 11 | Middle distal interphalangeal joint |
| 143-145 | Middle Tip | 12 | Middle fingertip |
| 146-148 | Ring MCP | 13 | Ring metacarpophalangeal joint |
| 149-151 | Ring PIP | 14 | Ring proximal interphalangeal joint |
| 152-154 | Ring DIP | 15 | Ring distal interphalangeal joint |
| 155-157 | Ring Tip | 16 | Ring fingertip |
| 158-160 | Pinky MCP | 17 | Pinky metacarpophalangeal joint |
| 161-163 | Pinky PIP | 18 | Pinky proximal interphalangeal joint |
| 164-166 | Pinky DIP | 19 | Pinky distal interphalangeal joint |
| 167-169 | Pinky Tip | 20 | Pinky fingertip |

### 2.2 Finger Joint Angles (Features 170-184)
**15 finger joint angles**

| Feature Index | Joint | Description | Range |
|---------------|-------|-------------|-------|
| 170 | Thumb CMC-MCP | Thumb base angle | 0-π/2 radians |
| 171 | Thumb MCP-IP | Thumb middle angle | 0-π/2 radians |
| 172 | Thumb IP-Tip | Thumb tip angle | 0-π/2 radians |
| 173 | Index MCP-PIP | Index base angle | 0-π/2 radians |
| 174 | Index PIP-DIP | Index middle angle | 0-π/2 radians |
| 175 | Index DIP-Tip | Index tip angle | 0-π/2 radians |
| 176 | Middle MCP-PIP | Middle base angle | 0-π/2 radians |
| 177 | Middle PIP-DIP | Middle middle angle | 0-π/2 radians |
| 178 | Middle DIP-Tip | Middle tip angle | 0-π/2 radians |
| 179 | Ring MCP-PIP | Ring base angle | 0-π/2 radians |
| 180 | Ring PIP-DIP | Ring middle angle | 0-π/2 radians |
| 181 | Ring DIP-Tip | Ring tip angle | 0-π/2 radians |
| 182 | Pinky MCP-PIP | Pinky base angle | 0-π/2 radians |
| 183 | Pinky PIP-DIP | Pinky middle angle | 0-π/2 radians |
| 184 | Pinky DIP-Tip | Pinky tip angle | 0-π/2 radians |

### 2.3 Inter-finger Distances (Features 185-194)
**10 distances between fingertips**

| Feature Index | Distance | Description |
|---------------|----------|-------------|
| 185 | Thumb-Index | Distance between thumb and index tips |
| 186 | Thumb-Middle | Distance between thumb and middle tips |
| 187 | Thumb-Ring | Distance between thumb and ring tips |
| 188 | Thumb-Pinky | Distance between thumb and pinky tips |
| 189 | Index-Middle | Distance between index and middle tips |
| 190 | Index-Ring | Distance between index and ring tips |
| 191 | Index-Pinky | Distance between index and pinky tips |
| 192 | Middle-Ring | Distance between middle and ring tips |
| 193 | Middle-Pinky | Distance between middle and pinky tips |
| 194 | Ring-Pinky | Distance between ring and pinky tips |

### 2.4 Hand Shape Features (Features 195-196)
**2 overall hand shape descriptors**

| Feature Index | Feature | Description |
|---------------|---------|-------------|
| 195 | Hand Openness | Average distance of fingertips from palm center |
| 196 | Finger Spread | Variance of fingertip positions (spread measurement) |

---

## FEATURE GROUP 3: RIGHT HAND FEATURES (Features 197-286)
### Total: 90 features

### Structure identical to Left Hand (Features 107-196)
- **Normalized Hand Landmarks**: Features 197-259 (63 features)
- **Finger Joint Angles**: Features 260-274 (15 features)
- **Inter-finger Distances**: Features 275-284 (10 features)
- **Hand Shape Features**: Features 285-286 (2 features)

---

## FEATURE GROUP 4: COMPACT FACE FEATURES (Features 287-378)
### Total: 92 features

### 4.1 Normalized Face Landmarks (Features 287-370)
**28 key landmarks × 3 coordinates = 84 features**

#### Feature Order by Facial Region

**Mouth Landmarks (Features 287-310): 8 landmarks × 3 = 24 features**
- Upper/Lower Lip Center, Left/Right Corners, Lip Peak/Bottom, Outer edges

**Eyebrow Landmarks (Features 311-328): 6 landmarks × 3 = 18 features**
- Left/Right eyebrow outer, inner, and peak points

**Eye Landmarks (Features 329-352): 8 landmarks × 3 = 24 features**
- Left/Right eye corners and upper/lower lids

**Head Structure (Features 353-370): 6 landmarks × 3 = 18 features**
- Nose tip, chin center/bottom, left/right cheeks, forehead center

### 4.2 Semantic Face Features (Features 371-378)
**8 derived facial expression measurements**

| Feature Index | Feature | Description | Range |
|---------------|---------|-------------|-------|
| 371 | Mouth Openness | Vertical lip separation | 0.0-0.15 |
| 372 | Mouth Width | Horizontal mouth extent | 0.05-0.12 |
| 373 | Mouth Relative X | Horizontal mouth displacement | ±0.02 |
| 374 | Mouth Relative Y | Vertical mouth displacement | ±0.03 |
| 375 | Eyebrow Raise | Average eyebrow elevation | ±0.05 |
| 376 | Eyebrow Asymmetry | Left-right brow difference | 0.0-0.03 |
| 377 | Eye Openness | Average eye opening | 0.0-0.08 |
| 378 | Eye Asymmetry | Left-right eye difference | 0.0-0.02 |

---

## FEATURE GROUP 5: RELATIONSHIP FEATURES (Features 379-384)
### Total: 6 features

**Spatial relationships between major body parts**

| Feature Index | Relationship | Description | Units |
|---------------|--------------|-------------|-------|
| 379 | Left Hand to Face | Distance from left wrist to nose | Normalized |
| 380 | Right Hand to Face | Distance from right wrist to nose | Normalized |
| 381 | Left Hand to Chest | Relative position of left hand to torso | X,Y,Z vector |
| 382 | Right Hand to Chest | Relative position of right hand to torso | X,Y,Z vector |
| 383 | Hands Distance | Distance between left and right wrists | Normalized |
| 384 | Hands Relative Position | Spatial relationship between hands | Vector |

**Note**: Some relationship features may be multi-dimensional, actual count varies (1-8 features total)

---

## FEATURE GROUP 6: METADATA FEATURES (Features 385-390)
### Total: 6 features

**Data quality and completeness indicators**

| Feature Index | Metadata | Description | Range |
|---------------|----------|-------------|-------|
| 385 | Pose Complete | Whether full pose was detected | 0.0-1.0 |
| 386 | Left Hand Complete | Whether left hand was detected | 0.0-1.0 |
| 387 | Right Hand Complete | Whether right hand was detected | 0.0-1.0 |
| 388 | Face Complete | Whether face was detected | 0.0-1.0 |
| 389 | Has Full Torso | Whether full torso is visible | 0.0-1.0 |
| 390 | Completeness Score | Overall data quality score | 0.0-1.0 |

---

## GEOMETRY-AWARE AUGMENTATION GUIDELINES

### 1. POSE AUGMENTATION CONSTRAINTS

#### **Anatomical Limits**
```python
# Joint angle constraints (radians)
ELBOW_MIN, ELBOW_MAX = 0.0, 2.6  # 0° to 150°
WRIST_ROTATION_LIMIT = 0.5       # Limited wrist rotation

# Limb length stability
LIMB_LENGTH_VARIATION = 0.1      # ±10% maximum variation

# Bilateral symmetry preference
BILATERAL_CORRELATION = 0.8      # Left-right correlation strength
```

#### **Sign Language Specific**
```python
# Shoulder position affects sign space
if sign_involves_large_movements:
    shoulder_stability_required = True
    
# Elbow angles critical for handshapes
preserve_elbow_angles_for_critical_handshapes = True
```

### 2. HAND AUGMENTATION CONSTRAINTS

#### **Finger Joint Coupling**
```python
# Anatomical finger coupling
FINGER_JOINT_COUPLING = {
    'ring_pinky_correlation': 0.7,    # Ring and pinky move together
    'mcp_pip_correlation': 0.6,       # Adjacent joints correlate
    'independence_thumb': 0.3         # Thumb most independent
}

# Handshape preservation
CRITICAL_HANDSHAPES = ['A', 'B', 'C', '1', '5', 'L', 'O', 'F']
```

#### **Inter-finger Distance Limits**
```python
# Physical constraints
MAX_FINGER_SPREAD = 0.25          # Maximum realistic spread
THUMB_OPPOSITION_RANGE = 0.20     # Thumb opposition limit

# Sign language constraints  
preserve_fingerspelling_distances = True
preserve_classifier_configurations = True
```

### 3. FACE AUGMENTATION CONSTRAINTS

#### **Expression Coupling**
```python
# Natural facial expression correlations
EXPRESSION_COUPLING = {
    'mouth_eyebrow_correlation': 0.4,
    'eye_eyebrow_correlation': 0.6,
    'bilateral_symmetry_preference': 0.8
}

# Grammatical marker preservation
PRESERVE_GRAMMATICAL_EXPRESSIONS = {
    'question_eyebrows': True,
    'negation_headshake': True,
    'mouth_morphemes': True
}
```

### 4. RELATIONSHIP FEATURE CONSTRAINTS

#### **Spatial Consistency**
```python
# Hand-face proximity limits
MIN_HAND_FACE_DISTANCE = 0.1     # Avoid collision
MAX_HAND_FACE_DISTANCE = 2.0     # Stay in sign space

# Hand-hand coordination
maintain_bilateral_coordination = True
preserve_two_handed_symmetry = True
```

### 5. AUGMENTATION PRIORITY SYSTEM

#### **HIGH Priority (Free Augmentation)**
- Overall body position and orientation
- Non-critical joint angles
- General facial expression intensity
- Hand position in sign space

#### **MEDIUM Priority (Constrained Augmentation)**
- Finger joint angles (preserve handshapes)
- Facial expression type (preserve grammar)
- Hand-hand relationships (preserve coordination)

#### **LOW Priority (Minimal/No Augmentation)**
- Critical handshapes and fingerspelling
- Grammatical facial expressions
- Mouth morphemes
- Contact relationships between hands

### 6. TEMPORAL CONSISTENCY RULES

#### **Motion Smoothness**
```python
# Maximum frame-to-frame changes
MAX_FRAME_CHANGE = {
    'pose_landmarks': 0.05,
    'hand_landmarks': 0.03,
    'face_landmarks': 0.02,
    'joint_angles': 0.1,
    'relationships': 0.08
}
```

#### **Sign Phase Awareness**
```python
# Different augmentation strategies by sign phase
if sign_phase == 'preparation':
    reduce_augmentation_intensity(0.3)
elif sign_phase == 'stroke':
    apply_full_augmentation()
elif sign_phase == 'retraction':
    smooth_transition_to_neutral()
```

### 7. VALIDATION FRAMEWORK

#### **Anatomical Validation**
```python
def validate_anatomy(features):
    # Check joint angle limits
    assert all(0 <= angle <= 2.6 for angle in elbow_angles)
    
    # Check hand-face collision
    assert all(distance > 0.1 for distance in hand_face_distances)
    
    # Check bilateral symmetry within reasonable bounds
    assert bilateral_asymmetry < 0.3
    
    return True
```

#### **Sign Language Validation**
```python
def validate_sign_language(features, sign_metadata):
    # Preserve critical handshapes
    if sign_metadata['handshape'] in CRITICAL_HANDSHAPES:
        preserve_hand_configuration(features)
    
    # Preserve grammatical markers
    if sign_metadata['type'] == 'question':
        assert features[375] > 0.02  # eyebrow_raise
    
    return True
```

---

## COMPLETE AUGMENTATION EXAMPLE

```python
def comprehensive_geometry_aware_augmentation(features, metadata):
    """
    Apply geometry-aware augmentation to all 391 features.
    
    Args:
        features: Array of 391 features [pose|left_hand|right_hand|face|relationships|metadata]
        metadata: Sign language and anatomical metadata
    
    Returns:
        augmented_features: Anatomically and linguistically valid augmented features
    """
    
    augmented = features.copy()
    
    # 1. Augment pose (features 0-106)
    pose_features = augmented[0:107]
    pose_features = augment_pose_with_constraints(pose_features, metadata)
    
    # 2. Augment hands (features 107-286) 
    left_hand = augmented[107:197]
    right_hand = augmented[197:287]
    left_hand, right_hand = augment_hands_bilaterally(left_hand, right_hand, metadata)
    
    # 3. Augment face (features 287-378)
    face_features = augmented[287:379]
    face_features = augment_face_with_grammar(face_features, metadata)
    
    # 4. Update relationships (features 379-384)
    relationships = compute_relationships(pose_features, left_hand, right_hand, face_features)
    
    # 5. Update metadata (features 385-390) 
    metadata_features = compute_completeness_scores(pose_features, left_hand, right_hand, face_features)
    
    # Reconstruct full feature vector
    augmented[0:107] = pose_features
    augmented[107:197] = left_hand  
    augmented[197:287] = right_hand
    augmented[287:379] = face_features
    augmented[379:385] = relationships
    augmented[385:391] = metadata_features
    
    # Final validation
    assert validate_anatomy(augmented)
    assert validate_sign_language(augmented, metadata)
    
    return augmented
```

This comprehensive documentation provides the foundation for creating sophisticated, anatomically-aware data augmentations across all modalities of your sign language recognition system while preserving the critical linguistic and biomechanical constraints necessary for meaningful sign language data. 