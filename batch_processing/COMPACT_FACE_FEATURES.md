# Compact Face Mode Feature Documentation

This document provides detailed information about the **92 facial features** extracted in Compact Face Mode, designed for sign language recognition. This documentation is intended to facilitate **geometry-aware data augmentations** that respect human facial anatomy and biomechanics.

## Overview

**Total Features**: 92
- **Normalized Landmarks**: 84 features (28 landmarks × 3 coordinates)
- **Semantic Features**: 8 features (derived measurements)

**Coordinate System**: All landmarks are normalized relative to nose position and scaled by estimated face size.

---

## Part 1: Normalized Landmarks (Features 0-83)

### Feature Order in Vector

The 84 landmark features appear in this exact order:

1. **Mouth Landmarks** (Features 0-23): 8 landmarks × 3 = 24 features
2. **Eyebrow Landmarks** (Features 24-41): 6 landmarks × 3 = 18 features  
3. **Eye Landmarks** (Features 42-65): 8 landmarks × 3 = 24 features
4. **Head Structure Landmarks** (Features 66-83): 6 landmarks × 3 = 18 features

---

## Detailed Feature Descriptions

### 1. MOUTH LANDMARKS (Features 0-23)

#### MediaPipe Landmark Indices: [13, 14, 61, 291, 17, 18, 78, 308]

| Feature Index | Landmark | MediaPipe ID | Anatomical Description |
|---------------|----------|--------------|------------------------|
| 0-2 | Upper Lip Center | 13 | Center point of upper lip vermillion border |
| 3-5 | Lower Lip Center | 14 | Center point of lower lip vermillion border |
| 6-8 | Left Mouth Corner | 61 | Left commissure (corner) of mouth |
| 9-11 | Right Mouth Corner | 291 | Right commissure (corner) of mouth |
| 12-14 | Upper Lip Peak | 17 | Highest point of upper lip (cupid's bow peak) |
| 15-17 | Lower Lip Bottom | 18 | Lowest point of lower lip |
| 18-20 | Left Mouth Outer | 78 | Outer edge of left side of mouth |
| 21-23 | Right Mouth Outer | 308 | Outer edge of right side of mouth |

#### Anatomical Constraints for Augmentation

**Mouth Opening Constraints:**
- **Vertical Range**: Upper-lower lip distance can vary from 0 (closed) to ~15% of face height
- **Horizontal Range**: Mouth width should remain relatively stable (±10% variation)
- **Symmetry**: Left-right mouth features should maintain approximate symmetry
- **Anatomical Limits**: Mouth cannot open beyond realistic human capabilities

**Sign Language Relevance:**
- **Critical for**: /p/, /b/, /f/, /v/ sounds, mouth morphemes, size/manner markers
- **Augmentation Priority**: HIGH - mouth shape is crucial for sign language
- **Correlated Movement**: Mouth opening often correlates with eyebrow position in emotional expressions

---

### 2. EYEBROW LANDMARKS (Features 24-41)

#### MediaPipe Landmark Indices: [70, 63, 105, 296, 293, 334]

| Feature Index | Landmark | MediaPipe ID | Anatomical Description |
|---------------|----------|--------------|------------------------|
| 24-26 | Left Eyebrow Outer | 70 | Outer (temporal) end of left eyebrow |
| 27-29 | Left Eyebrow Inner | 63 | Inner (nasal) end of left eyebrow |
| 30-32 | Left Eyebrow Peak | 105 | Highest point of left eyebrow arch |
| 33-35 | Right Eyebrow Outer | 296 | Outer (temporal) end of right eyebrow |
| 36-38 | Right Eyebrow Inner | 293 | Inner (nasal) end of right eyebrow |
| 39-41 | Right Eyebrow Peak | 334 | Highest point of right eyebrow arch |

#### Anatomical Constraints for Augmentation

**Eyebrow Movement Constraints:**
- **Vertical Range**: ±20% of eye-to-hairline distance
- **Symmetry**: Can be asymmetric but extreme asymmetry is unnatural
- **Muscle Groups**: Frontalis (raises) vs Corrugator/Procerus (lowers/furrows)
- **Anatomical Coupling**: Inner brow movement often coupled, outer brow more independent

**Sign Language Relevance:**
- **Critical for**: Questions (raised brows), negation (lowered brows), emphasis, conditionals
- **Augmentation Priority**: HIGH - eyebrow position is grammatically significant
- **Typical Patterns**: Both raised (questions), both lowered (negation), asymmetric (uncertainty)

---

### 3. EYE LANDMARKS (Features 42-65)

#### MediaPipe Landmark Indices: [33, 133, 159, 145, 362, 263, 386, 374]

| Feature Index | Landmark | MediaPipe ID | Anatomical Description |
|---------------|----------|--------------|------------------------|
| 42-44 | Left Eye Outer Corner | 33 | Outer canthus of left eye |
| 45-47 | Left Eye Inner Corner | 133 | Inner canthus of left eye |
| 48-50 | Left Eye Upper Lid | 159 | Highest point of left upper eyelid |
| 51-53 | Left Eye Lower Lid | 145 | Lowest point of left lower eyelid |
| 54-56 | Right Eye Outer Corner | 362 | Outer canthus of right eye |
| 57-59 | Right Eye Inner Corner | 263 | Inner canthus of right eye |
| 60-62 | Right Eye Upper Lid | 386 | Highest point of right upper eyelid |
| 63-65 | Right Eye Lower Lid | 374 | Lowest point of right lower eyelid |

#### Anatomical Constraints for Augmentation

**Eye Opening Constraints:**
- **Vertical Range**: 0% (closed) to 100% (wide open) of normal eye opening
- **Symmetry**: Eyes typically move together (conjugate movement)
- **Blink Patterns**: Eyes close simultaneously, open may be slightly asynchronous
- **Anatomical Limits**: Cannot open beyond natural palpebral fissure

**Sign Language Relevance:**
- **Critical for**: Gaze direction, intensity markers, role shifting, constructed action
- **Augmentation Priority**: MEDIUM - important for discourse functions
- **Typical Patterns**: Wide eyes (surprise/emphasis), squinted (concentration), normal (neutral)

---

### 4. HEAD STRUCTURE LANDMARKS (Features 66-83)

#### MediaPipe Landmark Indices: [1, 152, 172, 136, 365, 10]

| Feature Index | Landmark | MediaPipe ID | Anatomical Description |
|---------------|----------|--------------|------------------------|
| 66-68 | Nose Tip | 1 | Tip of nose (pronasale) |
| 69-71 | Chin Center | 152 | Center point of chin |
| 72-74 | Chin Bottom | 172 | Lowest point of chin (menton) |
| 75-77 | Left Cheek | 136 | Prominence of left cheek |
| 78-80 | Right Cheek | 365 | Prominence of right cheek |
| 81-83 | Forehead Center | 10 | Center point of forehead |

#### Anatomical Constraints for Augmentation

**Head Structure Constraints:**
- **Fixed Geometry**: These landmarks represent mostly rigid bone structure
- **Minimal Movement**: Very limited independent movement except for subtle muscle contractions
- **Proportional Relationships**: Maintain anatomical proportions between landmarks
- **Reference Frame**: Nose tip serves as the coordinate system origin

**Sign Language Relevance:**
- **Critical for**: Face orientation, head nods/shakes, spatial reference
- **Augmentation Priority**: LOW - mostly structural, limited variation
- **Typical Patterns**: Slight movements for emphasis, head position changes

---

## Part 2: Semantic Features (Features 84-91)

### Feature Order and Descriptions

| Feature Index | Feature Name | Description | Units/Range |
|---------------|--------------|-------------|-------------|
| 84 | mouth_openness | Vertical distance between lips | 0.0-0.15 (normalized) |
| 85 | mouth_width | Horizontal distance between mouth corners | 0.05-0.12 (normalized) |
| 86 | mouth_relative_x | Horizontal mouth displacement from nose | ±0.02 (normalized) |
| 87 | mouth_relative_y | Vertical mouth displacement from nose | ±0.03 (normalized) |
| 88 | eyebrow_raise | Average eyebrow height relative to nose | ±0.05 (normalized) |
| 89 | eyebrow_asymmetry | Difference between left and right brow height | 0.0-0.03 (normalized) |
| 90 | eye_openness | Average vertical eye opening | 0.0-0.08 (normalized) |
| 91 | eye_asymmetry | Difference between left and right eye opening | 0.0-0.02 (normalized) |

---

## Geometry-Aware Augmentation Guidelines

### 1. Anatomical Constraint Enforcement

#### **Mouth Constraints**
```python
# Realistic mouth opening limits
mouth_openness_min = 0.0      # Completely closed
mouth_openness_max = 0.15     # Maximum realistic opening
mouth_width_variation = ±10%  # Natural variation in mouth width

# Symmetry constraints
left_right_mouth_diff_max = 0.01  # Maximum natural asymmetry
```

#### **Eyebrow Constraints**
```python
# Eyebrow movement limits
eyebrow_raise_min = -0.03     # Maximum lowering (frown)
eyebrow_raise_max = 0.05      # Maximum raising (surprise)
eyebrow_asymmetry_max = 0.03  # Maximum natural asymmetry

# Correlated movements
if mouth_openness > 0.08:     # If mouth very open
    eyebrow_raise += 0.02     # Slight eyebrow raise (surprise)
```

#### **Eye Constraints**
```python
# Eye opening limits
eye_openness_min = 0.0        # Eyes closed
eye_openness_max = 0.08       # Eyes wide open
eye_asymmetry_max = 0.02      # Maximum natural asymmetry

# Synchronized movement
left_right_eye_correlation = 0.9  # Eyes usually move together
```

### 2. Sign Language Specific Rules

#### **Grammatical Markers**
```python
# Question markers: Raised eyebrows + slightly open mouth
if sign_type == "question":
    eyebrow_raise = random.uniform(0.02, 0.04)
    mouth_openness = random.uniform(0.02, 0.06)

# Negation markers: Lowered eyebrows + neutral mouth
if sign_type == "negation":
    eyebrow_raise = random.uniform(-0.03, -0.01)
    mouth_openness = random.uniform(0.0, 0.02)

# Emphasis markers: Wide eyes + raised eyebrows
if sign_type == "emphasis":
    eye_openness = random.uniform(0.06, 0.08)
    eyebrow_raise = random.uniform(0.03, 0.05)
```

#### **Mouth Morphemes (Critical - Do Not Augment)**
```python
# These mouth shapes are lexically contrastive - preserve exactly
critical_mouth_shapes = {
    "pah": mouth_openness < 0.01,  # Lips closed/barely open
    "cha": mouth_openness > 0.08,  # Mouth wide open  
    "fish": mouth_width < 0.06,    # Lips pursed
    "th": mouth_openness = 0.02    # Slight opening for tongue
}
```

### 3. Biomechanical Coupling Rules

#### **Natural Correlations**
```python
# Mouth-eyebrow coupling
if mouth_openness > 0.10:  # Very open mouth
    eyebrow_raise += 0.015  # Slight eyebrow lift

# Eye-eyebrow coupling  
if eye_openness > 0.07:    # Wide eyes
    eyebrow_raise += 0.02   # Raised eyebrows

# Symmetry preferences
eyebrow_asymmetry = random.uniform(0.0, 0.01)  # Prefer symmetry
eye_asymmetry = random.uniform(0.0, 0.005)     # Strong symmetry preference
```

#### **Muscle Group Constraints**
```python
# Frontalis muscle (forehead) - raises eyebrows
# Cannot raise one side without affecting the other
if abs(left_brow_raise - right_brow_raise) > 0.02:
    # Force more correlation
    avg_raise = (left_brow_raise + right_brow_raise) / 2
    left_brow_raise = avg_raise + random.uniform(-0.01, 0.01)
    right_brow_raise = avg_raise + random.uniform(-0.01, 0.01)
```

### 4. Augmentation Strategies by Priority

#### **HIGH Priority (Freely Augmentable)**
- **Eyebrow position**: Critical for grammar, wide variation acceptable
- **Overall mouth position**: Important for expressions
- **Eye openness**: Affects intensity and attention

#### **MEDIUM Priority (Cautious Augmentation)**
- **Mouth width**: Some variation acceptable but affects phonemes
- **Eye asymmetry**: Small variations only
- **Head structure**: Very limited augmentation

#### **LOW Priority (Preserve or Minimal Change)**
- **Mouth morpheme shapes**: Lexically contrastive
- **Extreme facial expressions**: May break sign meaning
- **Anatomically impossible configurations**: Never augment beyond human limits

### 5. Temporal Consistency Rules

#### **Smooth Transitions**
```python
# Ensure temporal smoothness in feature sequences
max_frame_to_frame_change = {
    'mouth_openness': 0.02,     # Mouth can't change too quickly
    'eyebrow_raise': 0.015,     # Eyebrows move moderately fast
    'eye_openness': 0.03,       # Eyes can change quickly (blinks)
    'mouth_width': 0.01         # Mouth width changes slowly
}
```

#### **Natural Gesture Timing**
```python
# Facial expressions should align with hand movements
if sign_phase == "preparation":
    # Minimal facial expression
    reduce_all_features_by_factor(0.5)
elif sign_phase == "stroke":
    # Peak facial expression
    enhance_grammatical_markers()
elif sign_phase == "retraction":
    # Return to neutral
    transition_to_neutral_smoothly()
```

### 6. Quality Assurance Checks

#### **Anatomical Validation**
```python
def validate_facial_configuration(features):
    # Check all constraints are satisfied
    assert 0.0 <= features[84] <= 0.15  # mouth_openness
    assert 0.05 <= features[85] <= 0.12  # mouth_width  
    assert -0.03 <= features[88] <= 0.05  # eyebrow_raise
    assert 0.0 <= features[90] <= 0.08   # eye_openness
    
    # Check correlations make sense
    if features[84] > 0.10:  # Very open mouth
        assert features[88] > -0.01  # Eyebrows shouldn't be very low
        
    return True
```

#### **Sign Language Validation**
```python
def validate_sign_language_constraints(features, sign_type):
    if sign_type in critical_mouth_morphemes:
        # Don't augment mouth features for these signs
        return preserve_original_mouth_features()
    
    if sign_type == "question":
        # Ensure eyebrows are raised
        assert features[88] > 0.01
        
    return True
```

---

## Usage Example for Data Augmentation

```python
def geometry_aware_face_augmentation(original_features, sign_metadata):
    """
    Apply geometry-aware augmentation to compact face features.
    
    Args:
        original_features: Array of 92 facial features
        sign_metadata: Dictionary with sign type, timing, etc.
    
    Returns:
        augmented_features: Augmented array respecting anatomical constraints
    """
    
    features = original_features.copy()
    
    # Extract current semantic features
    mouth_openness = features[84]
    mouth_width = features[85]
    eyebrow_raise = features[88]
    eye_openness = features[90]
    
    # Check if this sign has critical mouth morphemes
    if sign_metadata.get('has_mouth_morpheme', False):
        # Only augment non-mouth features
        features[88] += random.uniform(-0.01, 0.01)  # Slight eyebrow variation
        features[90] += random.uniform(-0.005, 0.005)  # Slight eye variation
    else:
        # Full augmentation possible
        
        # Augment eyebrow position (grammatically relevant)
        if sign_metadata.get('type') == 'question':
            features[88] = max(0.02, features[88] + random.uniform(0.0, 0.02))
        else:
            features[88] += random.uniform(-0.02, 0.02)
            
        # Augment mouth opening (expressively relevant)
        mouth_delta = random.uniform(-0.02, 0.02)
        features[84] = np.clip(features[84] + mouth_delta, 0.0, 0.15)
        
        # Correlate eyebrow with mouth opening
        if features[84] > 0.08:  # If mouth very open
            features[88] += 0.01  # Raise eyebrows slightly
            
        # Augment eye openness (attention/intensity)
        eye_delta = random.uniform(-0.01, 0.01)
        features[90] = np.clip(features[90] + eye_delta, 0.0, 0.08)
    
    # Update corresponding landmark positions
    features = update_landmarks_from_semantics(features)
    
    # Validate constraints
    assert validate_facial_configuration(features)
    assert validate_sign_language_constraints(features, sign_metadata)
    
    return features
```

This documentation provides the foundation for creating anatomically and linguistically aware facial feature augmentations that will enhance your model's robustness without breaking the semantic content of signs. 