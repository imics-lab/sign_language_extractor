#!/usr/bin/env python3
"""
Compact face feature extraction for sign language recognition.
Uses only key facial landmarks instead of all 468 MediaPipe face landmarks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

class CompactFaceFeatureExtractor:
    """Extract compact facial features for sign language recognition."""
    
    def __init__(self):
        # Define key landmark indices for facial expressions
        self.MOUTH_LANDMARKS = [
            13,   # Upper lip center
            14,   # Lower lip center  
            61,   # Left mouth corner
            291,  # Right mouth corner
            17,   # Upper lip peak
            18,   # Lower lip bottom
            78,   # Left mouth outer
            308   # Right mouth outer
        ]
        
        self.EYEBROW_LANDMARKS = [
            70,   # Left eyebrow outer
            63,   # Left eyebrow inner
            105,  # Left eyebrow peak
            296,  # Right eyebrow outer  
            293,  # Right eyebrow inner
            334   # Right eyebrow peak
        ]
        
        self.EYE_LANDMARKS = [
            33,   # Left eye outer corner
            133,  # Left eye inner corner
            159,  # Left eye upper lid
            145,  # Left eye lower lid
            362,  # Right eye outer corner
            263,  # Right eye inner corner
            386,  # Right eye upper lid
            374   # Right eye lower lid
        ]
        
        self.HEAD_LANDMARKS = [
            1,    # Nose tip
            152,  # Chin center
            172,  # Chin bottom
            136,  # Left cheek
            365,  # Right cheek
            10    # Forehead center
        ]
        
        # Combined list of all key landmarks
        self.KEY_LANDMARKS = (self.MOUTH_LANDMARKS + self.EYEBROW_LANDMARKS + 
                             self.EYE_LANDMARKS + self.HEAD_LANDMARKS)
        
    def extract_compact_face_features(self, face: np.ndarray, pose: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Extract compact facial features from key landmarks only.
        
        Args:
            face: Full face landmarks array (468 x 3)
            pose: Pose landmarks for reference frame
            
        Returns:
            Tuple of (features, transform_params)
        """
        # Use nose from pose as reference point (more stable than face landmarks)
        NOSE_IDX = 0  # Pose landmark index for nose
        nose_pos = pose[NOSE_IDX]
        
        # Extract only key landmarks
        key_face_landmarks = face[self.KEY_LANDMARKS]
        
        # Center on nose position
        face_centered = key_face_landmarks - nose_pos
        
        # Estimate face scale from pose (shoulder width * factor)
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        face_scale = np.linalg.norm(pose[RIGHT_SHOULDER] - pose[LEFT_SHOULDER]) * 0.3
        
        if face_scale > 0:
            face_normalized = face_centered / face_scale
        else:
            face_normalized = face_centered
            face_scale = 1.0
        
        # Store transformation parameters
        transform = {
            'nose_position': nose_pos,
            'face_scale': face_scale,
            'key_landmark_indices': self.KEY_LANDMARKS,
            'original_shape': face.shape
        }
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(key_face_landmarks, nose_pos, face_scale)
        
        features = {
            'landmarks_normalized': face_normalized,
            'semantic_features': semantic_features,
            'num_key_landmarks': len(self.KEY_LANDMARKS)
        }
        
        return features, transform
    
    def _extract_semantic_features(self, key_landmarks: np.ndarray, 
                                 nose_pos: np.ndarray, face_scale: float) -> Dict:
        """Extract semantic facial expression features."""
        features = {}
        
        # Mouth features
        mouth_landmarks = key_landmarks[:len(self.MOUTH_LANDMARKS)]
        features.update(self._analyze_mouth(mouth_landmarks, nose_pos, face_scale))
        
        # Eyebrow features  
        eyebrow_start = len(self.MOUTH_LANDMARKS)
        eyebrow_end = eyebrow_start + len(self.EYEBROW_LANDMARKS)
        eyebrow_landmarks = key_landmarks[eyebrow_start:eyebrow_end]
        features.update(self._analyze_eyebrows(eyebrow_landmarks, nose_pos, face_scale))
        
        # Eye features
        eye_start = eyebrow_end
        eye_end = eye_start + len(self.EYE_LANDMARKS)
        eye_landmarks = key_landmarks[eye_start:eye_end]
        features.update(self._analyze_eyes(eye_landmarks, nose_pos, face_scale))
        
        return features
    
    def _analyze_mouth(self, mouth_landmarks: np.ndarray, 
                      nose_pos: np.ndarray, face_scale: float) -> Dict:
        """Analyze mouth state for speech/expression."""
        # Indices within mouth_landmarks array
        upper_lip = mouth_landmarks[0]      # Index 0: upper lip center
        lower_lip = mouth_landmarks[1]      # Index 1: lower lip center
        left_corner = mouth_landmarks[2]    # Index 2: left corner
        right_corner = mouth_landmarks[3]   # Index 3: right corner
        
        # Mouth openness (vertical distance between lips)
        mouth_height = np.linalg.norm(upper_lip - lower_lip) / face_scale
        
        # Mouth width (horizontal distance between corners)
        mouth_width = np.linalg.norm(right_corner - left_corner) / face_scale
        
        # Mouth center relative to nose
        mouth_center = (upper_lip + lower_lip) / 2
        mouth_to_nose = (mouth_center - nose_pos) / face_scale
        
        return {
            'mouth_openness': mouth_height,
            'mouth_width': mouth_width,
            'mouth_relative_x': mouth_to_nose[0],
            'mouth_relative_y': mouth_to_nose[1]
        }
    
    def _analyze_eyebrows(self, eyebrow_landmarks: np.ndarray,
                         nose_pos: np.ndarray, face_scale: float) -> Dict:
        """Analyze eyebrow position for questions/emphasis."""
        # Left eyebrow (indices 0, 1, 2)
        left_outer = eyebrow_landmarks[0]
        left_inner = eyebrow_landmarks[1] 
        left_peak = eyebrow_landmarks[2]
        
        # Right eyebrow (indices 3, 4, 5)
        right_outer = eyebrow_landmarks[3]
        right_inner = eyebrow_landmarks[4]
        right_peak = eyebrow_landmarks[5]
        
        # Average eyebrow height relative to nose
        left_height = (left_peak[1] - nose_pos[1]) / face_scale
        right_height = (right_peak[1] - nose_pos[1]) / face_scale
        avg_eyebrow_height = (left_height + right_height) / 2
        
        # Eyebrow asymmetry (difference between left and right)
        eyebrow_asymmetry = abs(left_height - right_height)
        
        return {
            'eyebrow_raise': avg_eyebrow_height,
            'eyebrow_asymmetry': eyebrow_asymmetry
        }
    
    def _analyze_eyes(self, eye_landmarks: np.ndarray,
                     nose_pos: np.ndarray, face_scale: float) -> Dict:
        """Analyze eye state (open/closed, wide/squinting)."""
        # Left eye (indices 0-3)
        left_outer = eye_landmarks[0]
        left_inner = eye_landmarks[1]
        left_upper = eye_landmarks[2]
        left_lower = eye_landmarks[3]
        
        # Right eye (indices 4-7)
        right_outer = eye_landmarks[4]
        right_inner = eye_landmarks[5]
        right_upper = eye_landmarks[6]
        right_lower = eye_landmarks[7]
        
        # Eye openness (vertical distance between upper and lower lids)
        left_eye_openness = np.linalg.norm(left_upper - left_lower) / face_scale
        right_eye_openness = np.linalg.norm(right_upper - right_lower) / face_scale
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
        
        # Eye asymmetry
        eye_asymmetry = abs(left_eye_openness - right_eye_openness)
        
        return {
            'eye_openness': avg_eye_openness,
            'eye_asymmetry': eye_asymmetry
        }

# Integration function for the main feature extractor
def extract_compact_face_features(face_data: Dict) -> Optional[np.ndarray]:
    """
    Modified face feature extraction for use in create_npy_features_dataset.py
    Returns compact face features instead of full 1404-feature array.
    """
    if not isinstance(face_data, dict) or not face_data.get('landmarks_normalized'):
        return None
    
    feature_vector = []
    
    # Add normalized key landmarks only (28 landmarks × 3 = 84 features)
    landmarks = np.array(face_data['landmarks_normalized'])
    
    # Check if this is already compacted or full face landmarks
    if landmarks.shape[0] == 28:  # Already compacted
        feature_vector.extend(landmarks.flatten())
    elif landmarks.shape[0] == 468:  # Full face landmarks, need to compact
        extractor = CompactFaceFeatureExtractor()
        key_landmarks = landmarks[extractor.KEY_LANDMARKS]
        feature_vector.extend(key_landmarks.flatten())
    else:
        # Unknown format, skip
        return None
    
    # Add semantic features if available
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
    
    return np.array(feature_vector) if feature_vector else None

if __name__ == "__main__":
    # Test the compact face feature extractor
    print("Compact Face Feature Extractor")
    print(f"Total key landmarks: {len(CompactFaceFeatureExtractor().KEY_LANDMARKS)}")
    print(f"Expected feature count: {len(CompactFaceFeatureExtractor().KEY_LANDMARKS) * 3 + 8} features")
    print("Breakdown:")
    print(f"  - Key landmarks: {len(CompactFaceFeatureExtractor().KEY_LANDMARKS)} × 3 = {len(CompactFaceFeatureExtractor().KEY_LANDMARKS) * 3}")
    print(f"  - Semantic features: 8")
    print(f"  - Total: {len(CompactFaceFeatureExtractor().KEY_LANDMARKS) * 3 + 8} (vs 1407 for full face)") 