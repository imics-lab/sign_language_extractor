import numpy as np
from typing import Dict, List, Tuple, Optional
import mediapipe as mp
import json
import argparse
from pathlib import Path

class SignLanguageFeatureExtractor:
    """
    Extract invariant features from MediaPipe landmarks for sign language recognition.
    With reversible transformations for debugging and validation.
    """
    
    def __init__(self, use_compact_face: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            use_compact_face: If True, extract only 28 key facial landmarks (92 features total).
                            If False, extract all 468 facial landmarks (~1407 features total).
        """
        # Define important landmark indices
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.NOSE = 0
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        
        # Face feature extraction mode
        self.use_compact_face = use_compact_face
        
        # Define key landmark indices for compact face mode
        self.KEY_FACE_LANDMARKS = {
            'MOUTH': [13, 14, 61, 291, 17, 18, 78, 308],
            'EYEBROW': [70, 63, 105, 296, 293, 334],
            'EYE': [33, 133, 159, 145, 362, 263, 386, 374],
            'HEAD': [1, 152, 172, 136, 365, 10]
        }
        
        # Combined list of all key landmarks for compact mode
        self.COMPACT_LANDMARKS = (self.KEY_FACE_LANDMARKS['MOUTH'] + 
                                 self.KEY_FACE_LANDMARKS['EYEBROW'] + 
                                 self.KEY_FACE_LANDMARKS['EYE'] + 
                                 self.KEY_FACE_LANDMARKS['HEAD'])
        
    def extract_invariant_features(self, landmarks: Dict) -> Tuple[Dict, Dict]:
        """
        Convert MediaPipe landmarks to invariant feature representation.
        Handles missing or partially visible landmarks gracefully.
        
        Args:
            landmarks: Dictionary containing 'pose', 'left_hand', 'right_hand', 'face' landmarks
            
        Returns:
            Tuple of (features, transform_params) where transform_params contains
            information needed to reverse the transformation
        """
        features = {}
        transform_params = {}
        
        # Extract pose landmarks with visibility handling
        if landmarks.get('pose') and hasattr(landmarks['pose'], 'landmark'):
            pose = np.array([[lm.x, lm.y, lm.z] for lm in landmarks['pose'].landmark])
            
            # Extract visibility scores if available
            pose_visibility = None
            if hasattr(landmarks['pose'].landmark[0], 'visibility'):
                pose_visibility = np.array([lm.visibility for lm in landmarks['pose'].landmark])
            
            # Create body-centered coordinate system
            body_features, body_transform = self._extract_body_relative_features(pose, pose_visibility)
            features.update(body_features)
            transform_params['body'] = body_transform
        else:
            features['pose_available'] = False
            features['error'] = 'No pose landmarks detected'
            
        # Extract hand features only if pose is available (need wrist reference)
        if features.get('pose_available', False):
            # Left hand
            if landmarks.get('left_hand') and hasattr(landmarks['left_hand'], 'landmark'):
                if self._check_landmarks_visible([self.LEFT_WRIST], 
                                               features.get('landmarks_visibility_mask')):
                    left_hand = np.array([[lm.x, lm.y, lm.z] 
                                        for lm in landmarks['left_hand'].landmark])
                    features['left_hand'], transform_params['left_hand'] = \
                        self._extract_hand_features(left_hand, pose[self.LEFT_WRIST])
                else:
                    features['left_hand'] = {'available': False, 
                                           'reason': 'wrist_not_visible'}
            else:
                features['left_hand'] = {'available': False, 
                                       'reason': 'hand_not_detected'}
            
            # Right hand
            if landmarks.get('right_hand') and hasattr(landmarks['right_hand'], 'landmark'):
                if self._check_landmarks_visible([self.RIGHT_WRIST], 
                                               features.get('landmarks_visibility_mask')):
                    right_hand = np.array([[lm.x, lm.y, lm.z] 
                                         for lm in landmarks['right_hand'].landmark])
                    features['right_hand'], transform_params['right_hand'] = \
                        self._extract_hand_features(right_hand, pose[self.RIGHT_WRIST])
                else:
                    features['right_hand'] = {'available': False, 
                                            'reason': 'wrist_not_visible'}
            else:
                features['right_hand'] = {'available': False, 
                                        'reason': 'hand_not_detected'}
            
            # Face features
            if landmarks.get('face') and hasattr(landmarks['face'], 'landmark'):
                if self._check_landmarks_visible([self.NOSE], 
                                               features.get('landmarks_visibility_mask')):
                    face = np.array([[lm.x, lm.y, lm.z] 
                                   for lm in landmarks['face'].landmark])
                    features['face'], transform_params['face'] = \
                        self._extract_face_features(face, pose)
                else:
                    features['face'] = {'available': False, 
                                      'reason': 'nose_not_visible'}
            else:
                features['face'] = {'available': False, 
                                  'reason': 'face_not_detected'}
            
            # Extract relationships only if relevant parts are available
            features['relationships'] = self._extract_relationships_robust(
                pose, landmarks, features
            )
        
        # Add metadata about feature completeness
        features['feature_metadata'] = self._compute_feature_completeness(features, transform_params)
        
        return features, transform_params
    
    def _extract_relationships_robust(self, pose: np.ndarray, 
                                    landmarks: Dict, 
                                    features: Dict) -> Dict:
        """Extract spatial relationships between body parts, handling missing parts."""
        relationships = {}
        visibility_mask = features.get('landmarks_visibility_mask')
        
        # Hand to body relationships
        if isinstance(features.get('left_hand'), dict) and \
           features['left_hand'].get('landmarks_normalized') is not None:
            if self._check_landmarks_visible([self.LEFT_WRIST, self.NOSE], visibility_mask):
                relationships['left_hand_to_face'] = self._hand_to_face_distance(
                    pose[self.LEFT_WRIST], pose[self.NOSE], pose
                )
            if self._check_landmarks_visible([self.LEFT_WRIST, self.LEFT_SHOULDER, 
                                            self.RIGHT_SHOULDER], visibility_mask):
                relationships['left_hand_to_chest'] = self._hand_to_chest_position(
                    pose[self.LEFT_WRIST], pose
                )
        
        if isinstance(features.get('right_hand'), dict) and \
           features['right_hand'].get('landmarks_normalized') is not None:
            if self._check_landmarks_visible([self.RIGHT_WRIST, self.NOSE], visibility_mask):
                relationships['right_hand_to_face'] = self._hand_to_face_distance(
                    pose[self.RIGHT_WRIST], pose[self.NOSE], pose
                )
            if self._check_landmarks_visible([self.RIGHT_WRIST, self.LEFT_SHOULDER, 
                                            self.RIGHT_SHOULDER], visibility_mask):
                relationships['right_hand_to_chest'] = self._hand_to_chest_position(
                    pose[self.RIGHT_WRIST], pose
                )
        
        # Hand to hand relationship
        if (isinstance(features.get('left_hand'), dict) and 
            isinstance(features.get('right_hand'), dict) and
            features['left_hand'].get('landmarks_normalized') is not None and
            features['right_hand'].get('landmarks_normalized') is not None):
            if self._check_landmarks_visible([self.LEFT_WRIST, self.RIGHT_WRIST], 
                                           visibility_mask):
                relationships['hands_distance'] = self._normalized_distance(
                    pose[self.LEFT_WRIST], pose[self.RIGHT_WRIST], pose
                )
                relationships['hands_relative_position'] = self._hands_relative_position(pose)
        
        return relationships
    
    def _compute_feature_completeness(self, features: Dict, transform_params: Dict) -> Dict:
        """Compute metadata about which features were successfully extracted."""
        metadata = {
            'pose_complete': features.get('pose_available', False),
            'left_hand_complete': isinstance(features.get('left_hand'), dict) and 
                                features['left_hand'].get('landmarks_normalized') is not None,
            'right_hand_complete': isinstance(features.get('right_hand'), dict) and 
                                 features['right_hand'].get('landmarks_normalized') is not None,
            'face_complete': isinstance(features.get('face'), dict) and 
                           features['face'].get('landmarks_normalized') is not None,
            'has_full_torso': transform_params.get('body', {}).get('has_full_torso', False),
            'using_fallback': features.get('using_fallback', None)
        }
        
        # Compute overall completeness score
        completeness_score = sum([
            metadata['pose_complete'] * 0.4,
            metadata['left_hand_complete'] * 0.2,
            metadata['right_hand_complete'] * 0.2,
            metadata['face_complete'] * 0.1,
            metadata['has_full_torso'] * 0.1
        ])
        metadata['completeness_score'] = completeness_score
        
        return metadata
    
    def _extract_body_relative_features(self, pose: np.ndarray, 
                                       pose_visibility: Optional[np.ndarray] = None) -> Tuple[Dict, Dict]:
        """
        Extract body features invariant to position, rotation, and scale.
        Handles missing landmarks gracefully.
        
        Args:
            pose: Pose landmarks array
            pose_visibility: Optional visibility scores for each landmark
            
        Returns:
            Tuple of (features, transform) with metadata about what was computed
        """
        features = {}
        
        # Check which landmarks are available
        has_shoulders = self._check_landmarks_visible(
            [self.LEFT_SHOULDER, self.RIGHT_SHOULDER], pose_visibility
        )
        has_hips = self._check_landmarks_visible(
            [self.LEFT_HIP, self.RIGHT_HIP], pose_visibility
        )
        
        # Fallback strategies for missing landmarks
        if not has_shoulders:
            # Critical landmarks missing - return minimal features
            features['pose_available'] = False
            features['missing_landmarks'] = 'shoulders'
            transform = {'valid': False}
            return features, transform
            
        left_shoulder = pose[self.LEFT_SHOULDER]
        right_shoulder = pose[self.RIGHT_SHOULDER]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        
        # Handle zero shoulder width (e.g., when all landmarks are 0.0)
        if shoulder_width < 1e-6:
            # Set a default scale to avoid division by zero
            shoulder_width = 1.0
            print(f"Warning: Zero shoulder width detected, using default scale")
        
        # Handle missing hips - use shoulder-only reference frame
        if has_hips:
            left_hip = pose[self.LEFT_HIP]
            right_hip = pose[self.RIGHT_HIP]
            hip_center = (left_hip + right_hip) / 2
            body_center = (shoulder_center + hip_center) / 2
            
            # Full coordinate system with vertical reference
            y_axis = shoulder_center - hip_center
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
        else:
            # Fallback: use shoulders only
            body_center = shoulder_center
            
            # Estimate vertical direction (assume person is upright)
            # Use nose if available, otherwise assume Y-up
            if self._check_landmarks_visible([self.NOSE], pose_visibility):
                nose = pose[self.NOSE]
                y_axis = nose - shoulder_center
                y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
            else:
                y_axis = np.array([0, 1, 0])  # Default up direction
                
            features['using_fallback'] = 'shoulders_only'
        
        # Create coordinate system
        # X-axis: from left to right shoulder
        x_axis = (right_shoulder - left_shoulder) / shoulder_width
        
        # Ensure orthogonality
        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            # Degenerate case - shoulders aligned with vertical
            z_axis = np.array([0, 0, 1])
        else:
            z_axis = z_axis / z_norm
            
        # Recompute Y to ensure orthogonality
        y_axis = np.cross(z_axis, x_axis)
        
        # Rotation matrix from world to body coordinates
        R = np.array([x_axis, y_axis, z_axis]).T
        
        # Transform all pose landmarks to body-relative coordinates
        # Keep original values (including zeros) and only use NaN for truly missing landmarks
        body_relative_pose = np.zeros_like(pose)  # Initialize with zeros instead of NaN
        
        for i in range(len(pose)):
            if self._check_landmarks_visible([i], pose_visibility):
                # Translate to body center and rotate
                relative_pos = pose[i] - body_center
                body_relative_pose[i] = R.T @ relative_pos
                # Scale by shoulder width (now safe since we handled zero case)
                body_relative_pose[i] /= shoulder_width
            else:
                # Only set to NaN if truly missing (visibility score < threshold)
                # For web interface data without visibility scores, keep all landmarks
                if pose_visibility is not None:
                    body_relative_pose[i] = np.array([np.nan, np.nan, np.nan])
                else:
                    # No visibility data - treat all landmarks as valid (including zeros)
                    relative_pos = pose[i] - body_center
                    body_relative_pose[i] = R.T @ relative_pos
                    body_relative_pose[i] /= shoulder_width
        
        # Store transformation parameters
        transform = {
            'center': body_center,
            'rotation': R,
            'scale': shoulder_width,
            'original_shape': pose.shape,
            'valid': True,
            'has_full_torso': has_hips
        }
        
        # Extract features only for visible landmarks
        features.update({
            'pose_available': True,
            'landmarks_normalized': body_relative_pose,
            'joint_angles': self._calculate_joint_angles_robust(pose, pose_visibility),
            'limb_lengths_normalized': self._calculate_normalized_limb_lengths_robust(
                pose, shoulder_width, pose_visibility
            ),
            'landmarks_visibility_mask': pose_visibility if pose_visibility is not None else np.ones(len(pose))
        })
        
        return features, transform
    
    def _check_landmarks_visible(self, indices: List[int], 
                               visibility: Optional[np.ndarray],
                               threshold: float = 0.5) -> bool:
        """Check if landmarks at given indices are visible."""
        if visibility is None:
            return True  # Assume visible if no visibility data
            
        for idx in indices:
            if idx >= len(visibility) or visibility[idx] < threshold:
                return False
        return True
    
    def _calculate_joint_angles_robust(self, pose: np.ndarray, 
                                     visibility: Optional[np.ndarray]) -> List[float]:
        """Calculate angles between connected joints, handling missing landmarks."""
        angles = []
        angle_names = []
        
        # Define joint connections (parent, joint, child)
        connections = [
            (self.LEFT_SHOULDER, 13, 15, 'left_elbow'),
            (self.RIGHT_SHOULDER, 14, 16, 'right_elbow'),
            (13, 15, 17, 'left_wrist'),
            (14, 16, 18, 'right_wrist'),
        ]
        
        for parent, joint, child, name in connections:
            if self._check_landmarks_visible([parent, joint, child], visibility):
                vec1 = pose[parent] - pose[joint]
                vec2 = pose[child] - pose[joint]
                
                # Calculate angle
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
                angle_names.append(name)
            else:
                angles.append(np.nan)  # Missing angle
                angle_names.append(name + '_missing')
                
        return angles
    
    def _calculate_normalized_limb_lengths_robust(self, pose: np.ndarray, 
                                                reference_scale: float,
                                                visibility: Optional[np.ndarray]) -> List[float]:
        """Calculate normalized limb lengths, handling missing landmarks."""
        limb_connections = [
            (self.LEFT_SHOULDER, 13, 'left_upper_arm'),
            (13, 15, 'left_forearm'),
            (self.RIGHT_SHOULDER, 14, 'right_upper_arm'),
            (14, 16, 'right_forearm'),
        ]
        
        lengths = []
        for start, end, name in limb_connections:
            if self._check_landmarks_visible([start, end], visibility):
                length = np.linalg.norm(pose[end] - pose[start]) / reference_scale
                lengths.append(length)
            else:
                lengths.append(np.nan)  # Missing length
                
        return lengths
    
    def inverse_body_transform(self, normalized_pose: np.ndarray, 
                             transform_params: Dict) -> np.ndarray:
        """
        Convert body-relative coordinates back to original MediaPipe coordinates.
        
        Args:
            normalized_pose: Pose in body-relative coordinates
            transform_params: Dictionary with 'center', 'rotation', 'scale'
            
        Returns:
            Pose in original MediaPipe coordinates
        """
        center = transform_params['center']
        R = transform_params['rotation']
        scale = transform_params['scale']
        
        # Reverse the transformation
        original_pose = np.zeros_like(normalized_pose)
        for i in range(len(normalized_pose)):
            # Check for NaN values (truly missing landmarks)
            if np.any(np.isnan(normalized_pose[i])):
                # Keep NaN for missing landmarks (though this should be rare now)
                original_pose[i] = np.array([np.nan, np.nan, np.nan])
            else:
                # Undo scaling
                scaled_pos = normalized_pose[i] * scale
                # Undo rotation (apply R, which is the inverse of R.T)
                rotated_pos = R @ scaled_pos
                # Undo translation
                original_pose[i] = rotated_pos + center
            
        return original_pose
    
    def _extract_hand_features(self, hand: np.ndarray, 
                              wrist: np.ndarray) -> Tuple[Dict, Dict]:
        """Extract hand features in hand-local coordinate system."""
        
        # Translate to wrist-centered coordinates
        hand_centered = hand - wrist
        
        # Use palm size as scale reference
        # Palm landmarks in MediaPipe: 0 (wrist), 5 (index MCP), 17 (pinky MCP)
        palm_width = np.linalg.norm(hand[17] - hand[5])
        if palm_width > 0:
            hand_normalized = hand_centered / palm_width
        else:
            hand_normalized = hand_centered
            palm_width = 1.0  # Avoid division by zero
            
        # Store transformation parameters
        transform = {
            'wrist_position': wrist,
            'palm_width': palm_width,
            'original_shape': hand.shape
        }
        
        # Calculate finger joint angles
        finger_angles = self._calculate_finger_angles(hand)
        
        # Calculate inter-finger distances (important for sign language)
        finger_distances = self._calculate_finger_distances(hand_normalized)
        
        features = {
            'landmarks_normalized': hand_normalized,
            'finger_angles': finger_angles,
            'finger_distances': finger_distances,
            'hand_shape_features': self._extract_hand_shape(hand_normalized)
        }
        
        return features, transform
    
    def inverse_hand_transform(self, normalized_hand: np.ndarray, 
                             transform_params: Dict) -> np.ndarray:
        """
        Convert hand-relative coordinates back to original MediaPipe coordinates.
        
        Args:
            normalized_hand: Hand in hand-relative coordinates
            transform_params: Dictionary with 'wrist_position', 'palm_width'
            
        Returns:
            Hand in original MediaPipe coordinates
        """
        wrist_position = transform_params['wrist_position']
        palm_width = transform_params['palm_width']
        
        # Undo scaling
        hand_scaled = normalized_hand * palm_width
        # Undo translation
        original_hand = hand_scaled + wrist_position
        
        return original_hand
    
    def _extract_face_features(self, face: np.ndarray, 
                              pose: np.ndarray) -> Tuple[Dict, Dict]:
        """Extract facial features relative to head pose - compact or full mode."""
        
        # Use nose as reference point
        nose = pose[self.NOSE]
        
        # Estimate face size from pose landmarks
        face_scale = np.linalg.norm(pose[self.RIGHT_SHOULDER] - pose[self.LEFT_SHOULDER]) * 0.3
        
        if self.use_compact_face:
            # Compact mode: Extract only key landmarks (28 instead of 468)
            key_face_landmarks = face[self.COMPACT_LANDMARKS]
            face_centered = key_face_landmarks - nose
            face_normalized = face_centered / face_scale if face_scale > 0 else face_centered
            
            # Store transformation parameters for compact mode
            transform = {
                'nose_position': nose,
                'face_scale': face_scale,
                'key_landmark_indices': self.COMPACT_LANDMARKS,
                'original_shape': face.shape,
                'mode': 'compact'
            }
            
            # Extract semantic facial expression features
            semantic_features = self._extract_semantic_face_features(key_face_landmarks, nose, face_scale)
            
            features = {
                'landmarks_normalized': face_normalized,  # 28×3 = 84 features
                'semantic_features': semantic_features,   # 8 semantic features
                'num_landmarks': len(self.COMPACT_LANDMARKS),
                'mode': 'compact'
            }
            
        else:
            # Full mode: Extract all 468 facial landmarks
            face_centered = face - nose
            face_normalized = face_centered / face_scale if face_scale > 0 else face_centered
            
            # Store transformation parameters for full mode
            transform = {
                'nose_position': nose,
                'face_scale': face_scale,
                'original_shape': face.shape,
                'mode': 'full'
            }
            
            # For full mode, we can still extract semantic features from key landmarks
            key_face_landmarks = face[self.COMPACT_LANDMARKS]
            semantic_features = self._extract_semantic_face_features(key_face_landmarks, nose, face_scale)
            
            # Also extract some traditional facial features
            traditional_features = self._extract_traditional_face_features(face_normalized)
            
            features = {
                'landmarks_normalized': face_normalized,  # 468×3 = 1404 features
                'semantic_features': semantic_features,   # 8 semantic features  
                'traditional_features': traditional_features,  # Additional features
                'num_landmarks': len(face),
                'mode': 'full'
            }
        
        return features, transform
    
    def _extract_semantic_face_features(self, key_landmarks: np.ndarray, 
                                       nose_pos: np.ndarray, face_scale: float) -> Dict:
        """Extract semantic facial expression features from key landmarks."""
        features = {}
        
        # Mouth analysis (first 8 landmarks)
        mouth_landmarks = key_landmarks[:8]
        upper_lip = mouth_landmarks[0]      # Index 13: upper lip center
        lower_lip = mouth_landmarks[1]      # Index 14: lower lip center
        left_corner = mouth_landmarks[2]    # Index 61: left corner
        right_corner = mouth_landmarks[3]   # Index 291: right corner
        
        # Mouth openness (vertical distance between lips)
        mouth_height = np.linalg.norm(upper_lip - lower_lip) / face_scale
        
        # Mouth width (horizontal distance between corners)
        mouth_width = np.linalg.norm(right_corner - left_corner) / face_scale
        
        # Mouth center relative to nose
        mouth_center = (upper_lip + lower_lip) / 2
        mouth_to_nose = (mouth_center - nose_pos) / face_scale
        
        features.update({
            'mouth_openness': mouth_height,
            'mouth_width': mouth_width,
            'mouth_relative_x': mouth_to_nose[0],
            'mouth_relative_y': mouth_to_nose[1]
        })
        
        # Eyebrow analysis (next 6 landmarks)
        eyebrow_landmarks = key_landmarks[8:14]
        left_peak = eyebrow_landmarks[2]    # Index 105: left eyebrow peak
        right_peak = eyebrow_landmarks[5]   # Index 334: right eyebrow peak
        
        # Average eyebrow height relative to nose
        left_height = (left_peak[1] - nose_pos[1]) / face_scale
        right_height = (right_peak[1] - nose_pos[1]) / face_scale
        avg_eyebrow_height = (left_height + right_height) / 2
        
        # Eyebrow asymmetry
        eyebrow_asymmetry = abs(left_height - right_height)
        
        features.update({
            'eyebrow_raise': avg_eyebrow_height,
            'eyebrow_asymmetry': eyebrow_asymmetry
        })
        
        # Eye analysis (next 8 landmarks)
        eye_landmarks = key_landmarks[14:22]
        left_upper = eye_landmarks[2]       # Index 159: left eye upper lid
        left_lower = eye_landmarks[3]       # Index 145: left eye lower lid
        right_upper = eye_landmarks[6]      # Index 386: right eye upper lid
        right_lower = eye_landmarks[7]      # Index 374: right eye lower lid
        
        # Eye openness
        left_eye_openness = np.linalg.norm(left_upper - left_lower) / face_scale
        right_eye_openness = np.linalg.norm(right_upper - right_lower) / face_scale
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
        
        # Eye asymmetry
        eye_asymmetry = abs(left_eye_openness - right_eye_openness)
        
        features.update({
            'eye_openness': avg_eye_openness,
            'eye_asymmetry': eye_asymmetry
        })
        
        return features
    
    def _extract_traditional_face_features(self, face_normalized: np.ndarray) -> Dict:
        """Extract traditional facial features like eyebrow position and mouth shape."""
        features = {}
        
        # These are placeholder implementations - you can expand them as needed
        # Eyebrow position (using specific landmark indices)
        try:
            eyebrow_landmarks = face_normalized[self.KEY_FACE_LANDMARKS['EYEBROW']]
            eyebrow_height = np.mean(eyebrow_landmarks[:, 1])  # Y coordinate average
            features['eyebrow_position'] = eyebrow_height
        except:
            features['eyebrow_position'] = 0.0
        
        # Mouth shape features (using mouth landmark indices)
        try:
            mouth_landmarks = face_normalized[self.KEY_FACE_LANDMARKS['MOUTH']]
            mouth_width = np.max(mouth_landmarks[:, 0]) - np.min(mouth_landmarks[:, 0])
            mouth_height = np.max(mouth_landmarks[:, 1]) - np.min(mouth_landmarks[:, 1])
            features['mouth_width'] = mouth_width
            features['mouth_height'] = mouth_height
        except:
            features['mouth_width'] = 0.0
            features['mouth_height'] = 0.0
        
        return features
    
    def inverse_face_transform(self, normalized_face: np.ndarray, 
                             transform_params: Dict) -> np.ndarray:
        """
        Convert face-relative coordinates back to original MediaPipe coordinates.
        Handles both compact and full face modes.
        """
        nose_position = transform_params['nose_position']
        face_scale = transform_params['face_scale']
        mode = transform_params.get('mode', 'compact')
        
        # Undo scaling
        face_scaled = normalized_face * face_scale
        # Undo translation
        reconstructed_face = face_scaled + nose_position
        
        if mode == 'compact':
            # For compact mode, we need to reconstruct the full 468-landmark face
            # by placing the 28 key landmarks back in their original positions
            original_shape = transform_params['original_shape']
            key_indices = transform_params['key_landmark_indices']
            
            # Create full face array filled with nose position (reasonable default)
            full_face = np.tile(nose_position, (original_shape[0], 1))
            
            # Place the reconstructed key landmarks in their correct positions
            full_face[key_indices] = reconstructed_face
            
            return full_face
        else:
            # For full mode, we already have all landmarks
            return reconstructed_face
    
    def validate_transformation(self, original_landmarks: Dict) -> Dict:
        """
        Validate that the transformation is reversible by converting to invariant
        features and back, then comparing with original.
        
        Args:
            original_landmarks: Original MediaPipe landmarks
            
        Returns:
            Dictionary with validation metrics
        """
        # Extract features and transform parameters
        features, transform_params = self.extract_invariant_features(original_landmarks)
        
        validation_results = {}
        
        # Validate pose transformation
        if 'landmarks_normalized' in features:
            original_pose = np.array([[lm.x, lm.y, lm.z] 
                                    for lm in original_landmarks['pose'].landmark])
            normalized_pose = features['landmarks_normalized']
            reconstructed_pose = self.inverse_body_transform(normalized_pose, 
                                                           transform_params['body'])
            
            pose_error = np.mean(np.abs(original_pose - reconstructed_pose))
            validation_results['pose_reconstruction_error'] = pose_error
            validation_results['pose_max_error'] = np.max(np.abs(original_pose - reconstructed_pose))
        
        # Validate hand transformations
        for hand_side in ['left_hand', 'right_hand']:
            if hand_side in features and hand_side in original_landmarks:
                original_hand = np.array([[lm.x, lm.y, lm.z] 
                                        for lm in original_landmarks[hand_side].landmark])
                normalized_hand = features[hand_side]['landmarks_normalized']
                reconstructed_hand = self.inverse_hand_transform(normalized_hand, 
                                                               transform_params[hand_side])
                
                hand_error = np.mean(np.abs(original_hand - reconstructed_hand))
                validation_results[f'{hand_side}_reconstruction_error'] = hand_error
                validation_results[f'{hand_side}_max_error'] = np.max(np.abs(original_hand - reconstructed_hand))
        
        # Validate face transformation
        if 'face' in features and 'face' in original_landmarks:
            original_face = np.array([[lm.x, lm.y, lm.z] 
                                    for lm in original_landmarks['face'].landmark])
            normalized_face = features['face']['landmarks_normalized']
            reconstructed_face = self.inverse_face_transform(normalized_face, 
                                                           transform_params['face'])
            
            face_error = np.mean(np.abs(original_face - reconstructed_face))
            validation_results['face_reconstruction_error'] = face_error
            validation_results['face_max_error'] = np.max(np.abs(original_face - reconstructed_face))
        
        return validation_results
    
    def _extract_relationships(self, pose: np.ndarray, landmarks: Dict) -> Dict:
        """Extract spatial relationships between body parts."""
        relationships = {}
        
        # Hand to body relationships
        if 'left_hand' in landmarks:
            relationships['left_hand_to_face'] = self._hand_to_face_distance(
                pose[self.LEFT_WRIST], pose[self.NOSE], pose
            )
            relationships['left_hand_to_chest'] = self._hand_to_chest_position(
                pose[self.LEFT_WRIST], pose
            )
            
        if 'right_hand' in landmarks:
            relationships['right_hand_to_face'] = self._hand_to_face_distance(
                pose[self.RIGHT_WRIST], pose[self.NOSE], pose
            )
            relationships['right_hand_to_chest'] = self._hand_to_chest_position(
                pose[self.RIGHT_WRIST], pose
            )
            
        # Hand to hand relationship
        if 'left_hand' in landmarks and 'right_hand' in landmarks:
            relationships['hands_distance'] = self._normalized_distance(
                pose[self.LEFT_WRIST], pose[self.RIGHT_WRIST], pose
            )
            relationships['hands_relative_position'] = self._hands_relative_position(pose)
            
        return relationships
    
    def _calculate_joint_angles(self, pose: np.ndarray) -> List[float]:
        """Calculate angles between connected joints."""
        angles = []
        
        # Define joint connections (parent, joint, child)
        connections = [
            (self.LEFT_SHOULDER, 13, 15),  # Left elbow
            (self.RIGHT_SHOULDER, 14, 16),  # Right elbow
            (13, 15, 17),  # Left wrist
            (14, 16, 18),  # Right wrist
        ]
        
        for parent, joint, child in connections:
            vec1 = pose[parent] - pose[joint]
            vec2 = pose[child] - pose[joint]
            
            # Calculate angle
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)
            
        return angles
    
    def _calculate_normalized_limb_lengths(self, pose: np.ndarray, reference_scale: float) -> List[float]:
        """Calculate normalized limb lengths."""
        limb_connections = [
            (self.LEFT_SHOULDER, 13),  # Upper arm left
            (13, 15),  # Lower arm left
            (self.RIGHT_SHOULDER, 14),  # Upper arm right
            (14, 16),  # Lower arm right
        ]
        
        lengths = []
        for start, end in limb_connections:
            length = np.linalg.norm(pose[end] - pose[start]) / reference_scale
            lengths.append(length)
            
        return lengths
    
    def _calculate_finger_angles(self, hand: np.ndarray) -> List[float]:
        """Calculate angles for each finger joint."""
        angles = []
        
        # MediaPipe hand landmark indices for each finger
        fingers = {
            'thumb': [0, 1, 2, 3, 4],
            'index': [0, 5, 6, 7, 8],
            'middle': [0, 9, 10, 11, 12],
            'ring': [0, 13, 14, 15, 16],
            'pinky': [0, 17, 18, 19, 20]
        }
        
        for finger_name, indices in fingers.items():
            for i in range(len(indices) - 2):
                vec1 = hand[indices[i]] - hand[indices[i+1]]
                vec2 = hand[indices[i+2]] - hand[indices[i+1]]
                
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
                
        return angles
    
    def _calculate_finger_distances(self, hand: np.ndarray) -> List[float]:
        """Calculate distances between fingertips."""
        fingertips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        distances = []
        
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(hand[fingertips[i]] - hand[fingertips[j]])
                distances.append(dist)
                
        return distances
    
    def _extract_hand_shape(self, hand: np.ndarray) -> List[float]:
        """Extract hand shape features like openness, spread, etc."""
        features = []
        
        # Hand openness: average distance of fingertips from palm center
        palm_center = np.mean(hand[[0, 5, 9, 13, 17]], axis=0)
        fingertips = hand[[4, 8, 12, 16, 20]]
        openness = np.mean([np.linalg.norm(tip - palm_center) for tip in fingertips])
        features.append(openness)
        
        # Finger spread: variance of fingertip positions
        spread = np.var(fingertips, axis=0).sum()
        features.append(spread)
        
        return features
    
    def _calculate_eyebrow_position(self, face: np.ndarray) -> float:
        """Estimate eyebrow raise from facial landmarks."""
        # This is a placeholder - actual implementation depends on MediaPipe face landmark indices
        return 0.0
    
    def _calculate_mouth_shape(self, face: np.ndarray) -> List[float]:
        """Extract mouth shape parameters."""
        # Placeholder - extract features like mouth openness, width, etc.
        return [0.0, 0.0]
    
    def _hand_to_face_distance(self, hand_pos: np.ndarray, face_pos: np.ndarray, 
                              pose: np.ndarray) -> float:
        """Calculate normalized distance from hand to face."""
        shoulder_width = np.linalg.norm(pose[self.RIGHT_SHOULDER] - pose[self.LEFT_SHOULDER])
        distance = np.linalg.norm(hand_pos - face_pos) / shoulder_width
        return distance
    
    def _hand_to_chest_position(self, hand_pos: np.ndarray, pose: np.ndarray) -> List[float]:
        """Calculate hand position relative to chest in body coordinates."""
        chest_center = (pose[self.LEFT_SHOULDER] + pose[self.RIGHT_SHOULDER]) / 2
        relative_pos = hand_pos - chest_center
        shoulder_width = np.linalg.norm(pose[self.RIGHT_SHOULDER] - pose[self.LEFT_SHOULDER])
        return (relative_pos / shoulder_width).tolist()
    
    def _normalized_distance(self, pos1: np.ndarray, pos2: np.ndarray, 
                           pose: np.ndarray) -> float:
        """Calculate normalized distance between two positions."""
        shoulder_width = np.linalg.norm(pose[self.RIGHT_SHOULDER] - pose[self.LEFT_SHOULDER])
        return np.linalg.norm(pos1 - pos2) / shoulder_width
    
    def _hands_relative_position(self, pose: np.ndarray) -> List[float]:
        """Calculate relative position of hands."""
        left_hand = pose[self.LEFT_WRIST]
        right_hand = pose[self.RIGHT_WRIST]
        chest_center = (pose[self.LEFT_SHOULDER] + pose[self.RIGHT_SHOULDER]) / 2
        
        # Express in body coordinate system
        left_rel = left_hand - chest_center
        right_rel = right_hand - chest_center
        
        shoulder_width = np.linalg.norm(pose[self.RIGHT_SHOULDER] - pose[self.LEFT_SHOULDER])
        
        return np.concatenate([left_rel / shoulder_width, right_rel / shoulder_width]).tolist()


# Example usage and validation
def demonstrate_reversibility():
    """
    Demonstrate that the transformation is reversible.
    """
    # Create a mock MediaPipe result (you would use real data)
    # This is just for demonstration
    class MockLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
    
    class MockLandmarkList:
        def __init__(self, landmarks):
            self.landmark = landmarks
    
    # Create sample pose landmarks (33 landmarks for MediaPipe Pose)
    pose_landmarks = []
    for i in range(33):
        # Random positions for demonstration
        x, y, z = np.random.rand(3)
        pose_landmarks.append(MockLandmark(x, y, z))
    
    # Create sample hand landmarks (21 landmarks for MediaPipe Hand)
    hand_landmarks = []
    for i in range(21):
        x, y, z = np.random.rand(3)
        hand_landmarks.append(MockLandmark(x, y, z))
    
    # Create mock landmarks dictionary
    mock_landmarks = {
        'pose': MockLandmarkList(pose_landmarks),
        'left_hand': MockLandmarkList(hand_landmarks),
        'right_hand': MockLandmarkList(hand_landmarks)
    }
    
    # Create extractor and validate
    extractor = SignLanguageFeatureExtractor()
    validation_results = extractor.validate_transformation(mock_landmarks)
    
    print("Validation Results:")
    for key, value in validation_results.items():
        print(f"{key}: {value:.6f}")
    
    # The errors should be very small (close to machine precision)
    # if the transformation is correctly implemented
    
    return validation_results


# Function to visualize original vs reconstructed landmarks
def visualize_transformation(original_landmarks, reconstructed_landmarks):
    """
    Create a visual comparison of original and reconstructed landmarks.
    This would typically use matplotlib or another visualization library.
    """
    # Placeholder for visualization code
    pass


# === JSON File Processing Functions ===

# Landmark constants matching the web interface format
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21
POSE_FEATURES = NUM_POSE_LANDMARKS * 3    # 99
FACE_FEATURES = NUM_FACE_LANDMARKS * 3    # 1404
HAND_FEATURES = NUM_HAND_LANDMARKS * 3    # 63
TOTAL_FEATURES = POSE_FEATURES + FACE_FEATURES + 2 * HAND_FEATURES  # 1629

# Feature order: Pose, Face, Left Hand, Right Hand
POSE_START_IDX = 0
POSE_END_IDX = POSE_FEATURES
FACE_START_IDX = POSE_END_IDX
FACE_END_IDX = FACE_START_IDX + FACE_FEATURES
LH_START_IDX = FACE_END_IDX
LH_END_IDX = LH_START_IDX + HAND_FEATURES
RH_START_IDX = LH_END_IDX
RH_END_IDX = RH_START_IDX + HAND_FEATURES


class MockLandmark:
    """Mock landmark class compatible with MediaPipe format."""
    def __init__(self, x, y, z, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockLandmarkList:
    """Mock landmark list compatible with MediaPipe format."""
    def __init__(self, landmarks):
        self.landmark = landmarks


def flat_array_to_landmarks_dict(flat_frame: list) -> dict:
    """
    Convert flat 1629-feature array to MediaPipe-style landmarks dictionary.
    
    Args:
        flat_frame: List of 1629 features in order [pose, face, left_hand, right_hand]
        
    Returns:
        Dictionary with 'pose', 'face', 'left_hand', 'right_hand' landmark objects
    """
    if len(flat_frame) != TOTAL_FEATURES:
        raise ValueError(f"Expected {TOTAL_FEATURES} features, got {len(flat_frame)}")
    
    landmarks_dict = {}
    
    # Extract pose landmarks (33 landmarks)
    pose_data = flat_frame[POSE_START_IDX:POSE_END_IDX]
    pose_landmarks = []
    for i in range(0, len(pose_data), 3):
        pose_landmarks.append(MockLandmark(pose_data[i], pose_data[i+1], pose_data[i+2]))
    landmarks_dict['pose'] = MockLandmarkList(pose_landmarks)
    
    # Extract face landmarks (468 landmarks)
    face_data = flat_frame[FACE_START_IDX:FACE_END_IDX]
    face_landmarks = []
    for i in range(0, len(face_data), 3):
        face_landmarks.append(MockLandmark(face_data[i], face_data[i+1], face_data[i+2]))
    landmarks_dict['face'] = MockLandmarkList(face_landmarks)
    
    # Extract left hand landmarks (21 landmarks)
    lh_data = flat_frame[LH_START_IDX:LH_END_IDX]
    lh_landmarks = []
    for i in range(0, len(lh_data), 3):
        lh_landmarks.append(MockLandmark(lh_data[i], lh_data[i+1], lh_data[i+2]))
    landmarks_dict['left_hand'] = MockLandmarkList(lh_landmarks)
    
    # Extract right hand landmarks (21 landmarks)
    rh_data = flat_frame[RH_START_IDX:RH_END_IDX]
    rh_landmarks = []
    for i in range(0, len(rh_data), 3):
        rh_landmarks.append(MockLandmark(rh_data[i], rh_data[i+1], rh_data[i+2]))
    landmarks_dict['right_hand'] = MockLandmarkList(rh_landmarks)
    
    return landmarks_dict


def landmarks_dict_to_flat_array(landmarks_dict: dict) -> list:
    """
    Convert MediaPipe-style landmarks dictionary to flat 1629-feature array.
    
    Args:
        landmarks_dict: Dictionary with 'pose', 'face', 'left_hand', 'right_hand'
        
    Returns:
        List of 1629 features in order [pose, face, left_hand, right_hand]
    """
    flat_frame = []
    
    # Add pose landmarks
    if 'pose' in landmarks_dict and landmarks_dict['pose']:
        for landmark in landmarks_dict['pose'].landmark:
            flat_frame.extend([landmark.x, landmark.y, landmark.z])
    else:
        flat_frame.extend([0.0] * POSE_FEATURES)
    
    # Add face landmarks
    if 'face' in landmarks_dict and landmarks_dict['face']:
        for landmark in landmarks_dict['face'].landmark:
            flat_frame.extend([landmark.x, landmark.y, landmark.z])
    else:
        flat_frame.extend([0.0] * FACE_FEATURES)
    
    # Add left hand landmarks
    if 'left_hand' in landmarks_dict and landmarks_dict['left_hand']:
        for landmark in landmarks_dict['left_hand'].landmark:
            flat_frame.extend([landmark.x, landmark.y, landmark.z])
    else:
        flat_frame.extend([0.0] * HAND_FEATURES)
    
    # Add right hand landmarks
    if 'right_hand' in landmarks_dict and landmarks_dict['right_hand']:
        for landmark in landmarks_dict['right_hand'].landmark:
            flat_frame.extend([landmark.x, landmark.y, landmark.z])
    else:
        flat_frame.extend([0.0] * HAND_FEATURES)
    
    return flat_frame


def process_json_to_features_with_mode(input_json_path: str, output_json_path: str, use_compact_face: bool = True):
    """
    Convert raw landmark JSON file to features JSON file with specified face mode.
    
    Args:
        input_json_path: Path to input JSON file with raw 1629-feature landmarks
        output_json_path: Path to save features JSON file
        use_compact_face: Whether to use compact face features (28 landmarks) or full face features (468 landmarks)
    """
    print(f"Converting raw landmarks to features...")
    print(f"Input: {input_json_path}")
    print(f"Output: {output_json_path}")
    print(f"Face mode: {'compact' if use_compact_face else 'full'}")
    
    # Load raw landmark data
    try:
        with open(input_json_path, 'r') as f:
            raw_sequence = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    if not isinstance(raw_sequence, list):
        raise ValueError("Input JSON must be a list of frame arrays")
    
    print(f"Loaded {len(raw_sequence)} frames from input file")
    
    extractor = SignLanguageFeatureExtractor(use_compact_face=use_compact_face)
    features_sequence = []
    transform_params_sequence = []
    
    for frame_idx, flat_frame in enumerate(raw_sequence):
        try:
            if len(flat_frame) != TOTAL_FEATURES:
                print(f"Warning: Frame {frame_idx} has {len(flat_frame)} features, expected {TOTAL_FEATURES}")
                continue
            
            # Convert flat array to landmarks dict
            landmarks_dict = flat_array_to_landmarks_dict(flat_frame)
            
            # Extract features
            features, transform_params = extractor.extract_invariant_features(landmarks_dict)
            
            features_sequence.append(features)
            transform_params_sequence.append(transform_params)
            
            if (frame_idx + 1) % 10 == 0:
                print(f"Processed {frame_idx + 1}/{len(raw_sequence)} frames")
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            print(f"Frame data type: {type(flat_frame)}, length: {len(flat_frame) if hasattr(flat_frame, '__len__') else 'N/A'}")
            # Add a dummy entry to maintain sequence
            features_sequence.append({'error': f'Frame {frame_idx} failed: {str(e)}'})
            transform_params_sequence.append({'error': True})
            continue
    
    # Save features and transform parameters
    print(f"Saving {len(features_sequence)} processed frames...")
    
    output_data = {
        'features_sequence': features_sequence,
        'transform_params_sequence': transform_params_sequence,
        'metadata': {
            'num_frames': len(features_sequence),
            'original_num_frames': len(raw_sequence),
            'input_file': input_json_path,
            'feature_extractor_version': '1.0',
            'face_mode': 'compact' if use_compact_face else 'full'
        }
    }
    
    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Features extracted and saved to: {output_json_path}")
        print(f"Processed {len(features_sequence)} frames successfully")
    except Exception as e:
        print(f"Error saving output file: {e}")
        print(f"Attempting to save with default JSON encoder...")
        # Try with a more robust JSON serialization
        try:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.int64):
                    return int(obj)
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(output_json_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=convert_numpy)
            print(f"Successfully saved with custom JSON encoder")
        except Exception as e2:
            print(f"Failed to save even with custom encoder: {e2}")
            return


def process_json_to_features(input_json_path: str, output_json_path: str):
    """
    Convert raw landmark JSON file to features JSON file.
    Uses compact face mode by default for backward compatibility.
    
    Args:
        input_json_path: Path to input JSON file with raw 1629-feature landmarks
        output_json_path: Path to save features JSON file
    """
    process_json_to_features_with_mode(input_json_path, output_json_path, use_compact_face=True)


def process_features_to_landmarks(features_json_path: str, output_json_path: str):
    """
    Convert features JSON file back to raw landmark JSON file.
    
    Args:
        features_json_path: Path to features JSON file
        output_json_path: Path to save reconstructed landmarks JSON file
    """
    print(f"Converting features back to raw landmarks...")
    print(f"Input: {features_json_path}")
    print(f"Output: {output_json_path}")
    
    # Load features data
    with open(features_json_path, 'r') as f:
        features_data = json.load(f)
    
    features_sequence = features_data['features_sequence']
    transform_params_sequence = features_data['transform_params_sequence']
    
    extractor = SignLanguageFeatureExtractor()
    reconstructed_sequence = []
    
    for frame_idx, (features, transform_params) in enumerate(zip(features_sequence, transform_params_sequence)):
        try:
            # Reconstruct landmarks for each component
            reconstructed_landmarks = {}
            
            # Reconstruct pose
            if features.get('pose_available', False):
                pose_features = features.get('landmarks_normalized')
                if pose_features is not None:
                    pose_array = np.array(pose_features).reshape(-1, 3)
                    reconstructed_pose = extractor.inverse_body_transform(pose_array, transform_params.get('body', {}))
                    pose_landmarks = [MockLandmark(p[0], p[1], p[2]) for p in reconstructed_pose]
                    reconstructed_landmarks['pose'] = MockLandmarkList(pose_landmarks)
            
            # Reconstruct left hand
            if (isinstance(features.get('left_hand'), dict) and 
                features['left_hand'].get('landmarks_normalized') is not None):
                lh_features = features['left_hand']['landmarks_normalized']
                lh_array = np.array(lh_features).reshape(-1, 3)
                reconstructed_lh = extractor.inverse_hand_transform(lh_array, transform_params.get('left_hand', {}))
                lh_landmarks = [MockLandmark(p[0], p[1], p[2]) for p in reconstructed_lh]
                reconstructed_landmarks['left_hand'] = MockLandmarkList(lh_landmarks)
            
            # Reconstruct right hand
            if (isinstance(features.get('right_hand'), dict) and 
                features['right_hand'].get('landmarks_normalized') is not None):
                rh_features = features['right_hand']['landmarks_normalized']
                rh_array = np.array(rh_features).reshape(-1, 3)
                reconstructed_rh = extractor.inverse_hand_transform(rh_array, transform_params.get('right_hand', {}))
                rh_landmarks = [MockLandmark(p[0], p[1], p[2]) for p in reconstructed_rh]
                reconstructed_landmarks['right_hand'] = MockLandmarkList(rh_landmarks)
            
            # Reconstruct face
            if (isinstance(features.get('face'), dict) and 
                features['face'].get('landmarks_normalized') is not None):
                face_features = features['face']['landmarks_normalized']
                face_array = np.array(face_features).reshape(-1, 3)
                reconstructed_face = extractor.inverse_face_transform(face_array, transform_params.get('face', {}))
                face_landmarks = [MockLandmark(p[0], p[1], p[2]) for p in reconstructed_face]
                reconstructed_landmarks['face'] = MockLandmarkList(face_landmarks)
            
            # Convert back to flat array
            flat_frame = landmarks_dict_to_flat_array(reconstructed_landmarks)
            reconstructed_sequence.append(flat_frame)
            
        except Exception as e:
            print(f"Error reconstructing frame {frame_idx}: {e}")
            # Add a zero frame as fallback
            reconstructed_sequence.append([0.0] * TOTAL_FEATURES)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"Reconstructed {frame_idx + 1}/{len(features_sequence)} frames")
    
    # Save reconstructed landmarks
    with open(output_json_path, 'w') as f:
        json.dump(reconstructed_sequence, f, separators=(',', ':'))  # Compact format like original
    
    print(f"Landmarks reconstructed and saved to: {output_json_path}")
    print(f"Reconstructed {len(reconstructed_sequence)} frames")


def compare_landmark_files(original_path: str, reconstructed_path: str):
    """
    Compare original and reconstructed landmark files to validate the pipeline.
    
    Args:
        original_path: Path to original landmark JSON file
        reconstructed_path: Path to reconstructed landmark JSON file
    """
    print(f"Comparing original and reconstructed landmark files...")
    print(f"Original: {original_path}")
    print(f"Reconstructed: {reconstructed_path}")
    
    # Load both files
    with open(original_path, 'r') as f:
        original_data = json.load(f)
    
    with open(reconstructed_path, 'r') as f:
        reconstructed_data = json.load(f)
    
    if len(original_data) != len(reconstructed_data):
        print(f"WARNING: Frame count mismatch. Original: {len(original_data)}, Reconstructed: {len(reconstructed_data)}")
        min_frames = min(len(original_data), len(reconstructed_data))
        print(f"Comparing first {min_frames} frames only.")
    else:
        min_frames = len(original_data)
        print(f"Comparing {min_frames} frames...")
    
    # Calculate differences
    max_error = 0.0
    mean_error = 0.0
    total_comparisons = 0
    frame_errors = []
    
    for frame_idx in range(min_frames):
        orig_frame = original_data[frame_idx]
        recon_frame = reconstructed_data[frame_idx]
        
        if len(orig_frame) != len(recon_frame):
            print(f"WARNING: Feature count mismatch in frame {frame_idx}")
            continue
        
        # Calculate frame-wise error
        frame_error = 0.0
        for i in range(len(orig_frame)):
            error = abs(orig_frame[i] - recon_frame[i])
            frame_error += error
            mean_error += error
            max_error = max(max_error, error)
            total_comparisons += 1
        
        frame_errors.append(frame_error / len(orig_frame))
    
    mean_error /= total_comparisons
    mean_frame_error = np.mean(frame_errors)
    
    print(f"\n=== Comparison Results ===")
    print(f"Frames compared: {min_frames}")
    print(f"Total feature comparisons: {total_comparisons}")
    print(f"Maximum absolute error: {max_error:.8f}")
    print(f"Mean absolute error: {mean_error:.8f}")
    print(f"Mean per-frame error: {mean_frame_error:.8f}")
    print(f"Error standard deviation: {np.std(frame_errors):.8f}")
    
    # Determine if the reconstruction is acceptable
    if max_error < 1e-6:
        print(f"EXCELLENT: Reconstruction is nearly perfect (max error < 1e-6)")
    elif max_error < 1e-4:
        print(f"GOOD: Reconstruction is very good (max error < 1e-4)")
    elif max_error < 1e-2:
        print(f"ACCEPTABLE: Reconstruction has some error but may be acceptable (max error < 1e-2)")
    else:
        print(f"POOR: Reconstruction has significant error (max error >= 1e-2)")
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'mean_frame_error': mean_frame_error,
        'std_frame_error': np.std(frame_errors),
        'frames_compared': min_frames,
        'total_comparisons': total_comparisons
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Test sign language feature extraction pipeline with JSON files"
    )
    
    # Add face mode parameter
    parser.add_argument('--face_mode', choices=['compact', 'full'], default='compact',
                       help='Face feature extraction mode: compact (28 landmarks, ~92 features) or full (468 landmarks, ~1407 features)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Command: raw_to_features
    parser_raw2feat = subparsers.add_parser(
        'raw_to_features',
        help='Convert raw landmark JSON to features JSON'
    )
    parser_raw2feat.add_argument('input_json', help='Input raw landmark JSON file')
    parser_raw2feat.add_argument('output_json', help='Output features JSON file')
    
    # Command: features_to_raw
    parser_feat2raw = subparsers.add_parser(
        'features_to_raw',
        help='Convert features JSON back to raw landmark JSON'
    )
    parser_feat2raw.add_argument('input_json', help='Input features JSON file')
    parser_feat2raw.add_argument('output_json', help='Output reconstructed landmark JSON file')
    
    # Command: compare
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare original and reconstructed landmark files'
    )
    parser_compare.add_argument('original_json', help='Original landmark JSON file')
    parser_compare.add_argument('reconstructed_json', help='Reconstructed landmark JSON file')
    
    # Command: full_test
    parser_test = subparsers.add_parser(
        'full_test',
        help='Run full pipeline test: raw -> features -> raw -> compare'
    )
    parser_test.add_argument('input_json', help='Input raw landmark JSON file')
    parser_test.add_argument('--temp_dir', default='temp_test', 
                           help='Temporary directory for intermediate files')
    
    args = parser.parse_args()
    
    # Determine face mode
    use_compact_face = (args.face_mode == 'compact')
    print(f"Using face feature mode: {args.face_mode}")
    if use_compact_face:
        print("  - Compact mode: 28 key landmarks, ~92 face features")
    else:
        print("  - Full mode: 468 landmarks, ~1407 face features")
    
    if args.command == 'raw_to_features':
        # Update to use the face mode parameter
        def process_json_with_face_mode(input_path, output_path):
            process_json_to_features_with_mode(input_path, output_path, use_compact_face)
        process_json_with_face_mode(args.input_json, args.output_json)
    
    elif args.command == 'features_to_raw':
        process_features_to_landmarks(args.input_json, args.output_json)
    
    elif args.command == 'compare':
        compare_landmark_files(args.original_json, args.reconstructed_json)
    
    elif args.command == 'full_test':
        # Run complete pipeline test
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(exist_ok=True)
        
        input_path = Path(args.input_json)
        features_path = temp_dir / f"{input_path.stem}_features_{args.face_mode}.json"
        reconstructed_path = temp_dir / f"{input_path.stem}_reconstructed_{args.face_mode}.json"
        
        print("=== Starting Full Pipeline Test ===")
        print(f"1. Converting raw landmarks to features ({args.face_mode} face mode)...")
        process_json_to_features_with_mode(str(input_path), str(features_path), use_compact_face)
        
        print(f"\n2. Converting features back to landmarks...")
        process_features_to_landmarks(str(features_path), str(reconstructed_path))
        
        print(f"\n3. Comparing original vs reconstructed...")
        results = compare_landmark_files(str(input_path), str(reconstructed_path))
        
        print(f"\n=== Full Pipeline Test Complete ===")
        print(f"Intermediate files saved in: {temp_dir}")
        print(f"Features file: {features_path}")
        print(f"Reconstructed file: {reconstructed_path}")
        
        return results
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()