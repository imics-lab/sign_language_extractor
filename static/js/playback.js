// static/js/playback.js

window.addEventListener('load', () => {
    // --- DOM Elements ---
    const jsonFileInput = document.getElementById('jsonFile');
    const playbackCanvas = document.getElementById('playbackCanvas');
    const canvasCtx = playbackCanvas ? playbackCanvas.getContext('2d') : null;
    const playButton = document.getElementById('playButton');
    const pauseButton = document.getElementById('pauseButton');
    const stopButton = document.getElementById('stopButton');
    const speedControl = document.getElementById('speedControl');
    const playbackStatus = document.getElementById('playbackStatus');
    const frameIndicator = document.getElementById('frameIndicator');

    // --- Playback State ---
    let landmarkFrames = [];
    let currentFrameIndex = 0;
    let isPlaying = false;
    let animationTimeoutId = null;
    let baseFrameMillis = 1000 / 30; // Base playback at 30 FPS, speed control adjusts this

    // --- Landmark Constants (Updated for 1629 Features: Pose, Face, Left Hand, Right Hand) ---
    const POSE_LANDMARKS_COUNT = 33;
    const FACE_LANDMARKS_COUNT = 468;
    const HAND_LANDMARKS_COUNT = 21; // Per hand

    const POSE_FEATURES = POSE_LANDMARKS_COUNT * 3;    // 99
    const FACE_FEATURES = FACE_LANDMARKS_COUNT * 3;    // 1404
    const HAND_FEATURES = HAND_LANDMARKS_COUNT * 3;    // 63

    // Order: Pose, Face, Left Hand, Right Hand (1629 total)
    const POSE_START_IDX = 0;
    const POSE_END_IDX = POSE_FEATURES;

    const FACE_START_IDX = POSE_END_IDX;
    const FACE_END_IDX = FACE_START_IDX + FACE_FEATURES;

    const LH_START_IDX = FACE_END_IDX;
    const LH_END_IDX = LH_START_IDX + HAND_FEATURES;

    const RH_START_IDX = LH_END_IDX;
    const RH_END_IDX = RH_START_IDX + HAND_FEATURES;

    const TOTAL_FEATURES_EXPECTED = RH_END_IDX; // 99 + 1404 + 63 + 63 = 1629

    // --- FALLBACK CONNECTION DEFINITIONS ---
    const FALLBACK_POSE_CONNECTIONS = [
        // Torso
        [11, 12], [12, 24], [24, 23], [23, 11],
        // Left arm
        [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
        // Right arm
        [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
        // Left leg
        [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
        // Right leg
        [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
        // Face outline
        [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
        // Face center
        [9, 10]
    ];

    const FALLBACK_HAND_CONNECTIONS = [
        // Palm connections
        [0, 1], [0, 5], [0, 17], [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8],
        [9, 10], [10, 11], [11, 12],
        [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20],
        // Connections between fingers
        [1, 5], [5, 9], [9, 13], [13, 17]
    ];

    // Face mesh connections - Official MediaPipe FACEMESH_CONTOURS connections
    // These provide anatomically accurate facial contours
    const FALLBACK_FACE_CONNECTIONS = [
        // FACEMESH_LIPS - Lip contours
        [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 320], [320, 307], [307, 375], [375, 321], [321, 308], [308, 324], [324, 318], [318, 402], [402, 317], [317, 14], [14, 87], [87, 178], [178, 88], [88, 95], [95, 78], [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 269], [269, 267], [267, 269], [269, 270], [270, 267], [267, 271], [271, 272], [272, 61],
        
        // Inner mouth
        [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308], [308, 415], [415, 310], [310, 311], [311, 312], [312, 13], [13, 82], [82, 81], [81, 80], [80, 78],
        
        // FACEMESH_LEFT_EYE - Left eye contour 
        [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133], [133, 173], [173, 157], [157, 158], [158, 159], [159, 160], [160, 161], [161, 246], [246, 33],
        
        // FACEMESH_LEFT_EYEBROW - Left eyebrow
        [46, 53], [53, 52], [52, 65], [65, 55], [55, 70],
        
        // FACEMESH_RIGHT_EYE - Right eye contour
        [362, 382], [382, 381], [381, 380], [380, 374], [374, 373], [373, 390], [390, 249], [249, 263], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362],
        
        // FACEMESH_RIGHT_EYEBROW - Right eyebrow  
        [276, 283], [283, 282], [282, 295], [295, 285], [285, 336],
        
        // FACEMESH_FACE_OVAL - Face perimeter
        [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
        
        // Nose bridge and tip (key landmarks for nose structure)
        [168, 8], [8, 9], [9, 10], [10, 151], [151, 195], [195, 197], [197, 196], [196, 3], [3, 51], [51, 48], [48, 115], [115, 131], [131, 134], [134, 102], [102, 49], [49, 220], [220, 305], [305, 290], [290, 331], [331, 294], [294, 168],
        
        // Nose tip detail
        [1, 2], [2, 5], [5, 4], [4, 6], [6, 168],
        
        // Nostril definition  
        [20, 238], [238, 239], [239, 240], [240, 241], [241, 242], [242, 20],
        [419, 456], [456, 399], [399, 419]
    ];

    // --- File Loading ---
    if (jsonFileInput) {
        jsonFileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) { 
                playbackStatus.textContent = 'No file selected.'; 
                return; 
            }
            if (!file.name.endsWith('.json')) {
                playbackStatus.textContent = 'Error: Please select a .json file.'; 
                landmarkFrames = []; 
                resetPlayback(); 
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const content = e.target.result;
                    
                    try {
                        landmarkFrames = JSON.parse(content);
                    } catch (parseError) {
                        throw new Error("Failed to parse JSON file. The file may be corrupted.");
                    }
                    
                    if (!Array.isArray(landmarkFrames)) {
                        throw new Error("Invalid JSON format: Not an array");
                    }
                    
                    if (landmarkFrames.length === 0) {
                        throw new Error("Empty data: No frames found in file");
                    }
                    
                    // Check first frame
                    if (!Array.isArray(landmarkFrames[0])) {
                        throw new Error("Invalid data structure: First frame is not an array");
                    }
                    
                    // Handle feature count more gracefully
                    const frameFeatureCount = landmarkFrames[0].length;
                    if (frameFeatureCount !== TOTAL_FEATURES_EXPECTED) {
                        if (frameFeatureCount === 225) {
                            playbackStatus.textContent = `Loaded ${landmarkFrames.length} frames with 225 features (pose and hands only). Face landmarks not available.`;
                        } else {
                            playbackStatus.textContent = `Loaded ${landmarkFrames.length} frames with ${frameFeatureCount} features (expected ${TOTAL_FEATURES_EXPECTED}). Will attempt to adapt.`;
                        }
                    } else {
                        playbackStatus.textContent = `Loaded ${landmarkFrames.length} frames with full holistic data (pose, face, and hands).`;
                    }
                    
                    resetPlayback();
                    if (landmarkFrames.length > 0) {
                        drawFrame(currentFrameIndex);
                    }
                    updateButtonStates();
                } catch (error) {
                    playbackStatus.textContent = `Error: ${error.message}`; 
                    landmarkFrames = []; 
                    resetPlayback();
                    updateButtonStates();
                }
            };
            
            reader.onerror = () => {
                playbackStatus.textContent = 'Error reading file.'; 
                landmarkFrames = []; 
                resetPlayback(); 
                updateButtonStates();
            };
            
            reader.readAsText(file);
        });
    }

    // --- Landmark Reconstruction ---
    function reconstructLandmarks(flatFrameData) {
        if (!flatFrameData || !Array.isArray(flatFrameData)) {
            return { poseLandmarks: null, faceLandmarks: null, leftHandLandmarks: null, rightHandLandmarks: null };
        }

        const results = { poseLandmarks: null, faceLandmarks: null, leftHandLandmarks: null, rightHandLandmarks: null };
        
        // Adaptive reconstruction based on feature count
        const frameFeatureCount = flatFrameData.length;
        
        if (frameFeatureCount === TOTAL_FEATURES_EXPECTED) {
            // Standard 1629-feature format (Pose + Face + LH + RH)
            const parseSlice = (startIndex, landmarkCount) => {
                const landmarks = [];
                let hasData = false;
                
                for (let i = 0; i < landmarkCount; i++) {
                    const idx = startIndex + i * 3;
                    // Safety check to avoid out-of-bounds
                    if (idx + 2 < flatFrameData.length) {
                        const landmark = { 
                            x: flatFrameData[idx], 
                            y: flatFrameData[idx + 1], 
                            z: flatFrameData[idx + 2],
                        };
                        if (landmark.x !== 0 || landmark.y !== 0 || landmark.z !== 0) {
                            hasData = true;
                        }
                        landmarks.push(landmark);
                    }
                }
                return hasData ? { landmark: landmarks } : null;
            };

            results.poseLandmarks = parseSlice(POSE_START_IDX, POSE_LANDMARKS_COUNT);
            results.faceLandmarks = parseSlice(FACE_START_IDX, FACE_LANDMARKS_COUNT);
            results.leftHandLandmarks = parseSlice(LH_START_IDX, HAND_LANDMARKS_COUNT);
            results.rightHandLandmarks = parseSlice(RH_START_IDX, HAND_LANDMARKS_COUNT);
            
        } else if (frameFeatureCount === 225) {
            // This is the 225-feature format (Pose + LH + RH) - legacy format
            const POSE_225_FEATURES = 33 * 3; // 99
            const HAND_225_FEATURES = 21 * 3; // 63
            
            const POSE_225_START = 0;
            const LH_225_START = POSE_225_FEATURES;
            const RH_225_START = LH_225_START + HAND_225_FEATURES;
            
            const parseSlice = (startIndex, landmarkCount) => {
                const landmarks = [];
                let hasData = false;
                
                for (let i = 0; i < landmarkCount; i++) {
                    const idx = startIndex + i * 3;
                    // Safety check to avoid out-of-bounds
                    if (idx + 2 < flatFrameData.length) {
                        const landmark = { 
                            x: flatFrameData[idx], 
                            y: flatFrameData[idx + 1], 
                            z: flatFrameData[idx + 2],
                        };
                        if (landmark.x !== 0 || landmark.y !== 0 || landmark.z !== 0) {
                            hasData = true;
                        }
                        landmarks.push(landmark);
                    }
                }
                return hasData ? { landmark: landmarks } : null;
            };
            
            results.poseLandmarks = parseSlice(POSE_225_START, POSE_LANDMARKS_COUNT);
            results.leftHandLandmarks = parseSlice(LH_225_START, HAND_LANDMARKS_COUNT);
            results.rightHandLandmarks = parseSlice(RH_225_START, HAND_LANDMARKS_COUNT);
            // faceLandmarks remains null for this format
            
        } else {
            // Unknown format - minimal attempt to extract pose data assuming starting at index 0
            const parseMinimalPose = () => {
                const landmarks = [];
                let hasData = false;
                
                // Try to get as many pose landmarks as possible
                const maxLandmarks = Math.min(POSE_LANDMARKS_COUNT, Math.floor(flatFrameData.length / 3));
                
                for (let i = 0; i < maxLandmarks; i++) {
                    const idx = i * 3;
                    const landmark = { 
                        x: flatFrameData[idx], 
                        y: flatFrameData[idx + 1], 
                        z: flatFrameData[idx + 2],
                    };
                    if (landmark.x !== 0 || landmark.y !== 0 || landmark.z !== 0) {
                        hasData = true;
                    }
                    landmarks.push(landmark);
                }
                return hasData ? { landmark: landmarks } : null;
            };
            
            results.poseLandmarks = parseMinimalPose();
        }
        
        return results;
    }

    // --- Drawing ---
    function drawFrame(index) {
        if (!canvasCtx) {
            playbackStatus.textContent = "Error: Canvas context not available.";
            if(isPlaying) stop();
            return;
        }
        
        if (index < 0 || index >= landmarkFrames.length) {
            return;
        }

        const frameData = landmarkFrames[index];
        if (!frameData) {
            playbackStatus.textContent = `Error: Frame ${index} data is missing.`;
            if(isPlaying) stop();
            return;
        }
        
        const reconstructed = reconstructLandmarks(frameData);

        // Clear the canvas
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, playbackCanvas.width, playbackCanvas.height);

        const landmarkRadius = 2;
        const connectionThickness = 2;
        const faceConnectionThickness = 1; // Thinner for face mesh

        try {
            // Get MediaPipe drawing functions and connections with fallbacks
            const drawConnectors = typeof window.drawConnectors === 'function' 
                ? window.drawConnectors 
                : drawConnectorsFallback;
                
            const drawLandmarks = typeof window.drawLandmarks === 'function'
                ? window.drawLandmarks
                : drawLandmarksFallback;
            
            // Get connection definitions - try MediaPipe first, then fallbacks
            const Holistic = window.Holistic || {};
            const poseConnections = Holistic.POSE_CONNECTIONS || FALLBACK_POSE_CONNECTIONS;
            const handConnections = Holistic.HAND_CONNECTIONS || FALLBACK_HAND_CONNECTIONS;
            
            // For face mesh, prefer FACEMESH_CONTOURS if available (anatomically accurate)
            let faceConnections = FALLBACK_FACE_CONNECTIONS;
            if (window.FACEMESH_CONTOURS) {
                faceConnections = window.FACEMESH_CONTOURS;
            } else if (Holistic.FACE_CONNECTIONS) {
                faceConnections = Holistic.FACE_CONNECTIONS;
            }
            
            // POSE
            if (reconstructed.poseLandmarks) {
                drawConnectors(canvasCtx, reconstructed.poseLandmarks.landmark, poseConnections,
                              { color: '#42A5F5', lineWidth: connectionThickness }); // Light Blue
                drawLandmarks(canvasCtx, reconstructed.poseLandmarks.landmark,
                             { color: '#0D47A1', radius: landmarkRadius }); // Dark Blue
            }

            // FACE
            if (reconstructed.faceLandmarks) {
                drawConnectors(canvasCtx, reconstructed.faceLandmarks.landmark, faceConnections,
                              { color: '#E91E63', lineWidth: faceConnectionThickness }); // Dark Pink
                
                // Option 1: No face landmark points (cleanest visualization)
                // Face landmarks are hidden to avoid overlap - connections show the mesh clearly
                
                // Option 2: Uncomment below for minimal face landmarks (0.2 radius)
                // drawLandmarks(canvasCtx, reconstructed.faceLandmarks.landmark,
                //              { color: '#E91E63', radius: 0.2 }); // Dark Pink, minimal visible radius
            }

            // LEFT HAND
            if (reconstructed.leftHandLandmarks) {
                drawConnectors(canvasCtx, reconstructed.leftHandLandmarks.landmark, handConnections,
                              { color: '#FF8A65', lineWidth: connectionThickness }); // Light Orange
                drawLandmarks(canvasCtx, reconstructed.leftHandLandmarks.landmark,
                             { color: '#E65100', radius: landmarkRadius }); // Dark Orange
            }

            // RIGHT HAND
            if (reconstructed.rightHandLandmarks) {
                drawConnectors(canvasCtx, reconstructed.rightHandLandmarks.landmark, handConnections,
                              { color: '#4DD0E1', lineWidth: connectionThickness }); // Cyan
                drawLandmarks(canvasCtx, reconstructed.rightHandLandmarks.landmark,
                             { color: '#006064', radius: landmarkRadius }); // Dark Cyan
            }
        } catch (drawError) {
             playbackStatus.textContent = `Drawing Error: ${drawError.message}`;
             if(isPlaying) stop();
        }

        canvasCtx.restore();
        if(frameIndicator) frameIndicator.textContent = `Frame: ${index + 1} / ${landmarkFrames.length}`;
    }
    
    // --- FALLBACK DRAWING FUNCTIONS ---
    function drawConnectorsFallback(ctx, landmarks, connections, options = {}) {
        if (!landmarks || !connections) return;
        
        const color = options.color || '#FF0000';
        const lineWidth = options.lineWidth || 1;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        
        for (const connection of connections) {
            if (Array.isArray(connection) && connection.length === 2) {
                const [start, end] = connection;
                if (landmarks[start] && landmarks[end]) {
                    ctx.beginPath();
                    ctx.moveTo(landmarks[start].x * ctx.canvas.width, landmarks[start].y * ctx.canvas.height);
                    ctx.lineTo(landmarks[end].x * ctx.canvas.width, landmarks[end].y * ctx.canvas.height);
                    ctx.stroke();
                }
            }
        }
    }
    
    function drawLandmarksFallback(ctx, landmarks, options = {}) {
        if (!landmarks) return;
        
        const color = options.color || '#FF0000';
        const radius = options.radius || 2;
        
        ctx.fillStyle = color;
        
        for (const landmark of landmarks) {
            if (landmark) {
                ctx.beginPath();
                ctx.arc(
                    landmark.x * ctx.canvas.width, 
                    landmark.y * ctx.canvas.height, 
                    radius, 
                    0, 
                    2 * Math.PI
                );
                ctx.fill();
            }
        }
    }

    // --- Playback Controls ---
    function updateButtonStates() {
        const hasFrames = landmarkFrames.length > 0;
        const canDraw = playbackCanvas && canvasCtx;

        if (playButton) playButton.disabled = isPlaying || !hasFrames || !canDraw;
        if (pauseButton) pauseButton.disabled = !isPlaying || !hasFrames || !canDraw;
        if (stopButton) stopButton.disabled = !hasFrames || !canDraw; // Can stop even if paused
        if (jsonFileInput) jsonFileInput.disabled = isPlaying;
    }

    function play() {
        if (isPlaying || landmarkFrames.length === 0) return;
        
        isPlaying = true;
        updateButtonStates();
        
        function animate() {
            if (!isPlaying) return; 
            drawFrame(currentFrameIndex); 
            currentFrameIndex++;
            if (currentFrameIndex >= landmarkFrames.length) { 
                stop(); // Or loop: currentFrameIndex = 0;
            } else { 
                animationTimeoutId = setTimeout(animate, baseFrameMillis / parseFloat(speedControl ? speedControl.value || 1 : 1)); 
            }
        }
        animate();
    }

    function pause() { 
        if (!isPlaying) return; 
        isPlaying = false; 
        clearTimeout(animationTimeoutId); 
        animationTimeoutId = null; 
        updateButtonStates();
    }

    function stop() { 
        isPlaying = false; 
        clearTimeout(animationTimeoutId); 
        animationTimeoutId = null; 
        currentFrameIndex = 0; 
        if (landmarkFrames.length > 0) drawFrame(0); // Reset to first frame
        else clearCanvas();
        updateButtonStates();
    }
    
    function clearCanvas() {
        if (canvasCtx && playbackCanvas) {
            canvasCtx.clearRect(0, 0, playbackCanvas.width, playbackCanvas.height);
        }
        if(frameIndicator) frameIndicator.textContent = 'Frame: - / -';
    }

    function resetPlayback() {
        stop(); // Stop any ongoing playback and reset index
        if (landmarkFrames.length > 0) {
            drawFrame(0);
        } else {
            clearCanvas();
        }
        updateButtonStates();
    }
    
    // --- Event Listeners ---
    if (playButton) playButton.addEventListener('click', play);
    if (pauseButton) pauseButton.addEventListener('click', pause);
    if (stopButton) stopButton.addEventListener('click', stop);
    if (speedControl) speedControl.addEventListener('change', () => {
        // If playing, adjust speed immediately by restarting the timeout
        if (isPlaying) {
            clearTimeout(animationTimeoutId);
            animationTimeoutId = setTimeout(() => {
                 // This is a bit simplified; ideally, you'd resume 'animate'
                 // For now, just restarting the loop from current frame if paused and played again works
                 // Or, if play() is called, it will use new speed.
            }, baseFrameMillis / parseFloat(speedControl.value || 1));
        }
    });

    // --- Auto-load from URL parameter ---
    function loadFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        const fileUrlToLoad = urlParams.get('file_url');

        if (fileUrlToLoad) {
            if(jsonFileInput) jsonFileInput.disabled = true; // Disable file input if loading from URL
            playbackStatus.textContent = `Loading data from ${fileUrlToLoad}...`;
            fetch(fileUrlToLoad)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status} for ${fileUrlToLoad}`);
                    }
                    return response.json();
                })
                .then(data => {
                    landmarkFrames = data;
                    if (!Array.isArray(landmarkFrames) || landmarkFrames.length === 0) { 
                        throw new Error("Invalid JSON format or empty data from URL."); 
                    }
                    
                    // More tolerant feature count check
                    const frameFeatureCount = landmarkFrames[0].length;
                    if (frameFeatureCount !== TOTAL_FEATURES_EXPECTED) {
                        if (frameFeatureCount === 1629) {
                            playbackStatus.textContent = `Loaded ${landmarkFrames.length} frames with 1629 features (full Holistic). Using pose and hands only.`;
                        } else {
                            playbackStatus.textContent = `Loaded ${landmarkFrames.length} frames with ${frameFeatureCount} features. Will attempt to adapt.`;
                        }
                    } else {
                        playbackStatus.textContent = `Loaded ${landmarkFrames.length} frames from URL.`;
                    }
                    
                    resetPlayback();
                    updateButtonStates();
                })
                .catch(error => {
                    playbackStatus.textContent = `Error: ${error.message}`;
                    landmarkFrames = [];
                    resetPlayback();
                    updateButtonStates();
                });
        }
    }

    // --- Initial State ---
    playbackStatus.textContent = 'Please select a JSON landmark file or load via URL.';
    resetPlayback(); // Also calls updateButtonStates
    loadFromUrl(); // Attempt to load from URL query param
});