"""
Video Processing, Face Detection, and PhysFormer rPPG Heart Rate Estimation Engine.
Supports CUDA GPU acceleration with automatic CPU fallback.
Performs strictly Video -> Face Detection/ROI -> PhysFormer -> rPPG -> Heart Rate (BPM).
"""

import os
import tempfile
import gc
import numpy as np
import cv2
import torch
import streamlit as st
from scipy.signal import butter, filtfilt, welch, detrend

from models.physformer.Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp

# Configure PyTorch CPU threading to prevent saturating all cores and starving Streamlit's WebSocket loop
try:
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
except Exception:
    pass

# Global in-memory singleton cache to guarantee zero-latency re-use even outside Streamlit cache
_CACHED_PHYSFORMER_MODEL = None

# Model configuration path
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Physformer_VIPL_fold1.pkl")
PHYSFORMER_MODEL_PATH = os.getenv("PHYSFORMER_MODEL_PATH", DEFAULT_MODEL_PATH)


def get_torch_device():
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_physformer_model(model_path=None):
    """
    Cached singleton loader for PhysFormer model.
    Loads checkpoint onto CPU/CUDA device without reallocating on each request.
    """
    global _CACHED_PHYSFORMER_MODEL
    if _CACHED_PHYSFORMER_MODEL is not None and (model_path is None or _CACHED_PHYSFORMER_MODEL.get("model_path") == model_path):
        return _CACHED_PHYSFORMER_MODEL

    if model_path is None:
        model_path = PHYSFORMER_MODEL_PATH

    # If the default path doesn't exist, search common locations
    if not os.path.exists(model_path):
        candidates = [
            model_path,
            os.path.join(os.path.dirname(__file__), "..", "models", "Physformer_VIPL_fold1.pkl"),
            os.path.join(os.path.dirname(__file__), "..", "Physformer_VIPL_fold1.pkl"),
            "models/Physformer_VIPL_fold1.pkl",
            "Physformer_VIPL_fold1.pkl"
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                found = True
                break
        if not found:
            raise FileNotFoundError(f"PhysFormer checkpoint not found at: {model_path}")

    device = get_torch_device()
    
    # Instantiate official PhysFormer architecture (VIPL fold 1 configuration)
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(160, 128, 128),
        patches=(4, 4, 4),
        dim=96,
        ff_dim=144,
        num_heads=4,
        num_layers=12,
        dropout_rate=0.1,
        theta=0.7
    ).to(device)

    # Load checkpoint weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Strip any 'module.' prefixes from DataParallel if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[key] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    _CACHED_PHYSFORMER_MODEL = {
        "model": model,
        "device": device,
        "model_path": model_path,
        "loaded": True
    }

    return _CACHED_PHYSFORMER_MODEL


def detect_face_roi(first_frames, is_pre_cropped=False):
    """
    Detects face ROI in video frames using OpenCV Haar Cascades with multi-stage fallback.
    Returns:
        bbox: (x1, y1, x2, y2)
        annotated_frame_rgb: np.ndarray image with face bounding box
        face_found: bool
    """
    h_orig, w_orig = first_frames[0].shape[:2]
    
    # If the user specified that the video is already pre-cropped (e.g. VIPL-HR / UBFC face crop)
    if is_pre_cropped:
        bbox = (0, 0, w_orig, h_orig)
        annotated = first_frames[0].copy()
        cv2.rectangle(annotated, (4, 4), (w_orig - 4, h_orig - 4), (75, 99, 230), 2)
        return bbox, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), True

    # Try default and alt2 Haar cascades
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    ]
    
    found_box = None
    annotated_sample = first_frames[0].copy()

    for casc_path in cascade_paths:
        if not os.path.exists(casc_path):
            continue
        cascade = cv2.CascadeClassifier(casc_path)
        
        # Test across first few frames to find strongest detection
        for idx in range(min(15, len(first_frames))):
            gray = cv2.cvtColor(first_frames[idx], cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
            
            if len(faces) > 0:
                # Pick largest detected face
                best_face = max(faces, key=lambda b: b[2] * b[3])
                x, y, w, h = best_face
                
                # Expand box by 10% horizontally and 15% vertically for optimal forehead/cheek rPPG coverage
                pad_w = int(w * 0.10)
                pad_h = int(h * 0.15)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w_orig, x + w + pad_w)
                y2 = min(h_orig, y + h + pad_h)
                
                found_box = (x1, y1, x2, y2)
                annotated_sample = first_frames[idx].copy()
                break
        
        if found_box is not None:
            break

    if found_box is not None:
        x1, y1, x2, y2 = found_box
        # Draw bounding box on sample frame for user verification
        cv2.rectangle(annotated_sample, (x1, y1), (x2, y2), (75, 99, 230), 3)
        cv2.putText(
            annotated_sample, "Face ROI Detected", (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (75, 99, 230), 2, cv2.LINE_AA
        )
        return found_box, cv2.cvtColor(annotated_sample, cv2.COLOR_BGR2RGB), True
    else:
        # Fallback to centered crop if face not detected
        center_w, center_h = int(w_orig * 0.65), int(h_orig * 0.65)
        x1 = (w_orig - center_w) // 2
        y1 = (h_orig - center_h) // 2
        x2 = x1 + center_w
        y2 = y1 + center_h
        cv2.rectangle(annotated_sample, (x1, y1), (x2, y2), (230, 99, 75), 2)
        cv2.putText(
            annotated_sample, "Auto-Centered Crop (Face Detector Soft Fallback)", (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 99, 75), 2, cv2.LINE_AA
        )
        return (x1, y1, x2, y2), cv2.cvtColor(annotated_sample, cv2.COLOR_BGR2RGB), False


def extract_video_tensor(video_path, is_pre_cropped=False, num_frames=160):
    """
    Extracts video frames, detects face ROI, crops and resizes to (160, 128, 128, 3),
    normalizes to [-1, 1], and returns PyTorch tensor of shape (1, 3, 160, 128, 128).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file. Please verify format and integrity.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sanitize FPS
    if fps <= 0 or np.isnan(fps) or fps > 120:
        fps = 30.0

    if total_frames < num_frames:
        cap.release()
        raise ValueError(
            f"Video is too short ({total_frames} frames detected). "
            f"PhysFormer requires at least {num_frames} frames (~{num_frames / fps:.1f} seconds) "
            "for reliable spatio-temporal attention and frequency-domain analysis."
        )

    # Read frames
    raw_frames = []
    while len(raw_frames) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
    cap.release()

    # If webcam or stream had minor frame drop, pad up to num_frames with last frame
    if 120 <= len(raw_frames) < num_frames:
        last_f = raw_frames[-1]
        while len(raw_frames) < num_frames:
            raw_frames.append(last_f.copy())

    if len(raw_frames) < num_frames:
        raise ValueError(
            f"Unable to extract {num_frames} valid video frames (extracted {len(raw_frames)} frames)."
        )

    # Detect face ROI on initial sample frames
    bbox, annotated_frame_rgb, face_found = detect_face_roi(raw_frames[:25], is_pre_cropped=is_pre_cropped)
    x1, y1, x2, y2 = bbox

    # Crop ROI and resize each frame to 128x128
    processed_frames = []
    for f in raw_frames:
        cropped = f[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = f
        resized = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize into [-1, 1] per PhysFormer official spec: (image - 127.5) / 128.0
        norm = (rgb.astype(np.float32) - 127.5) / 128.0
        processed_frames.append(norm)

    # Stack into tensor: (1, 3, 160, 128, 128)
    np_clip = np.array(processed_frames)  # (160, 128, 128, 3)
    np_clip = np_clip.transpose((3, 0, 1, 2))  # (3, 160, 128, 128)
    tensor = torch.from_numpy(np_clip).unsqueeze(0).float()

    return tensor, fps, total_frames, bbox, annotated_frame_rgb, face_found


def calculate_hr_from_rppg(rppg_raw, fps):
    """
    Converts raw rPPG signal into estimated Heart Rate (BPM) via:
    1. Signal detrending (drift removal)
    2. Butterworth bandpass filtering [0.75 Hz - 2.5 Hz] (45 to 150 BPM)
    3. Welch Power Spectral Density (PSD) analysis
    4. Peak frequency identification: BPM = f_peak * 60
    5. Signal-to-Noise Ratio (SNR) and confidence estimation
    """
    rppg_raw = np.array(rppg_raw, dtype=np.float64)
    n = len(rppg_raw)
    
    # 1. Detrend
    detrended = detrend(rppg_raw)

    # 2. 2nd-order Butterworth bandpass filter: [0.75, 2.5] Hz
    nyquist = 0.5 * fps
    low_cutoff = 0.75 / nyquist
    high_cutoff = min(2.5 / nyquist, 0.99)
    
    if low_cutoff >= high_cutoff or high_cutoff >= 1.0:
        low_cutoff = 0.05
        high_cutoff = 0.95

    b, a = butter(2, [low_cutoff, high_cutoff], btype='band')
    filtered_rppg = filtfilt(b, a, detrended)

    # Normalize filtered signal to standard unit scale
    filtered_rppg = (filtered_rppg - np.mean(filtered_rppg)) / (np.std(filtered_rppg) + 1e-8)

    # 3. Frequency domain analysis via Welch PSD
    nperseg = min(n, 128)
    freqs, psd = welch(filtered_rppg, fs=fps, nperseg=nperseg, noverlap=nperseg // 2)

    # Constrain to physiological human heart rate band [0.75, 2.5] Hz (45 to 150 BPM)
    valid_mask = (freqs >= 0.75) & (freqs <= 2.5)
    if not np.any(valid_mask):
        valid_mask = (freqs > 0)

    freqs_valid = freqs[valid_mask]
    psd_valid = psd[valid_mask]

    # Find peak frequency
    peak_idx = np.argmax(psd_valid)
    peak_freq = freqs_valid[peak_idx]
    estimated_bpm = float(peak_freq * 60.0)

    # 4. SNR Estimation
    # Signal power within +/- 0.15 Hz around peak and its 2nd harmonic
    peak_band = (freqs_valid >= (peak_freq - 0.15)) & (freqs_valid <= (peak_freq + 0.15))
    p_signal = np.sum(psd_valid[peak_band])
    p_total = np.sum(psd_valid)
    p_noise = max(p_total - p_signal, 1e-9)

    snr_db = 10.0 * np.log10(max(p_signal / p_noise, 1e-3))
    
    # 5. Quality classification and confidence mapping
    if snr_db >= 5.0:
        quality = "High"
        confidence = float(np.clip(0.85 + (snr_db - 5.0) * 0.02, 0.85, 0.96))
    elif snr_db >= 2.0:
        quality = "Good"
        confidence = float(np.clip(0.75 + (snr_db - 2.0) * 0.03, 0.75, 0.85))
    elif snr_db >= 0.0:
        quality = "Moderate"
        confidence = float(np.clip(0.60 + snr_db * 0.07, 0.60, 0.75))
    else:
        quality = "Low / Noisy"
        confidence = float(np.clip(0.50 + (snr_db + 5.0) * 0.02, 0.40, 0.60))

    # Time axis in seconds
    time_axis = np.arange(n) / fps

    return {
        "heart_rate": round(estimated_bpm, 1),
        "dominant_freq_hz": round(float(peak_freq), 3),
        "snr_db": round(float(snr_db), 2),
        "signal_quality": quality,
        "confidence_score": round(confidence, 2),
        "filtered_rppg": filtered_rppg,
        "time_axis": time_axis,
        "freqs": freqs_valid,
        "psd": psd_valid
    }


def process_face_video(video_source, is_pre_cropped=False, model_path=None, progress_callback=None):
    """
    Full end-to-end inference pipeline:
    Patient Video -> Face Detection/ROI -> PhysFormer -> rPPG Signal -> Heart Rate (BPM)
    
    Args:
        video_source: file path or UploadedFile bytes from Streamlit
        is_pre_cropped: bool, true if video is already cropped around face
        model_path: optional custom path to PhysFormer checkpoint
        progress_callback: optional callback func(percent, message)
        
    Returns:
        Structured dictionary with heart rate, rPPG signals, frequency spectrum, and quality metrics.
    """
    if progress_callback:
        progress_callback(0.1, "Validating video format and reading stream...")

    # Handle Streamlit UploadedFile or raw bytes by writing to temporary file
    temp_file = None
    if hasattr(video_source, "read") or isinstance(video_source, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        if hasattr(video_source, "read"):
            temp_file.write(video_source.read())
        else:
            temp_file.write(video_source)
        temp_file.flush()
        video_path = temp_file.name
    elif isinstance(video_source, str) and os.path.exists(video_source):
        video_path = video_source
    else:
        raise ValueError("Invalid video source provided.")

    try:
        if progress_callback:
            progress_callback(0.20, "Extracting video frames and tracking facial ROI...")

        # Extract frames, detect face, normalize tensor
        tensor, fps, total_frames, bbox, annotated_frame_rgb, face_found = extract_video_tensor(
            video_path, is_pre_cropped=is_pre_cropped, num_frames=160
        )

        if progress_callback:
            progress_callback(0.45, "Loading PhysFormer Spatio-Temporal ViT model...")

        # Load model singleton (cached in memory)
        model_dict = load_physformer_model(model_path)
        model = model_dict["model"]
        device = model_dict["device"]

        if progress_callback:
            progress_callback(0.65, "Executing PhysFormer deep temporal cross-attention...")

        # Throttle PyTorch CPU threads so Tornado WebSocket stays fast and responsive
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 4))
        except Exception:
            pass

        # Run PhysFormer forward pass with inference_mode for maximum speed & minimal RAM
        with torch.inference_mode():
            tensor_dev = tensor.to(device)
            # Official PhysFormer inference with gra_sharp=2.0
            rppg_raw_tensor, _, _, _ = model(tensor_dev, gra_sharp=2.0)
            rppg_raw = rppg_raw_tensor.cpu().squeeze().numpy()

        # Promptly release intermediate tensors to free RAM
        del tensor_dev
        del tensor
        del rppg_raw_tensor
        gc.collect()

        if progress_callback:
            progress_callback(0.88, "Applying Butterworth bandpass filter & Welch frequency analysis...")

        # Compute Heart Rate from rPPG
        hr_results = calculate_hr_from_rppg(rppg_raw, fps)

        # Build clinical impression and recommended action
        bpm = hr_results["heart_rate"]
        if bpm < 60:
            status_text = "Resting Bradycardia Pattern"
            impression = f"Estimated heart rate is {bpm:.1f} BPM, which falls into the bradycardic range (< 60 BPM). Physiological in trained athletes or resting individuals."
            action = "Correlate with resting activity and standard contact pulse oximetry or ECG if symptomatic."
        elif bpm <= 100:
            status_text = "Normal Resting Heart Rate"
            impression = f"Estimated heart rate is {bpm:.1f} BPM, within the typical physiological resting adult range (60–100 BPM)."
            action = "Routine monitoring recommended as part of multimodal health screening."
        else:
            status_text = "Resting Tachycardia Pattern"
            impression = f"Estimated heart rate is {bpm:.1f} BPM, elevated above the standard resting range (> 100 BPM). May reflect stress, exertion, or physiological factors."
            action = "Recommend resting state verification and follow-up clinical assessment."

        response = {
            "status": "success",
            "heart_rate": bpm,
            "unit": "BPM",
            "status_classification": status_text,
            "confidence_score": hr_results["confidence_score"],
            "signal_quality": hr_results["signal_quality"],
            "snr_db": hr_results["snr_db"],
            "fps": round(float(fps), 2),
            "total_frames": total_frames,
            "frames_analyzed": 160,
            "duration_analyzed_sec": round(160.0 / fps, 2),
            "dominant_freq_hz": hr_results["dominant_freq_hz"],
            "rppg_signal": hr_results["filtered_rppg"].tolist(),
            "raw_rppg": rppg_raw.tolist(),
            "time_axis": hr_results["time_axis"].tolist(),
            "psd_freqs": hr_results["freqs"].tolist(),
            "psd_power": hr_results["psd"].tolist(),
            "face_detected": face_found,
            "face_bbox": bbox,
            "annotated_frame": annotated_frame_rgb,
            "clinical_impression": impression,
            "recommended_action": action,
            "disclaimer": (
                "AI/video-based investigational estimation using PhysFormer rPPG. "
                "Not intended as a primary diagnostic tool or certified clinical measurement."
            ),
            "device": str(device)
        }

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        return response

    finally:
        # Clean up temporary video file if created
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass


def record_webcam_video(
    output_path="scratch/recorded_camera_video.mp4",
    target_frames=160,
    camera_index=0,
    frame_callback=None
):
    """
    Captures target_frames (default 160 frames, ~5.3 seconds at 30 fps) from the local webcam
    and writes them to an MP4 video file.
    Calls frame_callback(current_frame, total_frames, rgb_frame) for live UI updates.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not access the camera. Please ensure a webcam is connected and not in use by another application.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps) or fps > 120:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None

    frames_captured = 0
    try:
        while frames_captured < target_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            writer.write(frame)
            frames_captured += 1

            if frame_callback is not None:
                display_frame = frame.copy()
                # Draw face detection bounding box on live preview
                if cascade is not None and frames_captured % 4 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(60, 60))
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (75, 99, 230), 2)
                        cv2.putText(
                            display_frame, "Face Detected", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (75, 99, 230), 2
                        )
                
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_callback(frames_captured, target_frames, rgb_frame)
    finally:
        cap.release()
        writer.release()

    return output_path, frames_captured, fps


# Pre-warm PhysFormer model into singleton cache on load
try:
    if os.path.exists(PHYSFORMER_MODEL_PATH):
        load_physformer_model(PHYSFORMER_MODEL_PATH)
except Exception:
    pass


