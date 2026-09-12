import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.video_processor import (

    process_face_video,
    record_webcam_video,
    load_physformer_model
)
from utils.pdf_report import generate_vital_signs_report

os.makedirs('scratch/test_videos', exist_ok=True)

print('=====================================================')
print('   JIVA VITAL SIGNS COMPREHENSIVE TEST SUITE')
print('=====================================================')

# Preload model once
print('\n[Setup] Preloading PhysFormer model...')
t0 = time.time()
m_info = load_physformer_model()
print(f"PhysFormer loaded in {time.time()-t0:.2f}s on device: {m_info['device']}")

results_summary = []

# ---------------------------------------------------------------
# TEST 1: Standard Frontal Face (.mp4, 30 fps, 180 frames, ~72 BPM)
# ---------------------------------------------------------------
print('\n[Test 1] Generating and testing Standard Frontal Face Video (.mp4, 30 fps, ~72 BPM)...')
vid1_path = 'scratch/test_videos/video1_standard_72bpm.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter(vid1_path, fourcc, 30.0, (480, 480))

img_base = cv2.imread('scratch/lena.jpg')
if img_base is None:
    img_base = np.ones((480, 480, 3), dtype=np.uint8) * 200
else:
    img_base = cv2.resize(img_base, (480, 480))

# 72 BPM = 1.20 Hz
for i in range(180):
    t = i / 30.0
    pulse = 2.0 * np.sin(2 * np.pi * 1.20 * t)
    f = img_base.astype(np.float32)
    f[:, :, 1] = np.clip(f[:, :, 1] + pulse, 0, 255)
    f[:, :, 2] = np.clip(f[:, :, 2] + pulse * 0.7, 0, 255)
    out1.write(f.astype(np.uint8))
out1.release()

res1 = process_face_video(vid1_path)
print(f"  Result: Status={res1['status']}, HR={res1['heart_rate']} {res1['unit']}, Classification='{res1['status_classification']}', Quality={res1['signal_quality']}, SNR={res1['snr_db']}dB, FaceDetected={res1['face_detected']}")
results_summary.append(('Test 1: Standard MP4 (72 BPM target)', res1['status'] == 'success', f"{res1['heart_rate']} BPM (SNR: {res1['snr_db']} dB)"))

# ---------------------------------------------------------------
# TEST 2: High Frame Rate (.avi, 60 fps, 240 frames, ~90 BPM)
# ---------------------------------------------------------------
print('\n[Test 2] Generating and testing High Frame Rate Video (.avi, 60 fps, ~90 BPM)...')
vid2_path = 'scratch/test_videos/video2_highfps_90bpm.avi'
fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter(vid2_path, fourcc2, 60.0, (480, 480))

# 90 BPM = 1.50 Hz
for i in range(240):
    t = i / 60.0
    pulse = 2.2 * np.sin(2 * np.pi * 1.50 * t)
    f = img_base.astype(np.float32)
    f[:, :, 1] = np.clip(f[:, :, 1] + pulse, 0, 255)
    f[:, :, 2] = np.clip(f[:, :, 2] + pulse * 0.8, 0, 255)
    out2.write(f.astype(np.uint8))
out2.release()

res2 = process_face_video(vid2_path)
print(f"  Result: Status={res2['status']}, HR={res2['heart_rate']} {res2['unit']}, Classification='{res2['status_classification']}', Quality={res2['signal_quality']}, SNR={res2['snr_db']}dB, FPS={res2['fps']}")
results_summary.append(('Test 2: High FPS AVI (60 fps, 90 BPM target)', res2['status'] == 'success', f"{res2['heart_rate']} BPM (FPS: {res2['fps']})"))

# ---------------------------------------------------------------
# TEST 3: Pre-cropped Face Video (VIPL-HR format, 25 fps, ~60 BPM)
# ---------------------------------------------------------------
print('\n[Test 3] Generating and testing Pre-cropped Benchmark Video (25 fps, ~60 BPM, is_pre_cropped=True)...')
vid3_path = 'scratch/test_videos/video3_precropped_60bpm.mp4'
fourcc3 = cv2.VideoWriter_fourcc(*'mp4v')
out3 = cv2.VideoWriter(vid3_path, fourcc3, 25.0, (128, 128))
face_crop = cv2.resize(img_base[150:350, 150:350], (128, 128))

# 60 BPM = 1.00 Hz
for i in range(160):
    t = i / 25.0
    pulse = 2.5 * np.sin(2 * np.pi * 1.00 * t)
    f = face_crop.astype(np.float32)
    f[:, :, 1] = np.clip(f[:, :, 1] + pulse, 0, 255)
    out3.write(f.astype(np.uint8))
out3.release()

res3 = process_face_video(vid3_path, is_pre_cropped=True)
print(f"  Result: Status={res3['status']}, HR={res3['heart_rate']} {res3['unit']}, Quality={res3['signal_quality']}, FPS={res3['fps']}, BBox={res3['face_bbox']}")
results_summary.append(('Test 3: Pre-cropped VIPL format (60 BPM target)', res3['status'] == 'success', f"{res3['heart_rate']} BPM (BBox: {res3['face_bbox']})"))

# ---------------------------------------------------------------
# TEST 4: Edge Case: Video Too Short (50 frames < 160 required)
# ---------------------------------------------------------------
print('\n[Test 4] Testing Error Handling on Short Video (50 frames)...')
vid4_path = 'scratch/test_videos/video4_too_short.mp4'
out4 = cv2.VideoWriter(vid4_path, fourcc, 30.0, (256, 256))
for i in range(50):
    out4.write(np.zeros((256, 256, 3), dtype=np.uint8))
out4.release()

short_error_caught = False
try:
    process_face_video(vid4_path)
except ValueError as ve:
    short_error_caught = True
    print(f"  Correctly caught expected ValueError: {ve}")
results_summary.append(('Test 4: Error Handling - Video Too Short (50 frames)', short_error_caught, 'Caught ValueError with descriptive message'))

# ---------------------------------------------------------------
# TEST 5: Live Webcam Recording & Processing
# ---------------------------------------------------------------
print('\n[Test 5] Testing Live Webcam Video Recording (160 frames from webcam)...')
rec_path = 'scratch/test_videos/video5_live_webcam.mp4'

def dummy_callback(curr, total, frame):
    if curr % 40 == 0 or curr == total:
        print(f"  Webcam capture progress: {curr}/{total} frames ({curr*100//total}%)")

try:
    out_rec, frames_rec, fps_rec = record_webcam_video(
        output_path=rec_path,
        target_frames=160,
        frame_callback=dummy_callback
    )
    print(f"  Webcam recorded {frames_rec} frames at {fps_rec:.1f} fps -> {rec_path} ({os.path.getsize(rec_path)} bytes)")
    
    print('  Passing live webcam video through PhysFormer pipeline...')
    res5 = process_face_video(rec_path)
    print(f"  Live Webcam Result: Status={res5['status']}, HR={res5['heart_rate']} {res5['unit']}, Quality={res5['signal_quality']}, FaceDetected={res5['face_detected']}, SNR={res5['snr_db']}dB, FPS={res5['fps']}")
    results_summary.append(('Test 5: Live Webcam Recording & Inference', res5['status'] == 'success', f"{res5['heart_rate']} BPM, Face: {res5['face_detected']}, Quality: {res5['signal_quality']}"))

    # Also test PDF report generation on this live result
    print('\n[Test 6] Testing PDF Report Generation on Live Webcam Result...')
    fig_w, ax1 = plt.subplots(figsize=(6, 2))
    ax1.plot(res5['time_axis'], res5['rppg_signal'])
    fig_s, ax2 = plt.subplots(figsize=(6, 2))
    ax2.plot(res5['psd_freqs'], res5['psd_power'])
    
    pdf_bytes = generate_vital_signs_report(
        findings=res5,
        fig_waveform=fig_w,
        fig_spectrum=fig_s,
        face_image=res5.get('annotated_frame')
    )
    pdf_ok = pdf_bytes is not None and len(pdf_bytes) > 50000
    print(f"  PDF Report generated: {len(pdf_bytes)} bytes (Valid: {pdf_ok})")
    results_summary.append(('Test 6: PDF Report on Live Webcam Result', pdf_ok, f"{len(pdf_bytes):,} bytes valid PDF"))

except Exception as e:
    print(f"  Webcam/Live error: {e}")
    results_summary.append(('Test 5 & 6: Live Webcam', False, str(e)))

print('\n=====================================================')
print('                   TEST SUMMARY                      ')
print('=====================================================')
all_pass = True
for name, passed, detail in results_summary:
    mark = 'PASS' if passed else 'FAIL'
    if not passed: all_pass = False
    print(f"[{mark}] {name} -> {detail}")

print(f"\nOVERALL TEST OUTCOME: {'ALL TESTS PASSED SUCCESSFULLY!' if all_pass else 'SOME TESTS FAILED'}")
