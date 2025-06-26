# Player Re-Identification in a Single Feed: Approach and Report

## Approach and Methodology

- **Object Detection:**
  - Used a fine-tuned Ultralytics YOLOv11 model (`best.pt`) to detect players in each frame of the video.
  - Frames were resized to 640x360 for faster inference.
  - Only every other frame was processed to further speed up the pipeline.

- **Tracking and Re-Identification:**
  - Integrated DeepSORT for multi-object tracking and re-identification.
  - Each detected player was assigned a unique ID, which was maintained even if the player left and re-entered the frame.
  - Detections were filtered and formatted to match DeepSORT's expected input.

- **Performance Optimizations:**
  - Used GPU acceleration when available.
  - Skipped frames and resized input to balance speed and accuracy.
  - Optionally disabled display for benchmarking.

## Techniques Tried and Outcomes

- **Frame Skipping:**
  - Processing every other frame nearly doubled the speed with minimal impact on tracking quality.
- **Frame Resizing:**
  - Reduced computational load and improved real-time performance.
- **YOLO + DeepSORT Integration:**
  - Successfully maintained consistent player IDs across occlusions and re-entries in a single video feed.
- **Error Handling:**
  - Added robust checks for detection formatting and empty frames to prevent runtime errors.

## Challenges Encountered

- **Detection Format Compatibility:**
  - DeepSORT required a specific detection tuple format, which caused errors until corrected.
- **Speed vs. Accuracy Tradeoff:**
  - Skipping too many frames or resizing too aggressively can reduce tracking accuracy.
- **Resource Constraints:**
  - Real-time performance is limited by hardware (CPU/GPU) and model size.

## Next Steps and Recommendations

- **Incomplete/To-Do:**
  - Evaluate tracking performance quantitatively (e.g., ID switches, MOTA/MOTP).
  - Test on longer or multi-camera videos for cross-camera re-identification.
  - Experiment with more advanced appearance feature extractors for improved re-ID.
  - Integrate a UI or output video writer for result visualization and review.

- **With More Time/Resources:**
  - Fine-tune the detection and tracking models for the specific dataset.
  - Implement batch processing and asynchronous pipelines for further speedup.
  - Explore transformer-based trackers or re-ID networks for higher accuracy.

---

*Prepared by: [D Tarun Sai]*
*Date: [26-06-2025]* 