# Player Re-Identification in a Single Feed

This project implements player re-identification in a single video feed using YOLOv11 for detection and DeepSORT for tracking. The system assigns consistent IDs to players, even if they leave and re-enter the frame.

## Setup Instructions

### 1. Clone or Download the Repository
Place all files (`player_reid.py`, `best.pt`, `15sec_input_720p.mp4`, etc.) in the same directory.

### 2. Install Dependencies
Install the required Python packages using pip:
pip install ultralytics opencv-python torch deep_sort_realtime numpy


### 3. Prepare Model and Video
- Place your YOLOv11 weights file (`best.pt`) in the project directory.
- Place your input video (`15sec_input_720p.mp4`) in the project directory.

### 4. Run the Code
python player_reid.py


- The script will be displaying the video with bounding boxes and consistent player IDs.
- Press `q` to quit the video window.

## Environment Requirements
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for speed)
- OS: Windows, Linux, or macOS

## Main Dependencies
- [ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8/YOLOv11)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [torch](https://pytorch.org/)
- [deep_sort_realtime](https://pypi.org/project/deep-sort-realtime/)
- [numpy]

## Notes
- For best results, ensure your `best.pt` model is trained for player detection.
- You can adjust frame skipping, resizing, and display options in `player_reid.py` for speed/accuracy trade-offs.

## Troubleshooting
- If you encounter errors about missing modules, double-check your Python environment and installed packages.
- For DeepSORT or YOLO errors, ensure the input formats match the expected types (see code comments).

---

*Prepared by: [D Tarun Sai]*
*Date: [26-06-2025]* 