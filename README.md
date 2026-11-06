# Football Vision — Player Tracking & Spatial Analysis

> **Project status:** Version 1 prototype / work in progress  
> **Last updated:** 6 November 2026

Football Vision is a learning project exploring how computer vision can be used to analyse football footage. It uses Python, OpenCV, NumPy and Ultralytics YOLO to detect and track players, estimate their image positions, visualise movement and experiment with ball-related analysis.

The current version is a prototype developed and tested on sample football footage. Its outputs are experimental and should not be treated as professional or validated performance statistics.

## Progress so far

### Player detection and tracking

- Loaded a pretrained Ultralytics YOLO model (`yolo11n.pt`).
- Detected people in football images and video frames.
- Processed videos frame by frame using streaming inference.
- Used the Ultralytics tracking interface to assign track IDs to detected players.
- Extracted bounding-box coordinates and estimated each player's ground position using the bottom-centre of the bounding box.

### Movement analysis

- Stored player positions over time in a dictionary, organised by tracking ID.
- Connected consecutive positions to create player movement trajectories.
- Created a top-down image by selecting four source points and applying an OpenCV perspective transformation.
- Transformed player positions into the top-down image coordinate system.
- Generated an experimental heatmap from the accumulated transformed player detections.

### Ball analysis

- Tested sports-ball detection using the COCO `sports ball` class in the pretrained YOLO model.
- Experimented with ball tracking and storing detected ball-centre coordinates across frames.
- Built a basic possession-estimation prototype that identifies the tracked player nearest to a detected ball, subject to a provisional image-distance threshold.

### Earlier experimental metrics

Pixel-based distance, speed and sprint-threshold calculations were also explored using player image coordinates. These are preliminary experiments only. They have not been calibrated to real pitch dimensions or validated as real-world distance, speed or sprint measurements.

## Technology stack

- **Python** — application code
- **Ultralytics YOLO** — object detection and tracking
- **OpenCV** — video/image processing, drawing and perspective transformation
- **NumPy** — coordinate arrays and heatmap calculations

## Current limitations

- A general-purpose pretrained YOLO model may miss a small, blurred, occluded or fast-moving football. Ball detection needs further testing and may require a football-specific detector or custom training data.
- Player tracking IDs can be lost, changed or assigned incorrectly, especially during occlusion or when players overlap.
- The current perspective transformation depends on manually selected points. The selected points must correspond to suitable points on the same flat pitch plane; inaccurate calibration distorts mapped positions.
- The heatmap counts detected positions. It is an exploratory visualisation, not yet a validated measure of time spent in each pitch area.
- Possession is estimated from proximity between a detected ball and a player's estimated foot position. Proximity alone does not prove control or possession.
- Distance and speed values are currently pixel-based. Reliable metres, metres per second or kilometres per hour require suitable pitch calibration and validation.
- Pass detection, team identification and reliable match-event detection have not yet been implemented.

## Suggested project layout

The filenames in the local development folder may differ. As the repository is organised, a structure like this is recommended:

```text
football-vision-player-tracking/
├── README.md
├── requirements.txt
├── .gitignore
├── player_tracking.py
├── perspective_calibration.py
├── topdown_trajectory.py
├── ball_detection.py
├── ball_tracking.py
├── possession_estimation.py
├── images/
│   └── example_outputs/       # screenshots or permitted sample outputs
└── VIDEOS/                    # local input footage; do not commit by default
```

Keep large video files, downloaded model weights, virtual environments and generated temporary files out of the Git repository unless there is a clear reason and you have permission to share them.

## Setup

The project was developed locally with Python 3.12 on Windows. To set up a Python environment:

1. Clone the repository or download the source code.
2. Open a terminal in the project directory.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

A basic `requirements.txt` can contain:

```text
ultralytics
opencv-python
numpy
```

The YOLO model weights (`yolo11n.pt`) may be downloaded by Ultralytics when the model is first loaded. Keep the weights out of Git unless you have a specific reason to distribute them.

## Running the project

The scripts are currently being developed and tested separately. Run the relevant script from the project root, for example:

```bash
python player_tracking.py
python perspective_calibration.py
python topdown_trajectory.py
python ball_detection.py
python ball_tracking.py
python possession_estimation.py
```

These are suggested script names for an organised repository. Adjust the commands to match the actual filenames you commit. Make sure the video path used in each script points to a video available on your own machine.

## Data and responsible use

Use football footage that you own or are permitted to process and share. Do not upload copyrighted match footage, private recordings, or large source videos to a public repository without the necessary rights and consent. For a public portfolio, prefer screenshots or short sample outputs that you are permitted to publish.

## Planned next steps

1. Organise the scripts and document how to reproduce each result.
2. Test player detection and tracking on several suitable videos and inspect ID consistency.
3. Improve pitch calibration and validate the top-down player mapping.
4. Improve ball detection and assess how often the ball is detected correctly.
5. Improve possession estimation using temporal evidence, not only nearest-player distance.
6. Explore team identification and pass detection.
7. Calibrate the pitch before reporting distance or speed in real-world units.
8. Build a simple report or dashboard to present validated outputs.

## Project approach

This project is being built incrementally as a practical way to learn Python and computer vision through football analytics. Features are added as prototypes first, then tested and improved. Accuracy, reproducibility and clear documentation are priorities for future versions.

---

**Football Vision — an ongoing computer-vision learning project.**

