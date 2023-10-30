# Feature Tracking System

This project implements a lane keeping and feature tracking system using computer vision techniques. The system uses SIFT (Scale-Invariant Feature Transform) for detecting and computing keypoints and descriptors in grayscale images. It then uses BFMatcher (Brute-Force Matcher) to perform feature matching. The system also calculates optical flow on the matched points and filters out points that were not successfully tracked.

## Dependencies

The project requires the following Python libraries:
- numpy
- OpenCV

## How to Run

1. Clone this repository.
2. Navigate to the directory containing the script.
3. Run the script using the command: `python main.py`

## Code Structure

The code is divided into two main functions:

- `track_feature(curr_img, prev_img)`: This function takes in the current and previous images, converts them to grayscale, detects and computes keypoints and descriptors using SIFT, performs feature matching using BFMatcher, applies a ratio test to get good matches, calculates optical flow on the matched points, filters out points that were not successfully tracked, and draws the tracks on the image.

- `main()`: This function reads in images from a specified path, calls the `track_feature` function on each pair of consecutive images, and displays the result.

Please replace `'C:/00/image_0/{0:06d}.png'` with the path to your own image sequence.

## Note

This code is intended for educational purposes and should be adapted for real-world applications.
