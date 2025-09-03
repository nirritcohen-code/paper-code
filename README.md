# paper-code
Supporting Information Code for Nirrit Cohen et al. (2025)

This archive contains the complete Python scripts and environment specifications used in the analysis of microplastic fiber dynamics and image processing for this study. The code enables full reproducibility of our methods for thresholding, binary image preparation, and fiber tracking.

Contents:

main.py: Orchestrates the entire processing workflow, including thresholding and tracking.

choose_binary_threshold.py: Script for selecting optimal binary thresholds from ND2 video frames.

save_bin_frames.py: Saves binary frames from the processed ND2 video using the chosen thresholds.

object_tracking.py: Tracks fibers in the binary image sequences, computes their dynamics, and exports relevant data.

clean_iso_pixels.py: Utility script used to remove isolated pixels from binary images to improve contour detection accuracy.

requirements.txt: Lists all required Python packages with versions for environment setup.

Usage:

Create a Python virtual environment (recommended) and activate it.

Install the required packages using:

nginx
Copy
Edit
pip install -r requirements.txt

Run the scripts in order of workflow:

First, determine thresholds with choose_binary_threshold.py (optional but recommended).

Next, use save_bin_frames.py to process and save binary images.

Finally, run object_tracking.py to extract fiber trajectories and features.

Alternatively, use main.py as an example script to orchestrate the workflow.

Notes:

All scripts were manually developed for this research and were not generated using artificial intelligence tools.

The scripts were tested on Python 3.12.0.

Example usage instructions can be found as comments within each script.

Please refer to the manuscript’s Methods section and Supporting Information for additional context on data processing.

