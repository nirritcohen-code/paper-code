# Supporting Information Code for Cohen et al. (2025)

This repository contains the complete Python scripts and environment specifications used in the analysis of **microplastic fiber dynamics** and **image processing** for the study.  
The code enables full reproducibility of the methods for thresholding, binary image preparation, and fiber tracking.

---

## 📂 Contents

- **`main.py`** – Orchestrates the entire processing workflow, including thresholding and tracking.  
- **`choose_binary_threshold.py`** – Selects optimal binary thresholds from ND2 video frames.  
- **`save_bin_frames.py`** – Saves binary frames from the processed ND2 video using the chosen thresholds.  
- **`object_tracking.py`** – Tracks fibers in the binary image sequences, computes their dynamics, and exports relevant data.  
- **`clean_iso_pixels.py`** – Removes isolated pixels from binary images to improve contour detection accuracy.  
- **`requirements.txt`** – Lists all required Python packages with versions for environment setup.  

---

## ⚙️ Usage

1. **Set up the environment** (recommended: Python 3.12.0)  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   pip install -r requirements.txt
2. **Run the workflow scripts**
- (Optional) Determine thresholds:

python choose_binary_threshold.py
   - (Optional) Determine thresholds: `python choose_binary_threshold.py`  
   - Process and save binary images: `python save_bin_frames.py`  
   - Extract fiber trajectories and features: `python object_tracking.py`
Alternatively, you can run the entire workflow with:

---

## 📝 Notes

- All scripts were manually developed for this research and **not generated using AI tools**.  
- Example usage instructions are included as comments within each script.  
- For additional methodological context, please refer to the **Methods** section and **Supporting Information** in the manuscript.  

---

✦ *Corresponding author: Nirrit Cohen*  
