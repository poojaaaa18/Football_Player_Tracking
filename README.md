
# Soccer Player Re-Identification and Tracking

This project implements a soccer player tracking system with consistent ID assignment across frames, including re-identification when players leave and reappear. It uses YOLOv5 for detection, OSNet for re-identification (ReID), HSV-based team classification, and Tesseract OCR for jersey number recognition.

---

## 🚀 How to Set Up and Run

### Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU)
- OpenCV
- ultralytics (YOLOv5)
- torchreid
- pytesseract
- scikit-learn

Install requirements:

```bash
pip install -r requirements.txt
```

📦 Model Weights

Download `best.pt` from [[Google Drive Link Here](https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP?usp=sharing)] and place it inside the `static/assets/` directory.


Make sure Tesseract OCR is installed and its path is correctly set in the script:


Download link: https://github.com/tesseract-ocr/tesseract/wiki

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Folder Structure

```
project/
│
├── static/
│   ├── assets/
│   │   └── best.pt                 # Trained YOLOv5 model
│   └── reid_model/
│       └── osnet_x1_0_imagenet.pth  # Pretrained OSNet model
├── main.py                         # Main tracking code
├── README.md
└── requirements.txt
```

### Run the Tracker

Put the input video under `static/assets/` and update the file path in `main.py`:

```python
input_video = "static/assets/15sec_input_720p.mp4"
```

Then run:

python main.py


The output will be saved as `output_with_consistent_ids.mp4`.

---

## 🔍 Dependencies and Environment

Ensure the following are installed:

- ultralytics==8.x
- torchreid
- opencv-python
- pytesseract
- numpy
- scikit-learn

---

## 📌 Notes

- Uses ReID + spatial heuristics + number & color matching to assign persistent IDs.
- Handles temporary occlusions, out-of-frame exits, and returns.
- Possession tracking and team stats can be extended further.

---

## Final Workflow Summary:
-Detect players using YOLOv5.

-Track and re-identify players over time using OSNet.

-Classify teams using jersey colors in HSV.

-Read jersey numbers with Tesseract OCR.

-Result: Each player is uniquely identified and tracked by number and team.
## Author
Pooja Kushwaha :https://github.com/poojaaaa18/Football_Player_Tracking
