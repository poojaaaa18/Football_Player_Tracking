# Soccer Player Tracking and Re-identification System – Project Report

## Introduction

Tracking individual players in a football match is a complex problem due to frequent occlusions, dynamic camera movements, and player similarity. Traditional tracking methods often fail to maintain consistent identities when players leave and re-enter the frame or become occluded. The goal of this project was to develop a robust and accurate player tracking system that assigns consistent and unique IDs to each player throughout a video, even in challenging scenarios.

## Problem with Existing Methods

We initially experimented with standard tracking algorithms like DeepSORT and ByteTrack. While these methods provided basic tracking, they struggled with re-identifying players after temporary disappearance. They rely heavily on bounding box overlap (IoU) and simple appearance features, which are insufficient for the dynamic nature of football games where players frequently change positions, cluster together, or temporarily exit the field of view.

Key limitations observed:
- Players received different IDs after leaving and re-entering the frame.
- No integration of semantic features such as jersey color or number.
- Frequent ID switches when players had similar builds or wore similar uniforms.
- Poor performance in scenes with heavy occlusion or motion blur.

## Proposed Solution

To overcome these issues, we developed a custom tracking pipeline that integrates multiple cues: appearance, spatial location, jersey number, and team color. The system is built using the following components:

- 🧠 YOLOv5: Used for high-performance player detection. It efficiently identifies players and the ball in each frame.
- 🔁 OSNet ReID Model: Extracts rich appearance embeddings for each detected player crop, helping to distinguish players even when clothing is similar.
- 🎨 HSV-based Team Classification: Uses color filtering to determine team affiliation (e.g., red vs. blue) based on dominant jersey color.
- 🔢 Jersey Number Recognition: Leverages Tesseract OCR with adaptive thresholding to read jersey numbers and boost re-identification accuracy.
- 📍 Position-aware Tracking: Calculates Euclidean distance and velocity predictions to improve matching accuracy for returning or occluded players.
- ✅ Matching Logic: A weighted scoring function combining appearance similarity, position proximity, jersey number similarity, and team classification for optimal ID assignment.

## Technical Highlights

- Appearance embeddings are extracted using OSNet with ImageNet-pretrained weights.
- Jersey number OCR is accelerated via Gaussian blurring and adaptive thresholding.
- Team colors are identified by analyzing HSV hue histograms in the upper torso region.
- Tracks are maintained even when players disappear for short durations using position prediction and recent embeddings.

## Challenges Faced

1. OCR Reliability:
   - Jersey numbers are often partially visible, blurred, or occluded.
   - Tesseract performance varies based on lighting and font size.

2. Similar Appearance:
   - Players with similar builds and jersey styles caused identity confusion.
   - Appearance-based similarity needed to be combined with contextual cues.

3. Occlusion and Re-entry:
   - Maintaining consistent IDs for players leaving and re-entering the frame was difficult.
   - Required the use of a lost-player buffer and re-matching logic using cosine similarity and positional data.

4. Color Misclassification:
   - HSV-based color classification occasionally failed under varying lighting conditions.
   - Tuned thresholds and heuristics were introduced for better accuracy.

5. Performance Optimization:
   - Frame skipping, resizing, and half-precision inference were used to speed up processing.
   - Batch processing of ReID features improved efficiency.

## Results

- The new system significantly outperforms DeepSORT in terms of ID consistency and re-identification accuracy.
- IDs remain consistent across frame gaps, occlusions, and camera shifts.
- Visual overlay on the video confirms correct team classification and jersey number reading in most cases.

## Future Improvements

- Train a custom OCR model tailored for jersey fonts to replace Tesseract.
- Introduce pose-based ReID features to further disambiguate players.
- Integrate ball tracking and possession estimation based on nearest-player analysis.
- Add player action recognition and heatmap generation for tactical analysis.
- Implement lightweight versions for real-time processing on edge devices.

## Conclusion

This project demonstrates a significant advancement in sports analytics by combining multiple cues—detection, appearance, color, number, and position—to build a highly accurate football player tracking system. While challenges remain, particularly around OCR and similar appearances, the approach provides a solid foundation for further development into real-time analytics and automated match analysis.
