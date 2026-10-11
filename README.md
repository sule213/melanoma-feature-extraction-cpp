# Melanoma Feature Extraction (C++)

Handcrafted feature extraction from skin lesion (melanoma) images, written in C++ with OpenCV. This is the feature extraction front end for the classification project in [melanoma-skin-lesion-classification](https://github.com/sule213/melanoma-skin-lesion-classification): it turns a lesion image into a numeric feature vector that the machine learning notebooks then classify.

## What it does

For each input image, the code:

1. Detects keypoints with the Features from Accelerated Segment Test (FAST) corner detector
2. Segments the lesion with GrabCut
3. Applies mean shift filtering and k-means colour quantisation
4. Measures the lesion contour and shape: area, perimeter, circularity, aspect ratio, roundness, compactness, extent, and fitted ellipse values
5. Computes colour and texture statistics: entropy, skewness, kurtosis, variance, luminance, and average intensity

The output feature set combines shape descriptors with colour and texture statistics for benign versus malignant classification.

## Repository contents

- `melanoma_feature_extraction.cpp` - the single source file (666 lines of C++).

## Building and running

You need a C++ compiler and OpenCV installed. Build with, for example:

```bash
g++ -std=c++17 melanoma_feature_extraction.cpp -o extract_features $(pkg-config --cflags --libs opencv4)
./extract_features
```

Input image paths are set in the code (the file includes example benign and malignant image paths). Edit them to point at your own lesion images before running.

## Notes

- No images are included in this repository.
- Companion classification notebooks: [melanoma-skin-lesion-classification](https://github.com/sule213/melanoma-skin-lesion-classification)
