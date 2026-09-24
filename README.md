# Nepali Food Semantic Segmentation

A deep learning-based web application for **semantic segmentation of Nepali food images** using the **SegFormer MiT-B2** architecture.

The application identifies and segments different food items at the pixel level, providing a visual representation of the detected food regions and their relative pixel-area percentages.

## Overview

Understanding the composition of a food image is an important step toward automated dietary monitoring. This project applies semantic segmentation to Nepali food images to identify individual food regions within a plate or meal image.

The deployed application allows users to:

* Upload a Nepali food image
* Perform semantic segmentation using a trained SegFormer model
* Visualize the segmentation result
* Identify detected food categories
* Calculate the relative pixel-area percentage of each detected food class
* View the original image alongside the segmentation output

> **Note:** The area percentage reported by this application represents the proportion of image pixels assigned to each food class. It is not a direct measurement of food mass, physical volume, serving size, or calorie content.
## Dataset Development and Model Evaluation

A dedicated Nepali food image dataset was developed for the semantic segmentation task. The dataset development process included **food image collection, data refinement, pixel-level annotation, quality checking, dataset splitting, and model evaluation**.

### Dataset Preparation

The dataset was prepared through the following stages:

1. **Image Collection**
   Nepali food images were collected to represent common food items found in Nepali meals.

2. **Data Refinement**
   The collected images were reviewed and refined to remove unsuitable, duplicate, or unusable samples and to improve the consistency of the dataset.

3. **Pixel-Level Annotation**
   Food regions in the images were manually annotated using **CVAT (Computer Vision Annotation Tool)**. Each food item was assigned to its corresponding semantic class at the pixel level.

4. **Annotation Quality Checking**
   The generated image-mask pairs were checked for annotation consistency, image-mask correspondence, valid dimensions, expected mask labels/colors, empty masks, and other potential data-quality issues.

5. **Dataset Splitting**
   The finalized dataset was divided into training, validation, and test sets. The split was performed while considering the presence of different food classes to maintain class representation across the subsets.

### Dataset Classes

The segmentation dataset contains **9 semantic classes**, consisting of 8 food categories and a background class:

| Class           | Category   |
| --------------- | ---------- |
| Background      | Background |
| Chutney         | Food       |
| Ghundruk        | Food       |
| Lentil (Dal)    | Food       |
| Meat Curry      | Food       |
| Rice (Bhat)     | Food       |
| Salad           | Food       |
| Spinach (Saag)  | Food       |
| Vegetable Curry | Food       |

### Exploratory Data Analysis and Quality Assessment

Before model training, the dataset was examined using several analyses, including:

* Class presence distribution
* Pixel-level class distribution
* Class co-occurrence
* Relative food-region area
* Image dimensions and aspect ratios
* Image-mask correspondence
* Mask label/color validation
* Empty-mask and invalid-sample checking

These analyses were used to understand the characteristics of the dataset and identify potential class-imbalance and annotation issues before model training.

### Evaluation of Multiple Segmentation Models

To investigate the effectiveness of different deep learning architectures for Nepali food semantic segmentation, the dataset was used to train and evaluate **five segmentation models**.

The evaluated architectures included:

| Model       | Backbone / Architecture |
| ----------- | ----------------------- |
| FCN         | ResNet-50               |
| U-Net       | ResNet-50               |
| DeepLabV3+  | ResNet-50               |
| SegFormer   | MiT-B2                  |
| Mask2Former | Swin-Tiny               |

The models were evaluated using segmentation-specific metrics, including:

* **Mean Intersection over Union (mIoU)**
* **Mean Dice coefficient**
* **Pixel Accuracy**
* **Per-class IoU**
* **Per-class Dice score**
* **Test loss**

The evaluation was performed using the same overall dataset framework to provide a consistent basis for comparing segmentation performance across the different architectures.

### Research Workflow

```text
Nepali Food Image Collection
            │
            ▼
      Data Refinement
            │
            ▼
     CVAT Pixel Annotation
            │
            ▼
   Annotation Quality Checking
            │
            ▼
     Exploratory Data Analysis
            │
            ▼
       Train / Validation / Test
            │
            ▼
     ┌──────┼────────┬──────────┐
     ▼      ▼        ▼          ▼
    FCN    U-Net  DeepLabV3+ SegFormer
     │      │        │          │
     └──────┴────────┴──────────┘
                    │
                    ▼
              Mask2Former
                    │
                    ▼
          Comparative Evaluation
                    │
                    ▼
          Final Segmentation Model
```

The dataset and model evaluation process provides the experimental foundation for the deployed SegFormer-based application.

---

## Model

The application uses:

**SegFormer MiT-B2**

SegFormer is a transformer-based semantic segmentation architecture designed for efficient pixel-level image segmentation.

The trained model is loaded from:

```text
best_model.pth
```

---

 
---
 

## Project Structure

The repository is organized as follows:

```text
nepali-food-segmentation/
│
├── app.py
├── best_model.pth
├── requirements.txt
└── README.md
```

### File Description

| File               | Description                               |
| ------------------ | ----------------------------------------- |
| `app.py`           | Streamlit web application                 |
| `best_model.pth`   | Trained SegFormer MiT-B2 model checkpoint |
| `requirements.txt` | Python dependencies                       |
| `README.md`        | Project documentation                     |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/nepali-food-segmentation.git
```

Move into the project directory:

```bash
cd nepali-food-segmentation
```

### 2. Install Dependencies

 
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Application Locally

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## Example Usage

1. Open the Streamlit application.
2. Upload a food image.
3. The image is processed by the trained SegFormer model.
4. The model generates a pixel-level segmentation mask.
5. Detected food regions are visualized.
6. The application displays the relative pixel-area percentage of the detected classes.

---

## Area Percentage

For each detected food class, the application calculates its relative pixel area using:

```text
Area Percentage =
(Number of pixels belonging to the food class /
 Number of pixels belonging to all detected food classes)
× 100
```

The calculation is based on the predicted segmentation mask.

Therefore, the resulting percentage should be interpreted as **image-based relative area**, rather than a physical portion measurement.

---

## Technologies Used

* **Python**
* **PyTorch**
* **Hugging Face Transformers**
* **Streamlit**
* **NumPy**
* **Pillow**
* **Semantic Segmentation**
* **SegFormer MiT-B2**

---

## Research Scope

This application is developed as part of research on:

**Deep Learning-Based Semantic Segmentation of Nepali Food Images for Dietary Monitoring**

The primary focus of the work is **pixel-level semantic segmentation of Nepali food images**.

The segmentation output can potentially serve as a foundation for future dietary analysis applications, such as more advanced portion estimation or nutritional assessment when additional information such as physical scale, depth, volume, or appropriate food-nutrition models is available.

---

## Limitations

The current system has several limitations:

* Segmentation performance depends on the quality and characteristics of the input image.
* Visually similar food categories may be difficult to distinguish.
* Very small or partially occluded food regions may be challenging to segment.
* Pixel-area percentage does not represent physical food mass or volume.
* The application does not directly determine calorie content from segmentation alone.
* Real-world portion estimation would require additional information and modeling beyond semantic segmentation.

---

## Future Work

Potential future extensions include:

* Improved segmentation of rare and visually similar food classes
* Larger and more diverse Nepali food datasets
* Advanced image-based portion estimation
* Depth- or scale-based volume estimation
* Integration with nutritional databases
* Food-specific nutritional estimation
* Mobile and edge-device deployment
* Further explainable AI analysis of segmentation predictions

---

## Acknowledgement

This project was developed as part of academic research on deep learning-based semantic segmentation of Nepali food images for dietary monitoring.

---

## License

This repository is intended primarily for academic and research purposes.

Please contact the project author before using the trained model or dataset for commercial purposes.
