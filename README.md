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

## Food Classes

The model uses **9 semantic classes**, including background:

| ID | Class           |
| -: | --------------- |
|  0 | Background      |
|  1 | Chutney         |
|  2 | Ghundruk        |
|  3 | Lentil (Dal)    |
|  4 | Meat Curry      |
|  5 | Rice (Bhat)     |
|  6 | Salad           |
|  7 | Spinach (Saag)  |
|  8 | Vegetable Curry |

---

## Application Workflow

```text
             Input Food Image
                    │
                    ▼
             Image Preprocessing
                    │
                    ▼
             SegFormer MiT-B2
                    │
                    ▼
          Pixel-Level Segmentation
                    │
             ┌──────┴──────┐
             ▼             ▼
       Food Classes    Segmentation Mask
             │             │
             └──────┬──────┘
                    ▼
          Relative Area Percentage
                    │
                    ▼
             Visual Results
```

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

It is recommended to use a Python virtual environment.

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
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
