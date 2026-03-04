# 🩺 Automated Chest X-ray Segmentation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)](https://tensorflow.org)
[![Model](https://img.shields.io/badge/Architecture-U--Net%20%2B%20ResNet50-green)](https://arxiv.org/abs/1505.04597)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

---

## PneumoScan-AI

Automated Lung Segmentation and Abnormality Detection in Chest X-rays using U-Net with ResNet50 Backbone

![AI](learn&experiments/vscode-readme.png)

## 🛠️ Recommended Conda Environment

    git clone https://github.com/Ahmed2797/PneumoScan-AI

    conda create -n brain python=3.12
    conda activate brain

    # Install pip packages from requirements.txt
    pip install -r requirements.txt

    ## 📂 Download Dataset
        ## smaill size
        https://drive.google.com/file/d/1bo0OC0oT2o8lx7d5fBmVMEyOtBMMCBp2/view?usp=sharing
        https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

    ## Train The model-pipeline
    python main.py

    ## web app
    python app.py

## 🚀 Technical Highlights

* **Architecture:** Hybrid U-Net with a pre-trained ResNet50 Encoder.
* **Data Pipeline:** Custom `tf.data` API and `Keras Sequence` generator for memory-efficient training.
* **Augmentation:** Real-time synchronized augmentation (Flip, Rotation, Zoom) for both images and masks.
* **Optimization:** Used **Dice Loss** and **BCE-Dice Hybrid Loss** to overcome class imbalance in medical imagery.
* **Inference:** Post-processing module with **OpenCV** to generate Bounding Boxes and Confidence Levels.

## Project: Automated Chest X-ray Pathology Segmentation

Architecture & Design: Developed a high-precision segmentation model using U-Net architecture integrated with a Pre-trained ResNet50 Backbone (Transfer Learning) to identify pulmonary abnormalities.

Data Engineering: Engineered a memory-efficient data pipeline using TensorFlow tf.data API and custom Keras Sequences, enabling seamless training on large-scale medical datasets.

Model Optimization: Implemented Hybrid BCE-Dice Loss to mitigate extreme class imbalance, achieving a Dice Coefficient of 0.88 and an IoU of 0.82.

Advanced Augmentation: Designed a synchronized data augmentation suite (Rotation, Zoom, Flip) to enhance model generalization and robustness against clinical imaging variability.

Clinical Interpretability: Integrated an automated post-processing module using OpenCV to extract Bounding Boxes and generate Confidence Scores, providing actionable insights for radiologists.

Deployment: Successfully deployed the model as a real-time web interface using Streamlit, allowing users to upload X-rays and receive instant diagnostic overlays.
