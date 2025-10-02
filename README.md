# pneumonia-gradcam
Chest X-ray classification using ResNet18 with Grad-CAM visualizations.
# Pneumonia Detection with CNN and Grad-CAM Visualizations

## Overview
This project uses a convolutional neural network (CNN) to classify chest X-ray images as **Pneumonia** or **Normal**. Additionally, it applies **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize which regions of the X-ray the model focuses on for its predictions.

The model demonstrates **transfer learning** using a pre-trained ResNet18 and fine-tuning on a custom dataset. Grad-CAM heatmaps provide interpretability, showing what the network “sees” when making decisions.

---

## Features
- **Transfer Learning:** Uses pre-trained ResNet18 from PyTorch and fine-tunes on your dataset.
- **Grad-CAM Visualizations:** Highlights image regions influencing the model’s decision.
- **Test on Random Subsets:** Demonstrates generalization by predicting on new, unseen images.
- **Overlay of Heatmaps:** Original image combined with Grad-CAM to interpret the model’s focus areas.

---

## Project Structure
pneumonia-gradcam/
│
├── pneumonia_gradcam.ipynb # Main Jupyter notebook with training, testing, and Grad-CAM
├── dataset from kaggle, X-ray Pnemonia, https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
├── README.md # This file

How to Use

The notebook automatically loads your dataset and splits it into train, validation, and test sets.

It fine-tunes a ResNet18 model for pneumonia classification.

Grad-CAM visualizations are generated for random test images:

Side by Side heatmap with the original image are displayed

The heatmap uses a color scale: red = highly important, blue = less important.



Results

Test accuracy on a new random subset: ~87%

Grad-CAM visualizations highlight regions of the lungs relevant to pneumonia detection.

Overlayed images provide interpretability for the model’s predictions.


Notes

The notebook is designed to run on CPU, but can be accelerated with GPU if available.



You can increase the number of test images or select different subsets to analyze more examples.

Acknowledgements

PyTorch
 for the deep learning framework.

ResNet18 model architecture from torchvision.

Grad-CAM methodology from Selvaraju et al., 2017.
