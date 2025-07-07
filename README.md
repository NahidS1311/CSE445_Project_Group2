# U-Net Image Segmentation for Semantic Labeling

This repository implements an image segmentation model using the **U-Net architecture**. U-Net is a powerful deep learning model designed for pixel-wise image segmentation, widely used for tasks such as medical image segmentation, object detection, and image restoration.

## Project Overview

This project applies the U-Net model for binary image segmentation, where the goal is to classify each pixel in an image as either part of a specific region of interest (foreground) or background. The model is trained using a dataset of images and corresponding segmentation masks, where each mask identifies the object or region in the image that needs to be segmented.

The U-Net model is built using TensorFlow/Keras and is designed to work on images of size 256x256 pixels. This implementation includes data preprocessing, model architecture, training, evaluation, and visualization.

## Dataset

The dataset consists of images and corresponding binary masks for segmentation tasks. The images are in PNG format, and the masks represent the segmentation areas as binary images (0 for background, 1 for the foreground).

The dataset is expected to be in a specific directory structure:

```
/content/working/
  └── <image_id>/
      ├── images/
      │    └── <image_id>.png
      └── masks/
           ├── mask1.png
           ├── mask2.png
           └── ...
```

Where `<image_id>` is a unique identifier for each sample, and the masks represent the segmented regions in the corresponding image.

## Requirements

To run this project, you need the following Python libraries:

- `tensorflow` (for model building and training)
- `numpy` (for numerical operations)
- `opencv-python` (for image loading and processing)
- `matplotlib` (for visualization)
- `scikit-learn` (for data splitting)
- `glob` (for file management)
- `os` (for file handling)

You can install the required libraries using pip:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

## File Structure

- `main.py`: The main script containing the entire pipeline for data loading, model building, training, and evaluation.
- `README.md`: This file.
- `/content/working/`: Directory where the dataset is expected to be located (containing images and masks).
- `/unet_model.keras`: The saved model after training.

## How to Use

### 1. Dataset Preparation

Place your dataset in the `/content/working/` directory with the structure described above. Ensure that each image has a corresponding mask.

### 2. Preprocessing

Run the following script to preprocess the data, load the images, and prepare the training and validation sets:

```python
!unzip -q /content/stage1_train.zip -d /content/working/
```

This will extract the dataset into the working directory.

### 3. Train the Model

To train the model, simply run the `main.py` script. The training process will begin, and the model will be trained on the provided images and masks. The model will be evaluated on a validation set, and training progress will be visualized.

```bash
python main.py
```

### 4. Model Architecture

The U-Net model consists of:
- **Encoder**: A series of convolutional blocks with max pooling to downsample the image and capture features.
- **Bottleneck**: The deepest layer of the network that captures high-level features.
- **Decoder**: A series of transpose convolutional blocks with skip connections that upsample the image to its original size.

The model is compiled with the **Adam optimizer** and **binary cross-entropy loss**. The **Dice coefficient** is used as a metric to evaluate the segmentation performance.

### 5. Evaluation and Visualization

After training, the model's performance is evaluated using the Dice coefficient metric, which measures the overlap between predicted and ground truth segmentation masks. A plot of the training and validation Dice coefficient over epochs is generated.

Additionally, some sample predictions are visualized alongside the ground truth masks and the predicted segmentation masks for qualitative evaluation.

### 6. Saving the Model

Once the model is trained, it is saved as a `.keras` file. This allows you to load and use the model for inference on new images:

```python
model.save('/content/unet_model.keras')
```

## Results

After training, the model should produce reasonably good segmentation masks, depending on the quality of the dataset and the training process. You can visualize the results with a side-by-side comparison of the original images, ground truth masks, and predicted masks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This code is based on the original U-Net architecture for image segmentation.
- The dataset used in this project is publicly available for academic research.

## References

- **U-Net**: [Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.](https://arxiv.org/abs/1505.04597)
