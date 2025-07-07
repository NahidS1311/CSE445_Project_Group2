# Nuclei Segmentation Using U-Net Architecture

## Project Overview

This project focuses on the development of a **nuclei segmentation model** using the **U-Net architecture** to automatically segment and identify the nuclei in **whole slide images (WSIs)**. The dataset used for training and evaluation is sourced from the **2018 Data Science Bowl: Nuclei Segmentation Challenge**. The purpose of this project is to build a deep learning model capable of accurately identifying nuclei in histopathological tissue samples, a key task in digital pathology for diagnosing diseases like cancer.

The project involves the following steps:
1. **Dataset Loading**: Loading and preprocessing the data (images and masks).
2. **Model Architecture**: Implementing and training a **U-Net model** for image segmentation.
3. **Model Evaluation**: Evaluating model performance based on **accuracy** and **Dice coefficient**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Technologies Used

This project utilizes several key technologies:

- **Python 3.x**: The programming language used for implementing the deep learning model.
- **TensorFlow 2.x**: A powerful deep learning framework used to build and train the U-Net model.
- **Keras**: A high-level neural networks API integrated with TensorFlow to build and manage models.
- **OpenCV**: A library used for reading and processing images.
- **NumPy**: Essential for numerical operations and handling multi-dimensional arrays.
- **Matplotlib**: Used for plotting graphs, visualizing training curves, and displaying results.
- **Scikit-learn**: Used for splitting the dataset into training and validation sets.
- **Google Colab**: The environment used for training the model on cloud-based GPUs.

---

## Usage

### 1. Load the Data

The dataset consists of images and corresponding masks. The **`data_loader`** function handles loading and preprocessing, resizing the images and masks to a uniform size (256x256 pixels).

To load and visualize a sample image with its corresponding mask:

```python
from data_loader import load_sample

# Load a sample image and its corresponding mask
img, mask = load_sample('image_id')

# Display the image and overlay the mask
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.imshow(mask.squeeze(), cmap='Reds', alpha=0.4)
plt.title("Overlayed Mask")
plt.axis('off')
plt.show()
```

### 2. Train the Model

To train the **U-Net** model, first load the training and validation data. Then, compile the model and train it:

```python
from unet_model import build_unet
from data_loader import data_loader

# Load data
X_train, Y_train = data_loader('/path/to/data')

# Build the U-Net model
model = build_unet(input_shape=(256, 256, 3))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'dice_coef'])

# Train the model
history = model.fit(X_train, Y_train, epochs=30, batch_size=16, validation_split=0.2)
```

### 3. Evaluate the Model

After training, evaluate the model's performance on the validation dataset. The model uses **Dice coefficient** as the primary evaluation metric:

```python
# Evaluate the model on the validation dataset
val_loss, val_accuracy, val_dice_coef = model.evaluate(X_val, Y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Dice Coefficient: {val_dice_coef}')
```

---

## Training the Model

The **U-Net architecture** is a powerful deep learning model designed specifically for **image segmentation tasks**. It consists of an **encoder-decoder** structure with **skip connections** to preserve spatial information. The architecture helps the model learn both local (fine-grained) and global (contextual) features in images, making it ideal for tasks like **nuclei segmentation**.

The model is trained using:
- **Binary Cross-Entropy Loss**: Suitable for binary segmentation tasks (nuclei vs background).
- **Accuracy**: Measures the percentage of correctly classified pixels.
- **Dice Coefficient**: A metric commonly used for evaluating segmentation performance, measuring the overlap between predicted and true masks.

---

## Evaluation Metrics

The model’s performance is measured using the following metrics:

- **Accuracy**: The fraction of pixels correctly classified as either background or nuclei.
- **Dice Coefficient**: The most important metric for segmentation, measuring the overlap between the predicted and ground truth masks.
  - Formula:
    \[
    	ext{Dice coefficient} = rac{2 \cdot |A \cap B|}{|A| + |B|}
    \]
    where \(A\) is the predicted mask and \(B\) is the ground truth mask.
- **Loss**: The binary cross-entropy loss used during training to measure the difference between predicted probabilities and the true labels.

---

## Results

The model achieved the following results after **30 epochs**:

- **Training Accuracy**: 97.15%
- **Validation Accuracy**: 97.28%
- **Training Dice Coefficient**: 0.9025
- **Validation Dice Coefficient**: 0.9071

The **Dice coefficient** demonstrated steady improvement throughout training. The model achieved its best performance by **Epoch 28**, where the validation Dice coefficient peaked at **0.9130** and validation loss was at its lowest.

---

## Future Work

- **Data Augmentation**: Implementing advanced techniques like **rotation**, **scaling**, and **flipping** to further increase dataset diversity and reduce overfitting.
- **Advanced Architectures**: Experimenting with **Attention U-Net** or **Transformers** for better focusing on relevant regions of the image.
- **Transfer Learning**: Fine-tuning pre-trained models (e.g., **VGG16**, **ResNet**) to improve feature extraction and reduce training time.
- **Better Dataset**: Use a different dataset with more varied cell background so that the final model is more robust.

---

## License

This project is licensed under the MIT License – see the [LICENSE.md](LICENSE.md) file for details.

---

## Acknowledgments

- The **2018 Data Science Bowl** dataset was provided by **Kaggle** and the **Data Science Bowl** team.
- Special thanks to **Google Colab** for providing free access to GPU resources, which enabled efficient model training.
