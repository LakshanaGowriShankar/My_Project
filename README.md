 
# Cotton Disease Classification Using Transfer Learning

This project is a transfer learning pipeline adapted from STMicroelectronics' [STM32Cube.AI resources](https://www.st.com/en/embedded-software/x-cube-ai.html) for classifying cotton leaf diseases.

The original notebook is licensed under the BSD-3-Clause license and was adapted to use a custom dataset for training and evaluation.

---

## 📁 Project Structure

- `cotton_disease_transfer_learning.py` - Core logic for training using MobileNet (V1/V2)
- `model_quant.tflite` - Quantized model for deployment (generated after training)
- `README.md` - Project overview and setup
- `LICENSE` - BSD-3-Clause license from STMicroelectronics

---

## 🚀 How to Run

This was originally run in Google Colab using GPU acceleration.

### Steps:
1. Mount your Google Drive.
2. Upload your dataset under `MyDrive/train` and `MyDrive/test`.
3. Run all cells to:
   - Train a MobileNet model using transfer learning
   - Evaluate accuracy and visualize results
   - Quantize the model using TensorFlow Lite

---
## 📦 Dataset

This project uses a custom cotton leaf disease dataset.

Due to size and licensing constraints, the dataset is **not included** in this repository.

To use the project:
1. Upload your training and test images to Google Drive as:
MyDrive/
├── train/
│ ├── diseased/
│ └── healthy/
└── test/
├── diseased/
└── healthy/

pgsql
Copy
Edit
2. Make sure the folder names match the class labels.
3. Mount Google Drive in Colab to access the dataset.

### 🔧 Adaptations made:
- Replaced original dataset with a custom cotton disease dataset (structured into `train/` and `test/` folders)
- Adjusted model input size and MobileNet version as needed
- Customized data augmentation pipeline using `imgaug`
- Modified training loop to include early stopping
- Added evaluation plots (accuracy/loss curves, confusion matrix, misclassified image visualization)
- Quantized and tested the trained model using TensorFlow Lite for performance analysis

These changes were made to adapt the model to a cotton leaf disease classification task for learning and experimentation purposes.


## 🛡 License

This project is distributed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause) as per the original STMicroelectronics implementation.

The original source is available at:
[https://github.com/STMicroelectronics/stm32ai](https://github.com/STMicroelectronics/stm32ai)

All changes made in this repository are for academic/non-commercial use only.
