# Food Recognition and Estimation

## Project Overview
This repository contains the code and resources for a food image recognition and estimation project. The project uses a subset of the Food-101 dataset to classify food images into three categories using a fine-tuned InceptionV3 model.

### Dataset
We use the [Food-101 dataset](https://www.kaggle.com/dansbecker/food-101) for this task, which contains 101 food categories.

The dataset used is a subset of the popular [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/), which contains images of different food classes. For this project, we focus on a smaller subset with 3 food categories:
- **Category 1**: Apple Pie
- **Category 2**: Pizza
- **Category 3**: Omelette

The dataset is split into training and validation sets, with 2250 training samples and 750 validation samples.

### Project Structure
- `data/`: Contains raw and preprocessed data.
- `notebooks/`: Jupyter notebooks for exploration, preprocessing, model training, and evaluation.
- `scripts/`: Python scripts for data loading, preprocessing, and model building.
- `models/`: Trained models saved during training.
- `app.py`: Flask or Streamlit app for inference.
- `tests/`: Unit tests for the project.

## Requirements

To run the notebook and replicate the results, you need to install the dependencies listed in `requirements.txt`.

## Key Steps

### 1. Data Preprocessing

Data augmentation techniques were applied to the training images to improve generalization. The following augmentations were used:
- **Rescaling**: All images were scaled by a factor of 1/255.
- **Shearing**: Random shearing was applied.
- **Zooming**: Random zooming.
- **Horizontal Flipping**: Random horizontal flips were performed.

The validation images were only rescaled.

### 2. Model Architecture

The model used is a pre-trained **InceptionV3** model from Keras, trained on the ImageNet dataset. The model was fine-tuned by adding custom layers on top of the pre-trained base:

- **Global Average Pooling Layer**: Reduces the feature map size.
- **Dense Layer**: Fully connected layer with 128 neurons and ReLU activation.
- **Dropout Layer**: Dropout regularization to prevent overfitting.
- **Output Layer**: Softmax activation function with 3 output classes.

### 3. Model Compilation

The model is compiled with the following parameters:
- **Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.0001 and momentum of 0.9.
- **Loss Function**: Categorical cross-entropy (since it is a multi-class classification problem).
- **Metrics**: Accuracy to track performance.

### 4. Model Training

The model was trained using the augmented training data for 30 epochs. The following callbacks were used during training:
- **ModelCheckpoint**: Saves the best model based on validation accuracy.
- **CSVLogger**: Logs the training history into a CSV file.

### 5. Evaluation and Prediction

After training, the model was evaluated using the validation dataset. Additionally, sample images of food items (apple pie, pizza, and omelette) were downloaded from the web, and the trained model predicted their classes.

### 6. Results

The model achieved good accuracy on the validation dataset, and the food items were correctly classified in the test images.

## Running the Code

1. Clone this repository:

    ```
    git clone https://github.com/juniorworku/PRODIGY_ML_05.git
    ```

2. Install the dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Open and run the Jupyter notebook:

    ```
    jupyter notebook Food_Recognition_and_Estimation.ipynb
    ```

## Conclusion

This project demonstrates how to fine-tune a pre-trained CNN model (InceptionV3) for food image classification using data augmentation techniques. The approach can be extended to larger datasets and more categories.

## License

This project is licensed under the MIT License.
