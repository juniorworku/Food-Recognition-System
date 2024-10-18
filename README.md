# Food Recognition and Calorie Estimation

## Project Overview
This project aims to develop a model that accurately recognizes food items from images and estimates their calorie content. It helps users track their dietary intake and make informed food choices.

### Dataset
We use the [Food-101 dataset](https://www.kaggle.com/dansbecker/food-101) for this task, which contains 101 food categories.

### Project Structure
- `data/`: Contains raw and preprocessed data.
- `notebooks/`: Jupyter notebooks for exploration, preprocessing, model training, and evaluation.
- `scripts/`: Python scripts for data loading, preprocessing, and model building.
- `models/`: Trained models saved during training.
- `app.py`: Flask or Streamlit app for inference.
- `tests/`: Unit tests for the project.
  
### How to Run
1. Clone the repository:

```sh
git clone https://github.com/juniiorworku/PRODIGY_ML_05.git

```
2. Install dependencies:

```sh
pip install -r requirements.txt
```
3. Download the dataset and extract it into `data/food-101/`.

4. Run the notebooks to train the models or use pre-trained models for inference.