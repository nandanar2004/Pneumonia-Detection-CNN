# Pneumonia Detection Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify chest X-ray images into Pneumonia and Normal categories.

The model is developed using Python and TensorFlow/Keras.

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Project Workflow
1. Dataset collection
2. Image preprocessing
3. Image resizing and normalization
4. CNN model development
5. Model training
6. Model evaluation
7. Pneumonia prediction using chest X-ray images

## Model Performance

The trained CNN model was evaluated on 624 test images.

| Evaluation Metric | Score |
|---|---:|
| Test Accuracy | 80.93% |
| Precision | 82.19% |
| Recall | 88.72% |
| AUC | 88.05% |
| Test Loss | 0.4900 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| NORMAL | 0.78 | 0.68 | 0.73 |
| PNEUMONIA | 0.82 | 0.89 | 0.85 |

The model achieved 80.93% test accuracy on the evaluated dataset.


## Project Files

| File | Description |
|---|---|
| model2_cnn_pneumonia_detection (1).ipynb | CNN implementation and training |
| model2_final.keras | Trained CNN model |

## Model Details
- Model Type: Convolutional Neural Network
- Framework: TensorFlow/Keras
- Task: Binary Image Classification
- Classes: Pneumonia and Normal

## Future Improvements
- Develop a Streamlit web application
- Improve model performance
- Add more evaluation visualizations

## Streamlit Web Application

This project includes a Streamlit interface for uploading chest X-ray images and generating predictions using the trained CNN model.

### Run the Application

1. Clone the repository:

```bash
git clone https://github.com/nandanan2004/Pneumonia-Detection-CNN.git
```

2. Navigate to the project directory:

```bash
cd Pneumonia-Detection-CNN
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run Streamlit:

```bash
streamlit run app.py
```

### Application Features
- Upload chest X-ray images.
- Preview uploaded images.
- Predict NORMAL or PNEUMONIA.
- Display pneumonia probability.


## Application Screenshots

### Home Page
![Home Page](screenshots/home.png)

### Uploaded Chest X-ray
![Uploaded X-ray](screenshots/uploaded_image.png)

### Prediction Result
![Prediction Result](screenshots/prediction_result.png)

## Disclaimer
This project is for educational purposes only and is not intended for medical diagnosis.

## Disclaimer
This project is developed for educational purposes only. It is not intended to replace professional medical diagnosis.
