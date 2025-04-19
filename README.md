# Speech Emotion Recognition System

## Overview
This project aims to classify emotions from speech signals using a deep learning model. The model was developed using TensorFlow and Librosa, leveraging key audio features like Mel Frequency Cepstral Coefficients (MFCCs) to achieve high accuracy in classifying emotions from speech.

## Features
- **Emotion Classification**: Classifies emotions like happy, sad, angry, etc., from speech signals.
- **Preprocessing**: Extracts important audio features, such as MFCCs, to improve model training.
- **Real-time Detection**: Implements real-time emotion detection on audio streams.

## Technologies Used
- **TensorFlow**: For building and training the deep learning model.
- **Librosa**: For audio signal processing and feature extraction.
- **Python**: The programming language used for the implementation.

## Model Training
The model is trained on a labeled dataset of audio clips, where each clip is associated with a particular emotion. The following steps were involved:
1. **Preprocessing**:
   - Audio signals are preprocessed by converting them into spectrograms.
   - Mel Frequency Cepstral Coefficients (MFCCs) are extracted as key features.
2. **Model Architecture**:
   - A deep neural network (DNN) was used for emotion classification.
   - The model consists of multiple layers, including convolutional and fully connected layers.
3. **Training**:
   - The model was trained on the preprocessed data, achieving a classification accuracy of 88%.

## How to Run
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abdinshaikh/Speech-Emotion-Recognition
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

### Running the Model
1. To train the model:
   ```bash
   python train_model.py
2. To detect emotion in real-time from an audio stream:
   ```bash
   python real_time_detection.py
   
### Results
- The model achieved an accuracy of 88% in classifying emotions from speech data.

### Dataset
- The model was trained on the RAVDESS dataset, which contains audio files of emotional speech and song, covering a range of emotions.

### Future Improvements
- Fine-tuning the model for better accuracy.

- Implementing a web-based interface for real-time emotion recognition.

- Training on a more diverse dataset for improved generalization.
