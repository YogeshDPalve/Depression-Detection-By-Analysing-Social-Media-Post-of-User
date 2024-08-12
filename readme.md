
# Depression Detection Using BERT

This project is a machine learning application designed to detect signs of depression in social media posts using the BERT algorithm. The application includes features such as early detection of depression symptoms, notifications to a guardian, suggestions for consulting doctors, and links to motivational YouTube videos.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [File Structure](#file-structure)
- [License](#license)

## Overview
This project analyzes social media posts to detect early signs of depression. If the model predicts that a user is at risk of depression, the application provides several interventions:
- Notifying a guardian via email.
- Suggesting a list of doctors for consultation.
- Providing motivational content through YouTube links.

## Features
- **Depression Detection:** Uses the BERT model to analyze social media text and predict whether the user might be experiencing depression.
- **Early Symptom Detection:** Helps in identifying early signs of depression to provide timely interventions.
- **Guardian Notification:** Sends an email notification to the user's guardian if depression is detected.
- **Doctor Suggestions:** Recommends a list of doctors for consultation.
- **Motivational Content:** Provides links to motivational YouTube videos for encouragement.

## Installation

### Prerequisites
- Python 3.7+
- Flask
- PyTorch
- Transformers

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/depression-detection.git
    cd depression-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained BERT model:
    Ensure that the pre-trained model is saved in the `model/` directory.

4. Set up environment variables (if any) in a `.env` file:
    ```env
    EMAIL_HOST=smtp.your-email.com
    EMAIL_PORT=587
    EMAIL_USER=your-email@example.com
    EMAIL_PASSWORD=your-password
    ```

## Usage
1. Run the Flask application:
    ```bash
    python app.py
    ```

2. Access the web interface:
   Open a web browser and go to `http://localhost:5000`.

3. Input the text you want to analyze, and the system will display the prediction.

## Model Details
The project uses the BERT model (`BertForSequenceClassification`) fine-tuned for sentiment analysis. The model is loaded from the `model/` directory and used to classify text inputs as either indicative of depression or not.

## File Structure
```plaintext
.
├── app.py                    # Main application file
├── model/
│   ├── config.json           # BERT model configuration
│   ├── pytorch_model.bin     # Pre-trained BERT model weights
├── static/
│   └── style.css             # Styling for the web interface
├── templates/
│   └── index.html            # Frontend HTML file
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
