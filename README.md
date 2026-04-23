# Grey Water Classifier

A machine learning project that classifies grey water quality using image analysis. This project uses TensorFlow and MobileNetV2 to categorize water samples into three quality levels: high, low, and medium.

## Features

- **Image Classification**: Classifies water quality from images into three categories (high, low, medium)
- **Deep Learning Model**: Uses MobileNetV2 pre-trained model with fine-tuning
- **Web Interface**: Streamlit-based web application for easy prediction
- **Training Scripts**: Complete training pipeline with data preprocessing
- **Evaluation Tools**: Model evaluation and performance metrics
- **Batch Prediction**: Command-line tools for batch processing

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd grey_water_classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model from scratch:

```bash
python train.py
```

This will:
- Load images from the `dataset/` folder
- Preprocess the data
- Train the MobileNetV2 model
- Save the trained model to `model/grey_water_model.h5`

### Running the Web App

To start the Streamlit web application:

```bash
streamlit run app/app.py
```

This will launch a web interface where you can upload water sample images and get quality predictions.

### Command-Line Prediction

To classify a single image:

```bash
python predict.py
```

Follow the prompts to select an image file for classification.

### Evaluating the Model

To evaluate model performance:

```bash
python evaluate.py
```

### Checking Dataset

To analyze the dataset:

```bash
python check_dataset.py
```

## Project Structure

```
grey_water_classifier/
├── app/
│   └── app.py                 # Streamlit web application
├── dataset/
│   ├── high/                  # High quality water images
│   ├── low/                   # Low quality water images
│   └── medium/                # Medium quality water images
├── model/
│   └── grey_water_model.h5    # Trained model file
├── notebooks/                 # Jupyter notebooks (if any)
├── app.py                     # Main application entry point
├── check_dataset.py           # Dataset analysis script
├── evaluate.py                # Model evaluation script
├── predict.py                 # Single image prediction script
├── preprocess.py              # Data preprocessing utilities
├── train.py                   # Model training script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.20.0+
- NumPy 2.0.0+
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit
- Seaborn

## Model Details

- **Architecture**: MobileNetV2 with custom classification head
- **Input Size**: 224x224 pixels
- **Classes**: 3 (high, low, medium)
- **Training**: Transfer learning with fine-tuning
- **Framework**: TensorFlow/Keras

## Dataset

The model expects images organized in the following structure:
```
dataset/
├── high/
├── low/
└── medium/
```

Each folder should contain images of water samples corresponding to their quality level.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license here]

## Contact

[Add contact information here]