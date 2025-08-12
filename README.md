# GourmetVision: Food Image Recognition Platform

## Description  
GourmetVision is an AI-powered web application that accurately classifies and identifies food items from images using deep learning. The neural network was trained using Google AI Platform and deployed as a scalable web service via Python Flask and Google Cloud Run.

## Features  
- Deep convolutional neural network for high-accuracy food image classification  
- Web-based interface for easy image upload and instant recognition  
- Cloud-based training and deployment for scalability and reliability  

## Getting Started  

### Prerequisites  
- Python 3.7+  
- Flask  
- TensorFlow 2.x  
- Docker (for containerisation)  
- Google Cloud SDK (for deployment to Cloud Run)  

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/NicolaJB/Food-Image-Recognition-Web-App.git
   cd Food-Image-Recognition-Web-App
   ```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Flask app locally:
```bash
python app.py
```

## Deployment to Google Cloud Run

To deploy the Food Image Recognition App to Google Cloud Run, follow these steps:

1. **Set up Google Cloud SDK**  
   Install and initialise the Google Cloud SDK by following instructions at [Google Cloud SDK Installation](https://cloud.google.com/sdk/docs/install), then run:  
   ```bash
   gcloud init
   ```
2. Enable required Google Cloud APIs
Enable the Cloud Run and Container Registry APIs for your project:
```bash
gcloud services enable run.googleapis.com containerregistry.googleapis.com
```
3. Build the Docker image
From the root directory of the project (where the Dockerfile is located), build and submit the Docker container image to Google Container Registry:

```bash
gcloud builds submit --tag gcr.io/[PROJECT-ID]/gourmetvision
```
Replace [PROJECT-ID] with your Google Cloud project ID.

## Deploy to Cloud Run

Deploy the container image to Cloud Run with the following command:

```bash
gcloud run deploy gourmetvision \
  --image gcr.io/[PROJECT-ID]/gourmetvision \
  --platform managed \
  --region [REGION] \
  --allow-unauthenticated
```
Replace [PROJECT-ID] with your project ID and [REGION] with your preferred Google Cloud region (e.g., us-central1).

## Access the App
After successful deployment, the command output will provide the URL to access your live application.

### Usage
- Access the web app via your browser.
- Upload an image of a food item.
- The app will return the predicted food category.

### License
This project is licensed under the Apache License 2.0

### Acknowledgements
This application is based on the concepts and methods presented in the 2020 project demonstration 'Train and Deploy Tensorflow models using Google AI Platform' by Nour Islam Mokhtari. This project is intended as a prototype.

Thanks to Google Cloud Platform for providing the infrastructure and tools that enabled scalable training and deployment.
