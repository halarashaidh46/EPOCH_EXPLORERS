# EPOCH_EXPLORERS – Face Recognition Login with Jarvis Voice Assistant
# Face Detection Project

This project is deployed as a web app on Hugging Face Spaces: [Try it here](https://huggingface.co/spaces/JanaBot/face_detection)


## Project Overview
EPOCH_EXPLORERS introduces a secure face-recognition login system that seamlessly transitions into a voice-controlled AI assistant (Jarvis).  
The system uses a custom FaceNet model trained on FaceFast data to ensure accurate identity verification.  
Users can interact with Jarvis via voice commands or uploaded audio for tasks such as online search, date/time, weather updates, and more.  

This combination of computer vision and voice AI provides a smooth and interactive experience, ideal for hackathons and real-world applications.

## Key Features
- Secure login using face recognition  
- Dynamic interface with Gradio for smooth transitions  
- Jarvis AI assistant responds to voice commands or audio uploads  
- Tasks supported: online search, date/time, weather info, and more  
- High accuracy with custom FaceNet model trained on FaceFast dataset  

## Tech Stack
- Language: Python  
- Deep Learning: FaceNet (InceptionResnetV1)  
- Frameworks & Libraries: PyTorch, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, PIL, Gradio  
- Tools / IDE: JIVMS IDE, Google Colab  
- Model Hosting: Hugging Face  

## Project Structure
EPOCH_EXPLORERS/
│── data/ # FaceFast dataset
│── models/ # Pre-trained FaceNet model
│── src/ # Source code (login + assistant)
│── requirements.txt # Python dependencies
│── README.md # Project documentation


## Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/username/EPOCH_EXPLORERS.git
cd EPOCH_EXPLORERS


2. Install dependencies:
pip install -r requirements.txt


3. Run the Gradio app:

python src/app.py
## License
MIT License – Free to use, modify, and share with attribution.
Team

Epoch Explorers Team – Innovators in AI, combining face recognition and voice intelligence for interactive and secure applications.
