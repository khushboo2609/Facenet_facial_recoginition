# FaceNet Face Recognition using Deep Learning

## 📌 Project Overview

This project explores **FaceNet**, a deep-learning-based face recognition approach that represents each detected face as a numerical feature vector called a *face embedding*. These embeddings can be compared to determine whether two face images belong to the same person.

FaceNet was introduced in the research paper *FaceNet: A Unified Embedding for Face Recognition and Clustering* by Florian Schroff, Dmitry Kalenichenko, and James Philbin.

The project can be used as a learning foundation for face verification, face identification, and face clustering.

## 🎯 Objectives

- Understand how FaceNet represents facial features using embeddings.
- Extract embeddings from face images.
- Compare face embeddings using a distance or similarity metric.
- Recognize known people by comparing a test image against a reference gallery.
- Evaluate recognition performance on known and unknown identities.
- Understand the strengths and limitations of deep-learning-based face recognition.

## 🧠 What is FaceNet?

FaceNet is a deep learning approach for learning a compact numerical representation of a face. Instead of directly assigning a name to an image, it maps a face into an embedding space where images of the same person should be close together and images of different people should generally be farther apart.

A common implementation uses a pretrained FaceNet model to generate a 128-dimensional embedding, although embedding dimensions depend on the model implementation.

## ⚙️ How It Works

1. **Input image:** Load a face image from a file, webcam, or application.
2. **Face detection:** Locate the face using a detector such as MTCNN or another compatible face detector.
3. **Face alignment and preprocessing:** Crop, align, resize, and normalize the face according to the model's requirements.
4. **Embedding extraction:** Pass the preprocessed face through the pretrained FaceNet model.
5. **Gallery creation:** Store reference embeddings alongside identity labels.
6. **Distance calculation:** Compare a test embedding with gallery embeddings, commonly using Euclidean distance or cosine similarity.
7. **Identity decision:** Select a matching identity when the score satisfies a validated threshold; otherwise, return `Unknown`.
8. **Evaluation:** Measure recognition accuracy and false-accept/false-reject behavior on a suitable test dataset.

## 🛠️ Technologies Used

- Python
- TensorFlow/Keras or a compatible pretrained FaceNet implementation
- NumPy
- OpenCV
- MTCNN or another compatible face detector
- Scikit-learn (optional, for evaluation and clustering)
- Matplotlib (optional, for visualization)

## 📂 Suggested Project Structure

```text
facenet-face-recognition/
├── dataset/
│   ├── gallery/
│   └── test/
├── embeddings/
│   └── gallery_embeddings.npz
├── models/
│   └── pretrained_model_files
├── scripts/
│   ├── extract_embeddings.py
│   ├── build_gallery.py
│   ├── recognize_face.py
│   └── evaluate_model.py
├── results/
│   └── evaluation_results.csv
├── requirements.txt
├── .gitignore
└── README.md
```

*This is a suggested structure. Update the names and paths to match the files in your actual repository.*

## 💻 Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd facenet-face-recognition
```

Replace `<YOUR_REPOSITORY_URL>` with your GitHub repository URL.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Important:** FaceNet implementations have different dependency requirements. Confirm that your chosen pretrained model is compatible with your installed Python, TensorFlow/Keras, and detector versions.

## ▶️ Usage

The following commands illustrate a suggested workflow. Replace the script names with the actual scripts in your project.

### Step 1: Extract embeddings

```bash
python scripts/extract_embeddings.py
```

### Step 2: Build the reference gallery

```bash
python scripts/build_gallery.py
```

### Step 3: Recognize a face

```bash
python scripts/recognize_face.py
```

### Step 4: Evaluate recognition performance

```bash
python scripts/evaluate_model.py
```

The scripts must be implemented in your repository before these commands will work.

## 📊 Evaluation Metrics

Evaluate the model using separate reference and test images. Where possible, ensure that test images are not reused as gallery images.

- **Known-identity accuracy:** Correct identifications among evaluated known-identity samples.
- **False acceptance rate:** Unknown people incorrectly accepted as a known identity.
- **False rejection rate:** Genuine matches rejected by the system.
- **Precision and recall:** Useful when evaluating identity-level predictions.
- **Inference time:** Time needed for detection, preprocessing, and embedding extraction.

For an open-set recognition system, test multiple thresholds and report the trade-off between correctly recognizing known people and rejecting unknown people. Do not choose a threshold solely on the final test set.

## ✅ Advantages

- Learns facial representations from data rather than relying on manually designed facial features.
- Supports face verification and identification using embedding comparisons.
- Reference embeddings can be stored and compared without retraining the embedding model for every newly registered identity.
- Can support face clustering and duplicate-face analysis.

## ⚠️ Limitations

- Performance depends on image quality, face detection, alignment, lighting, pose, and model selection.
- A distance threshold that works for one dataset may not work well for another.
- Similarity scores are not automatically calibrated probabilities.
- A pretrained model may have licensing, demographic-performance, or dataset restrictions.
- Recognition results should be evaluated on representative data before practical deployment.

## 🔮 Future Improvements

- Build a real-time webcam recognition interface.
- Add multi-face detection and per-face recognition.
- Implement registration and deletion of identities.
- Add unknown-person rejection and threshold calibration.
- Store embeddings in a database or vector index.
- Develop a FastAPI backend and a web interface.
- Add tests for image quality, multiple faces, and open-set recognition.

## 🔐 Privacy and Responsible Use

Face embeddings and face images can be sensitive biometric data. Obtain appropriate consent, secure stored images and embeddings, restrict access, and follow applicable privacy laws and organizational policies. Avoid using this project for consequential decisions without rigorous validation and safeguards.

## 📚 References

1. Schroff, F., Kalenichenko, D., and Philbin, J. **FaceNet: A Unified Embedding for Face Recognition and Clustering.** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
2. Official TensorFlow documentation: https://www.tensorflow.org/
3. OpenCV documentation: https://docs.opencv.org/
4. MTCNN project documentation, if used by the implementation.

## 👩‍💻 Author

**Khushboo Kumari**

Face Recognition | Deep Learning | Computer Vision
