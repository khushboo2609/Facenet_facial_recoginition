import cv2
import numpy as np
import os
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_FILE = "embeddings/facenet_embeddings.npz"

# Initial experimental threshold
# Smaller cosine distance = more similar
UNKNOWN_THRESHOLD = 0.50

PRINT_INTERVAL = 0.5


# ============================================================
# LOAD SAVED EMBEDDINGS
# ============================================================

if not os.path.exists(EMBEDDING_FILE):
    print("ERROR: FaceNet embeddings file not found!")
    print("Expected:", EMBEDDING_FILE)
    print("Run generate_embeddings.py first.")
    exit()

data = np.load(EMBEDDING_FILE)

embeddings = data["embeddings"]
names = data["names"]
files = data["files"]

print("=" * 60)
print("              FACENET REAL-TIME RECOGNITION")
print("=" * 60)

print(f"Total embeddings loaded: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")


# ============================================================
# CREATE REFERENCE EMBEDDING
# ============================================================

# Current dataset:
# Images 1-40 = reference/training images
# Images 41-50 = held-out test images

reference_embeddings = embeddings[:40]

# Calculate mean embedding
reference_embedding = np.mean(reference_embeddings, axis=0)

# L2 normalize the reference embedding
reference_embedding = reference_embedding / (
    np.linalg.norm(reference_embedding) + 1e-10
)

print(f"Reference embeddings: {len(reference_embeddings)}")
print(f"Unknown threshold: {UNKNOWN_THRESHOLD}")

print("=" * 60)


# ============================================================
# LOAD FACENET AND MTCNN
# ============================================================

print("\nLoading MTCNN face detector...")
detector = MTCNN()

print("Loading pretrained FaceNet model...")
embedder = FaceNet()

print("FaceNet model loaded successfully!")


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\nWebcam started.")
print("Press Q to quit.")
print("=" * 60)


last_print_time = 0


# ============================================================
# REAL-TIME RECOGNITION LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame from webcam.")
        break

    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # Detect faces using MTCNN
    # --------------------------------------------------------

    detections = detector.detect_faces(rgb_frame)

    # --------------------------------------------------------
    # Process detected faces
    # --------------------------------------------------------

    for detection in detections:

        x, y, w, h = detection["box"]

        # MTCNN can sometimes return negative coordinates
        x = max(0, x)
        y = max(0, y)

        x2 = min(rgb_frame.shape[1], x + w)
        y2 = min(rgb_frame.shape[0], y + h)

        # Make sure bounding box is valid
        if x2 <= x or y2 <= y:
            continue

        # ----------------------------------------------------
        # Crop face
        # ----------------------------------------------------

        face = rgb_frame[y:y2, x:x2]

        if face.size == 0:
            continue

        # ----------------------------------------------------
        # Resize to FaceNet input size
        # ----------------------------------------------------

        face = cv2.resize(face, (160, 160))

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        face_input = np.expand_dims(face, axis=0)

        # ----------------------------------------------------
        # Generate FaceNet embedding
        # ----------------------------------------------------

        embedding = embedder.embeddings(face_input)[0]

        # ----------------------------------------------------
        # Normalize embedding
        # ----------------------------------------------------

        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        # ----------------------------------------------------
        # Calculate cosine distance
        # ----------------------------------------------------

        cosine_similarity = np.dot(
            embedding,
            reference_embedding
        )

        cosine_distance = 1 - cosine_similarity

        # ----------------------------------------------------
        # Recognition decision
        # ----------------------------------------------------

        if cosine_distance <= UNKNOWN_THRESHOLD:

            identity = "Khushboo"
            status = "MATCH"

        else:

            identity = "Unknown"
            status = "UNKNOWN"

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        if status == "MATCH":
            box_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x, y),
            (x2, y2),
            box_color,
            2
        )

        # ----------------------------------------------------
        # Display name
        # ----------------------------------------------------

        label = f"{identity} | Distance: {cosine_distance:.4f}"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

        # ----------------------------------------------------
        # Display status
        # ----------------------------------------------------

        cv2.putText(
            frame,
            status,
            (x, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

        # ----------------------------------------------------
        # Print result periodically
        # ----------------------------------------------------

        current_time = time.time()

        if current_time - last_print_time >= PRINT_INTERVAL:

            print(
                f"{identity} | "
                f"Cosine distance: {cosine_distance:.6f} | "
                f"{status}"
            )

            last_print_time = current_time

    # --------------------------------------------------------
    # Show webcam
    # --------------------------------------------------------

    cv2.imshow(
        "FaceNet Real-Time Recognition",
        frame
    )

    # --------------------------------------------------------
    # Press Q to quit
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nWebcam recognition stopped.")
print("FaceNet real-time recognition completed.")