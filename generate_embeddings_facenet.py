import os
import cv2
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings"

IMAGE_SIZE = (160, 160)

# ==========================================
# CREATE EMBEDDINGS FOLDER
# ==========================================

os.makedirs(
    EMBEDDINGS_PATH,
    exist_ok=True
)


# ==========================================
# LOAD MODELS
# ==========================================

print("=" * 60)
print("             FACENET EMBEDDING GENERATION")
print("=" * 60)

print()
print("Loading MTCNN...")

detector = MTCNN()

print("MTCNN loaded.")

print()
print("Loading FaceNet...")

embedder = FaceNet()

print("FaceNet loaded.")


# ==========================================
# STORAGE
# ==========================================

all_embeddings = []
all_names = []
all_files = []


# ==========================================
# PROCESS DATASET
# ==========================================

for person_name in sorted(
    os.listdir(DATASET_PATH)
):

    person_folder = os.path.join(
        DATASET_PATH,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue


    print()
    print("=" * 60)
    print(f"Person: {person_name}")
    print("=" * 60)


    image_files = [
        f
        for f in os.listdir(person_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    # Numeric sorting

    try:

        image_files = sorted(
            image_files,
            key=lambda x: int(
                os.path.splitext(x)[0]
            )
        )

    except ValueError:

        image_files = sorted(
            image_files
        )


    successful = 0
    failed = 0


    # ======================================
    # PROCESS EACH IMAGE
    # ======================================

    for image_name in image_files:

        image_path = os.path.join(
            person_folder,
            image_name
        )


        print(
            f"Processing {image_name}...",
            end=" "
        )


        # ----------------------------------
        # READ IMAGE
        # ----------------------------------

        image = cv2.imread(
            image_path
        )


        if image is None:

            print("FAILED - image could not be read")

            failed += 1

            continue


        # ----------------------------------
        # BGR -> RGB
        # ----------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------
        # MTCNN FACE DETECTION
        # ----------------------------------

        detections = detector.detect_faces(
            image_rgb
        )


        if len(detections) == 0:

            print("FAILED - no face detected")

            failed += 1

            continue


        # ----------------------------------
        # SELECT LARGEST FACE
        # ----------------------------------

        largest_face = max(
            detections,
            key=lambda x:
            x["box"][2] * x["box"][3]
        )


        x, y, w, h = largest_face["box"]


        # Prevent negative coordinates

        x = max(0, x)
        y = max(0, y)


        # ----------------------------------
        # CROP FACE
        # ----------------------------------

        face = image_rgb[
            y:y+h,
            x:x+w
        ]


        if face.size == 0:

            print("FAILED - invalid face crop")

            failed += 1

            continue


        # ----------------------------------
        # RESIZE FACE
        # ----------------------------------

        face = cv2.resize(
            face,
            IMAGE_SIZE
        )


        # ----------------------------------
        # GENERATE FACENET EMBEDDING
        # ----------------------------------

        embedding = embedder.embeddings(
            np.expand_dims(
                face,
                axis=0
            )
        )[0]


        # ----------------------------------
        # NORMALIZE EMBEDDING
        # ----------------------------------

        embedding = embedding / np.linalg.norm(
            embedding
        )


        # ----------------------------------
        # STORE
        # ----------------------------------

        all_embeddings.append(
            embedding
        )

        all_names.append(
            person_name
        )

        all_files.append(
            image_name
        )


        successful += 1

        print("SUCCESS")


    print()
    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )


# ==========================================
# CHECK RESULTS
# ==========================================

if len(all_embeddings) == 0:

    print()
    print("ERROR: No embeddings were generated.")

    exit()


# ==========================================
# CONVERT TO NUMPY
# ==========================================

all_embeddings = np.array(
    all_embeddings
)

all_names = np.array(
    all_names
)

all_files = np.array(
    all_files
)


# ==========================================
# SAVE EMBEDDINGS
# ==========================================

output_file = os.path.join(
    EMBEDDINGS_PATH,
    "facenet_embeddings.npz"
)


np.savez(
    output_file,
    embeddings=all_embeddings,
    names=all_names,
    files=all_files
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print()
print("=" * 60)
print("             EMBEDDING SUMMARY")
print("=" * 60)

print()

print(
    f"Total embeddings: "
    f"{len(all_embeddings)}"
)

print(
    f"Embedding shape: "
    f"{all_embeddings.shape}"
)

print(
    f"Embedding dimensions: "
    f"{all_embeddings.shape[1]}"
)

print()

print(
    f"Saved to: "
    f"{output_file}"
)

print()
print("=" * 60)
print("        FACENET EMBEDDING GENERATION DONE")
print("=" * 60)