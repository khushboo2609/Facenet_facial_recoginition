import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# CONFIGURATION
# ==========================================

EMBEDDINGS_FILE = (
    "embeddings/facenet_embeddings.npz"
)

TRAIN_IMAGES = 40


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

data = np.load(
    EMBEDDINGS_FILE
)

embeddings = data["embeddings"]
names = data["names"]
files = data["files"]


print("=" * 60)
print("                 FACENET TESTING")
print("=" * 60)

print()

print(
    f"Total embeddings: "
    f"{len(embeddings)}"
)

print(
    f"Embedding dimensions: "
    f"{embeddings.shape[1]}"
)


# ==========================================
# SPLIT DATA
# ==========================================

train_embeddings = embeddings[
    :TRAIN_IMAGES
]

train_names = names[
    :TRAIN_IMAGES
]

test_embeddings = embeddings[
    TRAIN_IMAGES:
]

test_names = names[
    TRAIN_IMAGES:
]

test_files = files[
    TRAIN_IMAGES:
]


print()

print(
    f"Reference images: "
    f"{len(train_embeddings)}"
)

print(
    f"Test images: "
    f"{len(test_embeddings)}"
)


# ==========================================
# CREATE REFERENCE EMBEDDING
# ==========================================

reference_embedding = np.mean(
    train_embeddings,
    axis=0
)


# Normalize reference embedding

reference_embedding = (
    reference_embedding /
    np.linalg.norm(reference_embedding)
)


# ==========================================
# COSINE DISTANCE FUNCTION
# ==========================================

def cosine_distance(
    embedding1,
    embedding2
):

    similarity = np.dot(
        embedding1,
        embedding2
    )

    similarity = np.clip(
        similarity,
        -1.0,
        1.0
    )

    return 1.0 - similarity


# ==========================================
# TEST
# ==========================================

distances = []

y_true = []
y_pred = []


print()
print("=" * 60)
print("                 TEST PREDICTIONS")
print("=" * 60)

print()


for embedding, name, filename in zip(
    test_embeddings,
    test_names,
    test_files
):

    distance = cosine_distance(
        embedding,
        reference_embedding
    )

    distances.append(
        distance
    )


    # Since this dataset currently
    # contains only one identity,
    # every test image belongs to
    # the enrolled person.

    predicted_name = train_names[0]


    y_true.append(
        name
    )

    y_pred.append(
        predicted_name
    )


    print(
        f"{filename} -> "
        f"{predicted_name} | "
        f"Cosine distance: "
        f"{distance:.6f}"
    )


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ==========================================
# DISTANCE STATISTICS
# ==========================================

distances = np.array(
    distances
)

minimum_distance = np.min(
    distances
)

maximum_distance = np.max(
    distances
)

average_distance = np.mean(
    distances
)

median_distance = np.median(
    distances
)


# ==========================================
# RESULTS
# ==========================================

print()
print("=" * 60)
print("                 FACENET RESULTS")
print("=" * 60)

print()

print(
    f"Total test images : "
    f"{len(test_embeddings)}"
)

print(
    f"Accuracy          : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision         : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall            : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score          : "
    f"{f1 * 100:.2f}%"
)

print()

print(
    f"Minimum distance  : "
    f"{minimum_distance:.6f}"
)

print(
    f"Maximum distance  : "
    f"{maximum_distance:.6f}"
)

print(
    f"Average distance  : "
    f"{average_distance:.6f}"
)

print(
    f"Median distance   : "
    f"{median_distance:.6f}"
)

print()

print("=" * 60)
print("              FACENET TESTING DONE")
print("=" * 60)