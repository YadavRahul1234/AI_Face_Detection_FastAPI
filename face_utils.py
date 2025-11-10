import dlib
import cv2
import numpy as np
import base64
import json
from typing import Tuple, Optional

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array image."""
    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image

def encode_face(image: np.ndarray) -> Optional[np.ndarray]:
    """Encode face from image using dlib."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image)
    if not faces:
        return None
    # Use the first detected face
    shape = sp(rgb_image, faces[0])
    face_descriptor = facerec.compute_face_descriptor(rgb_image, shape)
    return np.array(face_descriptor)

def match_face(unknown_encoding: np.ndarray, known_encodings: list) -> Tuple[Optional[str], float]:
    """Match unknown face encoding with known encodings using Euclidean distance."""
    if not known_encodings:
        return None, 0.0

    distances = [np.linalg.norm(unknown_encoding - known) for known in known_encodings]
    min_distance_idx = np.argmin(distances)
    min_distance = distances[min_distance_idx]

    # Tolerance threshold: 0.45
    # 80% match cutoff: distance < 0.2 (higher confidence)
    if min_distance < 0.45:
        return f"Employee_{min_distance_idx}", min_distance
    return None, min_distance

def load_encodings_from_db(db_encodings: list) -> list:
    """Load face encodings from database JSON strings."""
    encodings = []
    for enc_str in db_encodings:
        enc_list = json.loads(enc_str)
        encodings.append(np.array(enc_list))
    return encodings
