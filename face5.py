import dlib
import cv2
import numpy as np

# Load pre-trained models for face detection and facial landmarks
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load images and extract face descriptors
def extract_face_descriptor(image_path):
    img = cv2.imread(image_path)
    dets = detector(img, 1)

    if len(dets) == 0:
        print(f"No face detected in {image_path}. Please provide an image with a face.")
        return None  # No face detected, return None

    for detection in dets:
        shape = shape_predictor(img, detection)
        face_descriptor = np.array(facerec.compute_face_descriptor(img, shape))
        return face_descriptor

# Function to calculate Euclidean distance between two descriptors
def euclidean_distance(descriptor1, descriptor2):
    if descriptor1 is None or descriptor2 is None:
        return float('inf')  # Return a large value indicating no match
    
    return np.linalg.norm(descriptor1 - descriptor2)

# Enroll users
user1_descriptor = extract_face_descriptor('palm1.JPG')
user2_descriptor = extract_face_descriptor('user2.jpeg')

# Match input image against enrolled users
input_descriptor = extract_face_descriptor('user2.jpeg')

# Set threshold for similarity
threshold = 0.6

# Compare face descriptors using Euclidean distance
distance_to_user1 = euclidean_distance(input_descriptor, user1_descriptor)
distance_to_user2 = euclidean_distance(input_descriptor, user2_descriptor)

if distance_to_user1 < threshold:
    print("Input image matches with User 1")
elif distance_to_user2 < threshold:
    print("Input image matches with User 2")
else:
    print("No face detected or no match found")
