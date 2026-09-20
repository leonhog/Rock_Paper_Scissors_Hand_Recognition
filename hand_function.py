import numpy as np


# ============================================================
# LANDMARK DISTANCE
# ============================================================

def distance_landmarks(hand_landmarks, point1, point2):
    """
    Calculate the 3D Euclidean distance between two hand landmarks.

    Each landmark contains three coordinates:
        x -> horizontal position
        y -> vertical position
        z -> depth position
    """

    # Get the 3D coordinates of the first landmark
    p1 = np.array([
        hand_landmarks[point1].x,
        hand_landmarks[point1].y,
        hand_landmarks[point1].z
    ])


    # Get the 3D coordinates of the second landmark
    p2 = np.array([
        hand_landmarks[point2].x,
        hand_landmarks[point2].y,
        hand_landmarks[point2].z
    ])


    # Return the Euclidean distance between the two points
    return np.linalg.norm(p1 - p2)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_value(hand_landmarks):
    """
    Create the normalized input values used by the
    Rock-Paper-Scissors prediction model.

    For each finger, the distance from the fingertip
    to the wrist is divided by the distance from the
    finger base to the wrist.

    This normalization makes the values less dependent
    on the size or distance of the hand from the camera.
    """

    # Landmark indices for each finger
    #
    # Format:
    # ("finger_name", fingertip_index, finger_base_index)
    finger_info = [
        ("thumb", 4, 2),
        ("index", 8, 5),
        ("middle", 12, 9),
        ("ring", 16, 13),
        ("pinky", 20, 17)
    ]


    # List that will contain the normalized values
    values = []


    # Calculate one value for each finger
    for _, fingertip, finger_base in finger_info:

        # Distance from the fingertip to the wrist
        fingertip_distance = distance_landmarks(
            hand_landmarks,
            fingertip,
            0
        )


        # Distance from the finger base to the wrist
        base_distance = distance_landmarks(
            hand_landmarks,
            finger_base,
            0
        )


        # Normalize the fingertip distance
        normalized_distance = fingertip_distance / base_distance

        values.append(normalized_distance)


    # Convert the list into a NumPy array
    # and reshape it into one row for the ML model.
    return np.array(values).reshape(1, -1)


# ============================================================
# CONVERT LANDMARKS TO PIXEL COORDINATES
# ============================================================

def compute_pixel_point(hand_landmarks, width, height):
    """
    Convert MediaPipe normalized landmark coordinates
    into pixel coordinates.

    MediaPipe coordinates are normalized between 0 and 1,
    while OpenCV uses pixel coordinates.
    """

    points = []


    # Convert every landmark into pixel coordinates
    for landmark in hand_landmarks:

        # Convert normalized X coordinate to pixels
        x = int(landmark.x * width)

        # Convert normalized Y coordinate to pixels
        y = int(landmark.y * height)


        # Store the point as an (x, y) tuple
        points.append((x, y))


    return points
