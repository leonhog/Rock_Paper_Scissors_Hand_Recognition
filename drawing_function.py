import cv2


# ============================================================
# HAND LANDMARK CONNECTIONS
# ============================================================

# Define the connections between MediaPipe hand landmarks.
#
# Each tuple represents:
# (start_landmark, end_landmark)
#
# These connections are used to draw the hand skeleton.
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),

    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),

    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),

    # Connections between the fingers
    (5, 9),
    (9, 13),
    (13, 17),

    # Connection between the wrist and the pinky base
    (0, 17)
]


# ============================================================
# DRAW A LANDMARK
# ============================================================

def draw_circle(frame, x, y):
    """
    Draw a circle at the specified pixel coordinates.

    Parameters:
        frame : OpenCV image
        x     : horizontal pixel coordinate
        y     : vertical pixel coordinate
    """

    cv2.circle(
        frame,
        (x, y),
        5,                  # Circle radius
        (0, 255, 0),        # Green color (BGR)
        -1                  # Filled circle
    )


# ============================================================
# DRAW HAND CONNECTIONS
# ============================================================

def draw_line_connection(frame, points, connections=HAND_CONNECTIONS):
    """
    Draw lines between connected hand landmarks.

    Parameters:
        frame       : OpenCV image
        points      : List of hand landmark pixel coordinates
        connections : List of landmark pairs to connect
    """

    # Go through every connection
    for start, end in connections:

        # Get the coordinates of the two landmarks
        x1, y1 = points[start]
        x2, y2 = points[end]


        # Draw a line between the two landmarks
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),    # Green color (BGR)
            2               # Line thickness
        )


# ============================================================
# DRAW A BOX AROUND THE HAND
# ============================================================

def draw_box_hand(frame, points, width, height, color):
    """
    Draw a rectangular box around the detected hand.

    The box is calculated from the minimum and maximum
    X and Y coordinates of the hand landmarks.

    Parameters:
        frame  : OpenCV image
        points : List of hand landmark pixel coordinates
        width  : Frame width
        height : Frame height
        color  : Box color in BGR format
    """

    # Find the minimum and maximum X coordinates
    x_min = min(x for x, y in points)
    x_max = max(x for x, y in points)

    # Find the minimum and maximum Y coordinates
    y_min = min(y for x, y in points)
    y_max = max(y for x, y in points)


    # Add some space around the hand
    margin = 20


    # Make sure the box stays inside the image
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)

    x_max = min(width, x_max + margin)
    y_max = min(height, y_max + margin)


    # Draw the rectangle around the hand
    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        color,
        2
    )


# ============================================================
# DRAW TEXT ABOVE THE HAND
# ============================================================

def draw_text(frame, text, points):
    """
    Display a text label above the detected hand.

    Parameters:
        frame  : OpenCV image
        text   : Text to display
        points : List of hand landmark pixel coordinates
    """

    # Find the top-left position of the hand
    x_min = min(x for x, y in points)
    y_min = min(y for x, y in points)


    # Add a margin around the hand
    margin = 20

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)


    # Display the text slightly above the hand
    cv2.putText(
        frame,
        text,
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,                # Font size
        (0, 0, 255),        # Red color (BGR)
        2,                  # Text thickness
        cv2.LINE_AA         # Anti-aliased text
    )

