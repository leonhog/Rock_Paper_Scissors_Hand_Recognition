import cv2
import mediapipe as mp

import drawing_function
import hand_function
import model_function


# ============================================================
# MEDIAPIPE HAND LANDMARKER CONFIGURATION
# ============================================================

# Shortcuts to the MediaPipe classes used in the program
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Configure the hand detection model
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        # Path to the MediaPipe hand landmarker model
        model_asset_path="hand_landmarker.task"
    ),

    # Process the video frame by frame
    running_mode=VisionRunningMode.IMAGE,

    # Detect up to two hands
    num_hands=2,

    # Confidence thresholds for hand detection
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# Create the hand detector
detector = HandLandmarker.create_from_options(options)


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

# Open the default camera
cap = cv2.VideoCapture(0)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # 1. Read a frame from the camera
    # --------------------------------------------------------

    ret, frame = cap.read()

    # Stop the program if the camera cannot provide a frame
    if not ret:
        print("Unable to read the video stream.")
        break


    # Get the dimensions of the current frame
    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # 2. Convert BGR to RGB
    # --------------------------------------------------------

    # OpenCV uses the BGR color format,
    # while MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Create a MediaPipe image from the RGB frame
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # 3. Detect hands
    # --------------------------------------------------------

    result = detector.detect(mp_image)


    # --------------------------------------------------------
    # 4. Process detected hands
    # --------------------------------------------------------

    if result.hand_landmarks:

        # Lists used to store information about each hand
        points = []
        predictions = []


        # Process each detected hand
        for hand_landmarks in result.hand_landmarks:

            # Convert the hand landmark coordinates
            # into pixel coordinates
            point = hand_function.compute_pixel_point(
                hand_landmarks,
                width,
                height
            )

            points.append(point)


            # Convert the landmarks into the format
            # expected by the machine learning model
            value = hand_function.create_value(hand_landmarks)


            # Predict the Rock-Paper-Scissors gesture
            # Possible results:
            #   "rock"
            #   "paper"
            #   "scissors"
            prediction = model_function.predict_rps(value)

            predictions.append(prediction)


            # Display the prediction on the video frame
            drawing_function.draw_text(
                frame,
                prediction,
                point
            )


        # ----------------------------------------------------
        # 5. Determine the winner
        # ----------------------------------------------------

        if predictions:

            # ------------------------------------------------
            # Case 1: Only one hand is detected
            #
            # Case 2: Two hands are detected but they show
            #         the same gesture.
            #
            # In both cases, there is no winner.
            # ------------------------------------------------
            if len(predictions) == 1 or predictions[0] == predictions[1]:

                # Blue means no winner / draw
                colors = [(255, 0, 0)] * len(predictions)


            # ------------------------------------------------
            # Case 3: Two different gestures are detected
            # ------------------------------------------------
            else:

                # Each tuple represents a winning combination
                # where the first player beats the second player.
                winning_combinations = {
                    ("rock", "scissors"),
                    ("scissors", "paper"),
                    ("paper", "rock"),
                }


                # Check whether the first hand wins
                first_hand_wins = (
                    predictions[0],
                    predictions[1]
                ) in winning_combinations


                if first_hand_wins:

                    # Green = winner
                    # Red   = loser
                    colors = [
                        (0, 255, 0),
                        (0, 0, 255)
                    ]

                else:

                    # The second hand wins
                    colors = [
                        (0, 0, 255),
                        (0, 255, 0)
                    ]


            # ------------------------------------------------
            # 6. Draw a colored box around each hand
            # ------------------------------------------------

            for point, color in zip(points, colors):

                drawing_function.draw_box_hand(
                    frame,
                    point,
                    width,
                    height,
                    color
                )


    # --------------------------------------------------------
    # 7. Display the processed frame
    # --------------------------------------------------------

    cv2.imshow("iPhone", frame)


    # --------------------------------------------------------
    # 8. Press "q" to quit the program
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

# Release the camera
cap.release()

# Close the MediaPipe hand detector
detector.close()

# Close all OpenCV windows
cv2.destroyAllWindows()

