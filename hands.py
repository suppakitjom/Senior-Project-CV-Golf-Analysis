import mediapipe as mp
import cv2

latest_gesture_text = {"Left": "None", "Right": "None"}
hand_positions = {"Left": None, "Right": None}

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
RunningMode = mp.tasks.vision.RunningMode


# Create a gesture recognizer instance with the live stream mode:
def print_result(
    result: GestureRecognizerResult,  # type: ignore
    output_image: mp.Image,
    timestamp_ms: int,
):
    global latest_gesture_text
    global hand_positions
    latest_gesture_text = {"Left": "None", "Right": "None"}
    hand_positions = {"Left": None, "Right": None}
    for i in range(min(len(result.gestures), len(result.handedness))):
        label = result.handedness[i][0].category_name
        hand_label = "Right" if label == "Left" else "Left"
        if result.gestures[i] and len(result.gestures[i]) > 0:
            latest_gesture_text[hand_label] = result.gestures[i][0].category_name

        if result.hand_landmarks and len(result.hand_landmarks) > i:
            landmarks = result.hand_landmarks[i]
            avg_x = sum([lm.x for lm in landmarks]) / len(landmarks)
            avg_y = sum([lm.y for lm in landmarks]) / len(landmarks)
            hand_positions[hand_label] = (avg_x, avg_y)
    # print(output_image.numpy_view().shape)
    # print(result)
    # print()


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2,
)
with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    prev_timestamp = 0

    while cap.isOpened():
        success, frame = cap.read()
        # flip
        frame = cv2.flip(frame, 1)
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the current timestamp in milliseconds from the capture.
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp <= prev_timestamp:
            timestamp = prev_timestamp + 1
        prev_timestamp = timestamp

        # Create an mp.Image from the RGB frame. Adjust if necessary based on your mp.Image API.
        input_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform asynchronous gesture recognition; print_result callback will be called with results.
        recognizer.recognize_async(input_image, timestamp)

        # Overlay the recognized gesture on the frame
        cv2.putText(
            frame,
            "Left: " + latest_gesture_text["Left"],
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            frame,
            "Right: " + latest_gesture_text["Right"],
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3,
        )

        for label, pos in hand_positions.items():
            if pos:
                cx = int(pos[0] * frame.shape[1])
                cy = int(pos[1] * frame.shape[0])
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        # Display the live video stream (optional).
        cv2.imshow("Live Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
