import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode

hand_positions = {"Left": None, "Right": None}


def handle_result(
    result: PoseLandmarkerResult,  # type: ignore
    output_image: mp.Image,
    timestamp_ms: int,
):
    # print("pose landmarker result: {}".format(result))
    global hand_positions
    hand_positions = {"Left": None, "Right": None}
    if result.pose_landmarks:
        left_x, left_y = (
            result.pose_landmarks[0][19].x,
            result.pose_landmarks[0][19].y,
        )
        right_x, right_y = (
            result.pose_landmarks[0][20].x,
            result.pose_landmarks[0][20].y,
        )

        if 0 <= left_x <= 1 and 0 <= left_y <= 1:
            hand_positions["Left"] = (left_x, left_y)
        if 0 <= right_x <= 1 and 0 <= right_y <= 1:
            hand_positions["Right"] = (right_x, right_y)

        # print(
        #     "Left hand position: ({}, {}), Right hand position: ({}, {})".format(
        #         hand_positions["Left"][0] if hand_positions["Left"] else None,
        #         hand_positions["Left"][1] if hand_positions["Left"] else None,
        #         hand_positions["Right"][0] if hand_positions["Right"] else None,
        #         hand_positions["Right"][1] if hand_positions["Right"] else None,
        #     )
        # )


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=handle_result,
)
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./test_long2.mov")
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

        # Perform asynchronous pose recognition; print_result callback will be called with results.
        landmarker.detect_async(input_image, timestamp)

        for label, pos in hand_positions.items():
            if pos:
                cx = int(pos[0] * frame.shape[1])
                cy = int(pos[1] * frame.shape[0])
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        # Display the live video stream (optional).
        cv2.imshow("Live Pose Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
