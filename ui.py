import streamlit as st
import cv2
import torch
import numpy as np
import os
import shutil
from collections import deque
import time
import threading
import sys
import mediapipe as mp

import feedback


# ---------------------------
# Setups
# ---------------------------

latest_gesture_text = {"Left": "None", "Right": "None"}
body_positions = {"Left": None, "Right": None, "Head": None}

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode


def handle_result_gesture(
    result: GestureRecognizerResult,  # type: ignore
    output_image: mp.Image,
    timestamp: int,
):
    global latest_gesture_text
    latest_gesture_text = {"Left": "None", "Right": "None"}

    for i in range(min(len(result.gestures), len(result.handedness))):
        label = result.handedness[i][0].category_name
        hand_label = "Right" if label == "Left" else "Left"
        if result.gestures[i] and len(result.gestures[i]) > 0:
            latest_gesture_text[hand_label] = result.gestures[i][0].category_name


def handle_result_pose(
    result: PoseLandmarkerResult,  # type: ignore
    output_image: mp.Image,
    timestamp_ms: int,
):
    # print("pose landmarker result: {}".format(result))
    global body_positions
    body_positions = {"Left": None, "Right": None, "Head": None}
    width = output_image.width
    height = output_image.height
    if result.pose_landmarks:
        left_x, left_y = (
            result.pose_landmarks[0][19].x,
            result.pose_landmarks[0][19].y,
        )
        right_x, right_y = (
            result.pose_landmarks[0][20].x,
            result.pose_landmarks[0][20].y,
        )
        head_x, head_y = (
            result.pose_landmarks[0][0].x,
            result.pose_landmarks[0][0].y,
        )
        if 0 <= left_x <= 1 and 0 <= left_y <= 1:
            body_positions["Left"] = (left_x * width, left_y * height)
        if 0 <= right_x <= 1 and 0 <= right_y <= 1:
            body_positions["Right"] = (right_x * width, right_y * height)
        if 0 <= head_x <= 1 and 0 <= head_y <= 1:
            body_positions["Head"] = (head_x * width, head_y * height)


gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=handle_result_gesture,
    num_hands=2,
    # min_hand_detection_confidence=0.3,
    # min_hand_presence_confidence=0.3,
    # min_tracking_confidence=0.3,
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=handle_result_pose,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)


def setup_swings_folder(folder: str = "swings") -> None:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def open_video(video_path):
    vdo = cv2.VideoCapture(video_path)
    vdo.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vdo.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not vdo.isOpened():
        st.error("Error: Could not open video source.")
        return None, None, None
    fps = int(vdo.get(cv2.CAP_PROP_FPS))
    width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")
    return vdo, fps, (width, height)


def save_swing_video(swing_id, frames, fps, frame_size):
    swing_video_path = os.path.join("swings", f"swing_{swing_id:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(swing_video_path, fourcc, fps, frame_size)
    for frame_num, frame in frames:
        out.write(frame)
    out.release()
    return swing_video_path


def run_replay(
    latest_swing_path, video_placeholder, text_placeholder, mode_placeholder
):
    """
    Replay the saved swing video in a loop until feedback.feedback finishes.
    Capture its return value and display it on the right panel.
    """
    mode_placeholder.markdown(
        "<h2 style='text-align:center; color:blue;'>▶️ REPLAY & FEEDBACK MODE ACTIVE</h2>",
        unsafe_allow_html=True,
    )
    text_placeholder.write("")
    # Create an event to signal when TTS/feedback is done.
    feedback_stop_event = threading.Event()

    # Use a mutable container to capture the return value.
    tts_feedback = [None]

    def run_feedback():
        # feedback.feedback is expected to return a value.
        tts_feedback[0] = feedback.feedback(latest_swing_path, feedback_stop_event)

    feedback_thread = threading.Thread(target=run_feedback)
    feedback_thread.start()

    # Loop the replay video until the stop event is set.
    while not feedback_stop_event.is_set():
        cap_replay = cv2.VideoCapture(latest_swing_path)
        if not cap_replay.isOpened():
            text_placeholder.write("Error opening replay video.")
            return
        # Replay one cycle.
        while True:
            ret, frame = cap_replay.read()
            if not ret or feedback_stop_event.is_set():
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", width=340)
            time.sleep(1 / 30)

        if feedback_stop_event.is_set():
            text_placeholder.write("### Swing Feedback \n" + tts_feedback[0][1])
        cap_replay.release()
    feedback_thread.join()

    # tts_stop_event = threading.Event()
    # tts_thread = threading.Thread(
    #     target=feedback.synthesize_speech, args=(tts_feedback[0][0], tts_stop_event)
    # )
    # tts_thread.start()
    # # Loop the replay video until the stop event is set.
    # while not tts_stop_event.is_set():
    #     cap_replay = cv2.VideoCapture(latest_swing_path)
    #     if not cap_replay.isOpened():
    #         text_placeholder.write("Error opening replay video.")
    #         return
    #     # Replay one cycle.
    #     while True:
    #         ret, frame = cap_replay.read()
    #         if not ret or tts_stop_event.is_set():
    #             break
    #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         video_placeholder.image(frame_rgb, channels="RGB", width=340)
    #         time.sleep(1 / 30)

    #     if tts_stop_event.is_set():
    #         text_placeholder.write("### Swing Feedback \n" + tts_feedback[0][1])
    #     cap_replay.release()
    # time.sleep(2)


# ---------------------------
# Main processing function (Continuous Detection/Replay)
# ---------------------------
def run_detection(video_source):
    setup_swings_folder("swings")
    mode_placeholder = st.empty()

    # Set the video source.
    video_path = video_source  # Use the provided video source
    rotate_live = video_path == 4

    # Create two columns: left for video, right for text.
    left_column, right_column = st.columns([3, 2])
    video_placeholder = left_column.empty()
    text_placeholder = right_column.empty()

    # Open live video capture.
    vdo, fps, orig_frame_size = open_video(video_path)
    if vdo is None:
        return
    original_width, original_height = orig_frame_size
    if rotate_live:
        original_width, original_height = original_height, original_width
    frame_size = (original_width, original_height)

    # Detection parameters.
    roi_x1 = int(original_width * 0.2)
    roi_y1 = int(original_height * 0.2)
    roi_x2 = int(original_width * 0.8)
    roi_y2 = int(original_height * 0.8)
    roi_presence_threshold = int(fps * 3.0)
    person_in_roi_counter = 0
    SWING_VELOCITY_THRESHOLD = 45.0
    pre_motion_buffer = deque(maxlen=int(fps * 1.5))
    state = "idle"
    motion_frames = []
    post_motion_counter = 0
    post_motion_frame_count = int(fps * 1)
    cooldown_duration = int(fps * 5)
    cooldown_counter = 0
    swing_threshold = int(fps * 1.2)
    swing_count = 0
    frame_count = 0
    prev_left_wrist, prev_right_wrist = None, None
    with GestureRecognizer.create_from_options(gesture_options) as recognizer:
        with PoseLandmarker.create_from_options(pose_options) as landmarker:
            while vdo.isOpened():
                mode_placeholder.markdown(
                    "<h2 style='text-align:center; color:green;'>🔴 DETECTION MODE ACTIVE</h2>",
                    unsafe_allow_html=True,
                )
                ret, frame = vdo.read()
                if not ret:
                    break

                if rotate_live:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                frame_count += 1
                original_frame = frame.copy()
                pre_motion_buffer.append((frame_count, original_frame))
                max_left_velocity, max_right_velocity = 0.0, 0.0

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                swing_detected = False
                person_in_roi_this_frame = False

                input_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                recognizer.recognize_async(input_image, frame_count)
                landmarker.detect_async(input_image, frame_count)

                if body_positions["Left"]:
                    left_x, left_y = body_positions["Left"]
                    if roi_x1 <= left_x <= roi_x2 and roi_y1 <= left_y <= roi_y2:
                        person_in_roi_this_frame = True

                    if prev_left_wrist is not None:
                        left_velocity = np.linalg.norm(
                            np.array(body_positions["Left"]) - np.array(prev_left_wrist)
                        )
                        max_left_velocity = max(max_left_velocity, left_velocity)

                        # if max_left_velocity > SWING_VELOCITY_THRESHOLD:
                        #     swing_detected = True

                    prev_left_wrist = left_x, left_y

                if body_positions["Right"]:
                    right_x, right_y = body_positions["Right"]
                    if roi_x1 <= right_x <= roi_x2 and roi_y1 <= right_y <= roi_y2:
                        person_in_roi_this_frame = True

                    if prev_right_wrist is not None:
                        right_velocity = np.linalg.norm(
                            np.array(body_positions["Right"])
                            - np.array(prev_right_wrist)
                        )
                        max_right_velocity = max(max_right_velocity, right_velocity)

                        # if max_right_velocity > SWING_VELOCITY_THRESHOLD:
                        #     swing_detected = True

                    prev_right_wrist = right_x, right_y

                if body_positions["Head"]:
                    head_x, head_y = body_positions["Head"]
                    if roi_x1 <= head_x <= roi_x2 and roi_y1 <= head_y <= roi_y2:
                        person_in_roi_this_frame = True

                if (max_left_velocity > SWING_VELOCITY_THRESHOLD) and (max_right_velocity > SWING_VELOCITY_THRESHOLD):
                    swing_detected = True
                
                if person_in_roi_this_frame:
                    person_in_roi_counter += 1
                else:
                    person_in_roi_counter = 0
                roi_active = person_in_roi_counter >= roi_presence_threshold

                # Draw ROI and overlay status.
                resize_scale = 0.45
                resized_frame = cv2.resize(
                    frame, (0, 0), fx=resize_scale, fy=resize_scale
                )
                disp_roi_x1 = int(roi_x1 * resize_scale)
                disp_roi_y1 = int(roi_y1 * resize_scale)
                disp_roi_x2 = int(roi_x2 * resize_scale)
                disp_roi_y2 = int(roi_y2 * resize_scale)
                roi_color = (0, 255, 0) if roi_active else (0, 0, 255)
                cv2.rectangle(
                    resized_frame,
                    (disp_roi_x1, disp_roi_y1),
                    (disp_roi_x2, disp_roi_y2),
                    roi_color,
                    2,
                )
                cv2.putText(
                    resized_frame,
                    f"State: {state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    resized_frame,
                    f"Frame: {frame_count}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    resized_frame,
                    f"Swings: {swing_count}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                # display hands speed
                cv2.putText(
                    resized_frame,
                    f"Left Speed: {max_left_velocity:.2f}",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    resized_frame,
                    f"Right Speed: {max_right_velocity:.2f}",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                for label, pos in body_positions.items():
                    if pos:
                        resized_pos = (
                            int(pos[0] * resize_scale),
                            int(pos[1] * resize_scale),
                        )
                        cv2.circle(resized_frame, resized_pos, 10, (0, 0, 255), -1)

                frame_rgb_display = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # Update the video feed in the left column.
                video_placeholder.image(frame_rgb_display, channels="RGB", width=340)
                time.sleep(1 / 60)

                # State machine update.
                if roi_active:
                    if swing_detected:
                        if state == "idle":
                            state = "motion"
                            motion_frames = list(pre_motion_buffer)
                            motion_frames.append((frame_count, original_frame))
                        elif state in ["motion", "post_motion"]:
                            motion_frames.append((frame_count, original_frame))
                            if state == "post_motion":
                                state = "motion"
                    else:
                        if state == "motion":
                            state = "post_motion"
                            post_motion_counter = 0
                            motion_frames.append((frame_count, original_frame))
                        elif state == "post_motion":
                            motion_frames.append((frame_count, original_frame))
                            post_motion_counter += 1
                            if post_motion_counter >= post_motion_frame_count:
                                if len(motion_frames) >= swing_threshold:
                                    swing_count += 1
                                    latest_swing_path = save_swing_video(
                                        swing_count,
                                        motion_frames.copy(),
                                        fps,
                                        frame_size,
                                    )
                                    # Replay the saved swing video until TTS/feedback completes.
                                    run_replay(
                                        latest_swing_path,
                                        video_placeholder,
                                        text_placeholder,
                                        mode_placeholder,
                                    )
                                    # After replay and feedback, reset state and buffers.
                                    state = "idle"
                                    motion_frames = []
                                    pre_motion_buffer.clear()
                                else:
                                    state = "idle"
                                    motion_frames = []
                                    pre_motion_buffer.clear()
                        elif state == "cooldown":
                            cooldown_counter += 1
                            if cooldown_counter >= cooldown_duration:
                                state = "idle"
                                motion_frames = []
                                pre_motion_buffer.clear()
                else:
                    state = "idle"
                    motion_frames = []
                    pre_motion_buffer.clear()

    vdo.release()
    right_column.write(f"Detection ended. Total swings saved: {swing_count}")

if __name__ == "__main__":
    # If "--live" is passed as a command-line argument, use live video capture (device 4), else use test video file.
    testvid = "test_long2.mov"
    video_source = 4 if "live" in sys.argv else testvid
    run_detection(video_source)
