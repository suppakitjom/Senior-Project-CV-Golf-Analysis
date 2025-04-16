import cv2
import torch
import numpy as np
import os
import shutil
from collections import deque
import concurrent.futures
from ultralytics import YOLO

# ---------------------------
# Debug helper
# ---------------------------
DEBUG = False


def debug_print(msg: str) -> None:
    if DEBUG:
        print(msg)


# ---------------------------
# Setup functions
# ---------------------------
def setup_swings_folder(folder: str = "swings") -> None:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def load_model(model_path: str):
    model = YOLO(model_path)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    debug_print(f"Using device: {device}")
    return model, device


def open_video(video_path):
    vdo = cv2.VideoCapture(video_path)
    vdo.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vdo.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not vdo.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    fps = int(vdo.get(cv2.CAP_PROP_FPS)) or 60
    width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")
    return vdo, fps, (width, height)


# ---------------------------
# Drawing functions
# ---------------------------
def draw_keypoints_and_skeleton(frame, keypoints):
    if not DEBUG:
        return
    # Draw keypoints
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    # Define skeleton connections (based on indices)
    skeleton_pairs = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    for partA, partB in skeleton_pairs:
        if partA < len(keypoints) and partB < len(keypoints):
            ptA, ptB = keypoints[partA], keypoints[partB]
            if all(ptA) and all(ptB):
                cv2.line(
                    frame,
                    (int(ptA[0]), int(ptA[1])),
                    (int(ptB[0]), int(ptB[1])),
                    (255, 0, 0),
                    2,
                )


# ---------------------------
# Swing video saving (asynchronous)
# ---------------------------
def save_swing_video(swing_id, frames, fps, frame_size):
    swing_video_path = os.path.join("swings", f"swing_{swing_id:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(swing_video_path, fourcc, fps, frame_size)
    for frame_num, frame in frames:
        out.write(frame)
    out.release()
    first_frame = frames[0][0]
    last_frame = frames[-1][0]
    debug_print(
        f"Saved swing {swing_id}: {len(frames)} frames (from frame {first_frame} to {last_frame}) as {swing_video_path}"
    )


# ---------------------------
# Main processing function
# ---------------------------
def main():
    setup_swings_folder("swings")
    model, device = load_model("YOLO/yolo11n-pose.pt")

    # Set video source. For a live camera feed, set video_path = 0.
    video_path = "test_long2.mov"  # Change to 0 for webcam.
    # video_path = 4
    # Determine if live feed camera is being used
    rotate_live = video_path == 4

    vdo, fps, orig_frame_size = open_video(video_path)
    original_width, original_height = orig_frame_size

    # If live feed camera is active, rotate frames and update dimensions accordingly.
    if rotate_live:
        # After rotation, width becomes original height and vice versa.
        original_width, original_height = original_height, original_width
    frame_size = (original_width, original_height)

    # Define a fixed ROI (center of the frame) based on the (possibly rotated) dimensions.
    roi_x1 = int(original_width * 0.25)
    roi_y1 = int(original_height * 0.25)
    roi_x2 = int(original_width * 0.75)
    roi_y2 = int(original_height * 0.75)
    roi_presence_threshold = int(fps * 3.0)
    person_in_roi_counter = 0

    # Swing detection and state-machine parameters
    SWING_VELOCITY_THRESHOLD = 350.0
    pre_motion_buffer = deque(maxlen=int(fps * 2))  # 2 seconds buffer
    state = "idle"
    motion_frames = []
    post_motion_counter = 0
    post_motion_frame_count = int(fps * 1.5)
    cooldown_duration = int(fps * 5)
    cooldown_counter = 0
    swing_threshold = int(fps * 1.2)
    swing_count = 0

    # ThreadPool for asynchronous saving
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    prev_left_wrist, prev_right_wrist = None, None
    frame_count = 0
    display_scale = 1

    while True:
        ret, frame = vdo.read()
        if not ret:
            if (
                state in ["motion", "post_motion"]
                and len(motion_frames) >= swing_threshold
            ):
                swing_count += 1
                executor.submit(
                    save_swing_video, swing_count, motion_frames.copy(), fps, frame_size
                )
            debug_print("End of video reached.")
            break

        # If using live feed, rotate the frame 90° clockwise.
        if rotate_live:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_count += 1
        original_frame = frame.copy()
        pre_motion_buffer.append((frame_count, original_frame))
        max_left_velocity, max_right_velocity = 0.0, 0.0

        # Convert BGR to RGB for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, device=device, verbose=False)
        swing_detected = False
        person_in_roi_this_frame = False

        # Process all detection results
        for result in results:
            if not hasattr(result, "keypoints") or result.keypoints is None:
                continue
            keypoints_arr = result.keypoints.xy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy() if hasattr(result, "boxes") else []
            if keypoints_arr.size == 0:
                continue

            for i, keypoints in enumerate(keypoints_arr):
                if scores.size > 0 and scores[i] < 0.5:
                    continue
                if keypoints.shape[0] < 17:
                    debug_print(
                        f"Frame {frame_count}: Skipping keypoints with shape: {keypoints.shape}"
                    )
                    continue

                draw_keypoints_and_skeleton(frame, keypoints)
                center_x, center_y = np.mean(keypoints[:, 0]), np.mean(keypoints[:, 1])
                if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                    person_in_roi_this_frame = True

                left_wrist, right_wrist = keypoints[9], keypoints[10]
                if prev_left_wrist is not None and prev_right_wrist is not None:
                    left_velocity = np.linalg.norm(
                        np.array(left_wrist) - np.array(prev_left_wrist)
                    )
                    right_velocity = np.linalg.norm(
                        np.array(right_wrist) - np.array(prev_right_wrist)
                    )
                    max_left_velocity = max(max_left_velocity, left_velocity)
                    max_right_velocity = max(max_right_velocity, right_velocity)
                    if (
                        left_velocity > SWING_VELOCITY_THRESHOLD
                        or right_velocity > SWING_VELOCITY_THRESHOLD
                    ):
                        swing_detected = True
                        debug_print(
                            f"Frame {frame_count}: Swing detected! Left: {left_velocity:.2f}, Right: {right_velocity:.2f}"
                        )
                prev_left_wrist, prev_right_wrist = left_wrist, right_wrist

        # Update ROI presence counter
        if person_in_roi_this_frame:
            person_in_roi_counter += 1
        else:
            person_in_roi_counter = 0
        roi_active = person_in_roi_counter >= roi_presence_threshold

        # Draw ROI rectangle
        roi_color = (0, 255, 0) if roi_active else (0, 0, 255)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)
        status_text = "ROI Active" if roi_active else "ROI Inactive"
        cv2.putText(
            frame,
            status_text,
            (roi_x1, roi_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            roi_color,
            2,
        )

        # --- State Machine for Swing Detection & Saving ---
        if roi_active:
            if swing_detected:
                if state == "idle":
                    state = "motion"
                    # Start by using pre-motion frames to capture the swing buildup
                    motion_frames = list(pre_motion_buffer)
                    motion_frames.append((frame_count, original_frame))
                    debug_print(
                        f"Frame {frame_count}: State Transition: idle -> motion"
                    )
                elif state in ["motion", "post_motion"]:
                    motion_frames.append((frame_count, original_frame))
                    if state == "post_motion":
                        state = "motion"
                        post_motion_counter = 0
                        debug_print(
                            f"Frame {frame_count}: State Transition: post_motion -> motion"
                        )
            else:
                if state == "motion":
                    state = "post_motion"
                    post_motion_counter = 0
                    motion_frames.append((frame_count, original_frame))
                    debug_print(
                        f"Frame {frame_count}: State Transition: motion -> post_motion"
                    )
                elif state == "post_motion":
                    motion_frames.append((frame_count, original_frame))
                    post_motion_counter += 1
                    if post_motion_counter >= post_motion_frame_count:
                        if len(motion_frames) >= swing_threshold:
                            swing_count += 1
                            executor.submit(
                                save_swing_video,
                                swing_count,
                                motion_frames.copy(),
                                fps,
                                frame_size,
                            )
                            debug_print(
                                f"Frame {frame_count}: Swing saved (frames: {len(motion_frames)})"
                            )
                            state = "cooldown"
                            cooldown_counter = 0
                        else:
                            debug_print(
                                f"Frame {frame_count}: Swing aborted (insufficient frames)."
                            )
                            state = "idle"
                            motion_frames = []
                            pre_motion_buffer.clear()
                elif state == "cooldown":
                    cooldown_counter += 1
                    if cooldown_counter >= cooldown_duration:
                        state = "idle"
                        motion_frames = []
                        pre_motion_buffer.clear()
                        debug_print(
                            f"Frame {frame_count}: Cooldown complete. Returning to idle."
                        )
        else:
            state = "idle"
            motion_frames = []
            pre_motion_buffer.clear()

        # --- Display output ---
        frame_display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
        cv2.rectangle(frame_display, (10, 10), (240, 90), (50, 50, 50), -1)
        cv2.putText(
            frame_display,
            f"State: {state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_display,
            f"Frame: {frame_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_display,
            f"Swings: {swing_count}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_display,
            f"Left: {int(max_left_velocity)}",
            (frame_display.shape[1] - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_display,
            f"Right: {int(max_right_velocity)}",
            (frame_display.shape[1] - 150, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("YOLO-Pose Golf Swing Detection", frame_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vdo.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=True)
    debug_print(f"\nTotal swings saved: {swing_count}")


if __name__ == "__main__":
    main()
