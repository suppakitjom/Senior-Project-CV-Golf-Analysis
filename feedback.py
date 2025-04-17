import math
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import simpleaudio as sa
import torch
import torch.nn.functional as F
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

from config import OPENAI_API_KEY
from eval import Normalize, ToTensor
from model import EventDetector

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
event_names = {
    0: "Address",
    1: "Toe-up",
    2: "Mid-backswing (arm parallel)",
    3: "Top",
    4: "Mid-downswing (arm parallel)",
    5: "Impact",
    6: "Mid-follow-through (shaft parallel)",
    7: "Finish",
}


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        ]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if not ret:
                break
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[0.406 * 255, 0.456 * 255, 0.485 * 255],
            )
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))
        sample = {"images": np.asarray(images), "labels": np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def analyze_video(path, seq_length=64):
    ds = SampleVideo(
        path,
        transform=transforms.Compose(
            [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
    )

    try:
        save_dict = torch.load(
            "./SwingNet Pretrained.tar", weights_only=True, map_location=device
        )
    except FileNotFoundError:
        print(
            "Model weights not found. Download model weights and place in the appropriate folder."
        )
        return None

    model.load_state_dict(save_dict["model_state_dict"])
    model.to(device)
    model.eval()

    for sample in dl:
        images = sample["images"]
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length :, :, :, :]
            else:
                image_batch = images[
                    :, batch * seq_length : (batch + 1) * seq_length, :, :, :
                ]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    confidence = [probs[e, i] for i, e in enumerate(events)]
    result = {
        event_names[i]: {"frame": e, "confidence": np.round(confidence[i], 3)}
        for i, e in enumerate(events)
    }
    return result


def pose_analysis(position, keypoints, previous_keypoints=None):
    feedbacks = []

    def calculate_angle(pointA, pointB, pointC):
        AB = math.sqrt((pointB[0] - pointA[0]) ** 2 + (pointB[1] - pointA[1]) ** 2)
        BC = math.sqrt((pointC[0] - pointB[0]) ** 2 + (pointC[1] - pointB[1]) ** 2)
        AC = math.sqrt((pointC[0] - pointA[0]) ** 2 + (pointC[1] - pointA[1]) ** 2)
        angle = math.degrees(math.acos((AB**2 + BC**2 - AC**2) / (2 * AB * BC)))
        return angle

    if position == "Address":
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[0], keypoints[11], keypoints[12]]
        ):
            nose = keypoints[0]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            spine_angle = calculate_angle(nose, left_hip, right_hip)
            is_spine_correct = 0 <= abs(90 - spine_angle) <= 15
            feedbacks.append(
                f"Spine angle is {spine_angle:.2f}°. {'Great posture!' if is_spine_correct else 'Try to keep a more upright spine.'}"
            )
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[6], keypoints[15], keypoints[16]]
        ):
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            shoulder_width = math.dist(left_shoulder, right_shoulder)
            foot_width = math.dist(left_ankle, right_ankle)
            is_foot_width_correct = foot_width >= shoulder_width
            feedbacks.append(
                f"Foot width is wider than shoulder by {foot_width - shoulder_width:.2f} units. {'Good stance for stability.' if is_foot_width_correct else 'Consider widening your stance.'}"
            )
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[7], keypoints[9]]
        ):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            feedbacks.append(
                f"Left arm angle is {left_arm_angle:.2f}°. {'Nice and straight!' if is_left_arm_straight else 'Try to straighten your left arm more.'}"
            )

    elif position == "Toe-up":
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[7], keypoints[9]]
        ):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            feedbacks.append(
                f"Left arm angle is {left_arm_angle:.2f}°. {'Nice and straight!' if is_left_arm_straight else 'Try to straighten your left arm more.'}"
            )

    elif position == "Mid-backswing (arm parallel)":
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[7], keypoints[9]]
        ):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            feedbacks.append(
                f"Left arm angle is {left_arm_angle:.2f}°. {'Nice and straight!' if is_left_arm_straight else 'Try to straighten your left arm more.'}"
            )

    elif position == "Top":
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[7], keypoints[9]]
        ):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 135 <= left_arm_angle <= 180
            feedbacks.append(
                f"Left arm angle is {left_arm_angle:.2f}°. {'Nice and straight!' if is_left_arm_straight else 'Try to straighten your left arm more.'}"
            )

        if all(k is not None and len(k) > 0 for k in [keypoints[14], keypoints[16]]):
            right_knee = keypoints[14]
            right_ankle = keypoints[16]
            threshold = -25  # Adjust this threshold as needed.
            if right_ankle[0] - right_knee[0] > threshold:
                feedbacks.append(
                    "Right knee is shifted too far forward over the right foot. Try to keep it more aligned."
                )
            else:
                feedbacks.append(
                    "Right knee is well positioned relative to the right foot."
                )

        if all(
            k is not None and len(k) > 0
            for k in [keypoints[0], keypoints[11], keypoints[12]]
        ):
            nose = keypoints[0]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            spine_angle = calculate_angle(nose, left_hip, right_hip)

            # Compute the horizontal midpoint of the hips.
            hip_midpoint_x = (left_hip[0] + right_hip[0]) / 2
            x_offset = nose[0] - hip_midpoint_x

            # Determine tilt direction based on the nose's position relative to the hip midpoint.
            tilt_direction = "right" if x_offset > 0 else "left"

            # Provide feedback that now includes the tilt direction when spine angle is off.
            if spine_angle < 75 or spine_angle > 115:
                feedbacks.append(
                    f"Reverse spine angle detected at Top (spine angle: {spine_angle:.2f}°). It appears you're tilting to the {tilt_direction}. Work on keeping your spine aligned."
                )
            else:
                feedbacks.append("Spine angle is within an acceptable range.")

    elif position == "Mid-downswing (arm parallel)":
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[7], keypoints[9]]
        ):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            feedbacks.append(
                f"Left arm angle is {left_arm_angle:.2f}°. {'Nice and straight!' if is_left_arm_straight else 'Try to straighten your left arm more.'}"
            )

    elif position == "Impact":
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[7], keypoints[9]]
        ):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            feedbacks.append(
                f"Left arm angle is {left_arm_angle:.2f}°. {'Nice and straight!' if is_left_arm_straight else 'Try to straighten your left arm more.'}"
            )

        if all(k is not None and len(k) > 0 for k in [keypoints[11], keypoints[15]]):
            left_hip = keypoints[11]
            left_ankle = keypoints[15]
            threshold = 25  # Adjust threshold as needed.
            if abs(left_hip[0] - left_ankle[0]) > threshold:
                feedbacks.append(
                    "Left hip is not well aligned over the left foot. Focus on weight transfer."
                )
            else:
                feedbacks.append("Left hip is well positioned over the left foot.")

    elif position == "Mid-follow-through (shaft parallel)":
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[5], keypoints[7], keypoints[9]]
        ):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            is_left_arm_straight = 140 <= left_arm_angle <= 180
            feedbacks.append(
                f"Left arm angle is {left_arm_angle:.2f}°. {'Nice and straight!' if is_left_arm_straight else 'Try to straighten your left arm more.'}"
            )
        if all(
            k is not None and len(k) > 0
            for k in [keypoints[6], keypoints[8], keypoints[10]]
        ):
            right_shoulder = keypoints[6]
            right_elbow = keypoints[8]
            right_wrist = keypoints[10]
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            is_right_arm_straight = 150 <= right_arm_angle <= 180
            feedbacks.append(
                f"Right arm angle is {right_arm_angle:.2f}°. {'Nice and straight!' if is_right_arm_straight else 'Try to straighten your right arm more.'}"
            )

    elif position == "Finish":
        if all(k is not None and len(k) > 0 for k in [keypoints[5], keypoints[7]]):
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            is_left_arm_raised = left_elbow[1] - 30 <= left_shoulder[1]
            feedbacks.append(
                f"Left elbow is {'at or above' if is_left_arm_raised else 'below'} shoulder height. "
                f"{'Great finish!' if is_left_arm_raised else 'Try to keep your left elbow raised in your finish.'}"
            )

    # --- Compare with previous keypoints if available ---
    if previous_keypoints:
        # Check for head movements
        if position in ["Mid-follow-through (shaft parallel)", "Finish"]:
            pass
        else:
            last_position = list(previous_keypoints.keys())[-1]
            prev_kpts = previous_keypoints[last_position]
            if (
                keypoints[0] is not None
                and len(keypoints[0]) > 0
                and prev_kpts[0] is not None
                and len(prev_kpts[0]) > 0
            ):
                nose_current = keypoints[0]
                nose_prev = prev_kpts[0]
                movement = math.dist(nose_current, nose_prev)
                if movement > 15:
                    feedbacks.append(
                        f"Head moved {movement:.2f} units since {last_position}."
                    )
    # ---------------------------------------------------------

    print("Position:", position, feedbacks, end="\n\n")
    return feedbacks


def get_feedback(path, visualize=False):
    swing_feedbacks = {}
    model = YOLO("YOLO/yolo11x-pose.pt", verbose=False)

    previous_keypoints = {}

    for key, value in analyze_video(path).items():
        frame = value["frame"]
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = cap.read()
        cap.release()
        results = model(img, verbose=False)
        for result in results:
            current_keypoints = result.keypoints.xy[0].cpu().numpy()
            # Pass previous_keypoints to allow comparisons with earlier positions.
            feedback = pose_analysis(
                key, current_keypoints, previous_keypoints=previous_keypoints
            )
            if feedback:
                swing_feedbacks[key] = feedback
            # Store current keypoints for future comparisons.
            previous_keypoints[key] = current_keypoints
            if visualize:
                annotated_frame = result.plot(
                    labels=False, boxes=False, masks=False, kpt_radius=10
                )
                resize_scale = 0.3
                annotated_frame = cv2.resize(
                    annotated_frame,
                    (
                        int(annotated_frame.shape[1] * resize_scale),
                        int(annotated_frame.shape[0] * resize_scale),
                    ),
                )
                cv2.putText(
                    annotated_frame,
                    f"{key} - Confidence: {value['confidence']:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Annotated Frame", annotated_frame)
                cv2.imwrite("output/" + key + ".jpg", annotated_frame)
                cv2.waitKey(0)
        cv2.destroyAllWindows()
    return swing_feedbacks


def video_loop(video_path, stop_event):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 60
    delay = int(1000 / fps)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow("Video Display", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()


def generate_feedback_script(feedback_data):
    feedback_text = ""
    for position, comments in feedback_data.items():
        feedback_text += f"{position}:\n" + "\n".join(comments) + "\n\n"

    messages = [
        SystemMessage(
            content=(
                "You are an encouraging golf coach giving simple feedback to a beginner. "
                "Make the feedback conversational, avoid using numbers, and keep it very easy to understand. "
                "Keep it very short and to the point, focusing on improvements while also offering praise. "
                "Your output should be in a spoken language style for a voiceover."
            )
        ),
        HumanMessage(
            content=(
                f"Here’s some golf swing feedback. Summarize it in a way that's easy to understand. "
                f"\n\n{feedback_text}"
            )
        ),
    ]

    class Response(BaseModel):
        script: str = Field(
            ..., description="The generated feedback script in a short paragraph."
        )
        summary: str = Field(
            ..., description="A summary of the feedback in the format of bullet points."
        )

    llm = ChatOpenAI(
        model_name="gpt-4.1-nano-2025-04-14", openai_api_key=OPENAI_API_KEY
    ).with_structured_output(Response)
    response = llm.invoke(messages)

    print("Generated Script:\n", response.script)
    return response.script, response.summary


def synthesize_speech(script, stop_event=None):
    client = OpenAI()
    speech_file_path = str(Path(__file__).parent / "speech.wav")

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="ash",
        input=script,
        instructions="Speak like an encouraging golf instructor, offering clear and constructive feedback with a confident and supportive tone.",
        response_format="wav",
    ) as response:
        response.stream_to_file(speech_file_path)

    sa.WaveObject.from_wave_file(speech_file_path).play().wait_done()

    if stop_event:
        stop_event.set()

    # remove the audio file after playback
    if os.path.exists(speech_file_path):
        os.remove(speech_file_path)


def feedback(video_path, stop_event):
    swing_feedbacks = get_feedback(video_path, visualize=False)
    script, summary = generate_feedback_script(swing_feedbacks)
    stop_event.set()
    return script, summary


if __name__ == "__main__":
    video_path = "./swing_001 (1).mp4"
    # Uncomment the following lines to run video display and feedback generation concurrently.
    # stop_event = threading.Event()
    # video_thread = threading.Thread(target=video_loop, args=(video_path, stop_event))
    # feedback_thread = threading.Thread(target=feedback, args=(video_path, stop_event))
    # video_thread.start()
    # feedback_thread.start()
    # feedback_thread.join()
    # video_thread.join()

    # For testing purposes, run get_feedback with visualization enabled.
    # get_feedback(video_path, visualize=True)
    feedback(video_path, stop_event=threading.Event())
