import cv2
import platform

# On macOS, use AVFoundation backend for better FPS control
if platform.system() == "Darwin":
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(0)

# Select MJPG codec (improves FPS reliability) and set desired frame rate
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FPS, 60)  # change 60 to your target FPS

# Set the resolution (width x height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Verify the settings (optional)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Resolution: {}x{}, FPS: {}".format(width, height, fps))

# Don't forget to release the camera when done
cap.release()
