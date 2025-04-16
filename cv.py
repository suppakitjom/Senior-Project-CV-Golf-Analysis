import cv2

# Use the appropriate index for your external camera (0, 1, etc.)
cap = cv2.VideoCapture(4)

# Set the resolution (width x height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set the frames per second (FPS)
cap.set(cv2.CAP_PROP_FPS, 60)

# Verify the settings (optional)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Resolution: {}x{}, FPS: {}".format(width, height, fps))

# Don't forget to release the camera when done
cap.release()
