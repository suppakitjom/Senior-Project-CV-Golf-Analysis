import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Load Video ---
video_path = "jomvids/swing_001.mp4"  # change this
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames = []
success = True
while success:
    success, frame = cap.read()
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
cap.release()

# --- Create Figure ---
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

frame_img = ax.imshow(frames[0])
ax.axis("off")
ax.set_title("Video Frame Viewer")

# --- Slider Setup ---
slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
frame_slider = Slider(
    ax=slider_ax,
    label="Frame",
    valmin=0,
    valmax=len(frames) - 1,
    valinit=0,
    valfmt="%0.0f",
)


def update_frame(val):
    idx = int(val)
    frame_img.set_data(frames[idx])
    fig.canvas.draw_idle()
    ax.set_title(f"Frame {idx} / {len(frames) - 1}")


frame_slider.on_changed(update_frame)

plt.show()
