import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# from matplotlib.animation import FuncAnimation  # optional, for autoplay
from matplotlib.widgets import Slider

speed_factor = 5

# 1) Load & clean
df_raw = pd.read_excel("testswings/jom1.xlsx", header=1).iloc[:, 1:]
is_bone = df_raw.iloc[0] == "Bone Marker"
is_pos = df_raw.iloc[3] == "Position"
df_clean = df_raw.loc[:, is_bone & is_pos]
df_clean.columns = (
    df_clean.iloc[1].str.split(":").str[1] + "." + df_clean.iloc[4].astype(str)
)
df = df_clean.iloc[5:].reset_index(drop=True).astype(float)

marker_names = sorted({c.split(".")[0] for c in df.columns})
coords = {m: df[[f"{m}.X", f"{m}.Y", f"{m}.Z"]].values for m in marker_names}

# 2) Region & color maps
region_map = {
    # head
    "RFHD": "head",
    "LFHD": "head",
    "RBHD": "head",
    "LBHD": "head",
    # neck
    "C7": "neck",
    "CLAV": "neck",
    # trunk
    # "RBAK": "trunk",
    # "LBAK": "trunk",
    "RFRM": "trunk",
    "LFRM": "trunk",
    # shoulders & arms
    "RSHO": "shoulder",
    "LSHO": "shoulder",
    "RUPA": "upper_arm",
    "LUPA": "upper_arm",
    "RELB": "elbow",
    "LELB": "elbow",
    "RWRA": "wrist",
    "LWRA": "wrist",
    "RWRB": "wrist",
    "LWRB": "wrist",
    # pelvis & legs
    "RASI": "pelvis",
    "LASI": "pelvis",
    "RPSI": "pelvis",
    "LPSI": "pelvis",
    "RTHI": "thigh",
    "LTHI": "thigh",
    "RKNE": "thigh",
    "LKNE": "thigh",
    "RTIB": "shin",
    "LTIB": "shin",
    "RANK": "shin",
    "LANK": "shin",
    "RHEE": "foot",
    "LHEE": "foot",
    "RTOE": "foot",
    "LTOE": "foot",
}
region_colors = {
    "head": "red",
    "neck": "orange",
    "trunk": "green",
    "shoulder": "blue",
    "upper_arm": "purple",
    "elbow": "brown",
    "wrist": "gray",
    "pelvis": "cyan",
    "thigh": "magenta",
    "shin": "olive",
    "foot": "black",
}

# 3) Set up 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Marker Animation")

scatters = {}
for m in marker_names:
    sec = region_map.get(m, None)
    color = region_colors.get(sec, "black")
    scatters[m] = ax.scatter([], [], [], s=20, color=color, label=m)

bones = [
    ("C7", "CLAV"),
    ("CLAV", "RFHD"),
    ("CLAV", "LFHD"),
    ("C7", "RSHO"),
    ("RSHO", "RUPA"),
    ("RUPA", "RELB"),
    ("RELB", "RWRB"),
    ("RSHO", "RFRM"),
    ("RFRM", "RWRB"),
    ("C7", "LSHO"),
    ("LSHO", "LUPA"),
    ("LUPA", "LELB"),
    ("LELB", "LWRB"),
    ("LSHO", "LFRM"),
    ("LFRM", "LWRB"),
    ("RASI", "LASI"),
    ("RASI", "RTHI"),
    ("RTHI", "RKNE"),
    ("RKNE", "RTIB"),
    ("RTIB", "RHEE"),
    ("LASI", "LTHI"),
    ("LTHI", "LKNE"),
    ("LKNE", "LTIB"),
    ("LTIB", "LHEE"),
]
lines = {}
for start, end in bones:
    (line,) = ax.plot([], [], [], linewidth=2, color="black")
    lines[(start, end)] = line

# equal axis limits
x_all = df[[f"{m}.X" for m in marker_names]].values.flatten()
y_all = df[[f"{m}.Y" for m in marker_names]].values.flatten()
z_all = df[[f"{m}.Z" for m in marker_names]].values.flatten()
x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
z_min, z_max = np.nanmin(z_all), np.nanmax(z_all)
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
mid_x = (x_max + x_min) / 2
mid_y = (y_max + y_min) / 2
mid_z = (z_max + z_min) / 2
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.view_init(elev=0, azim=180)  # YZ plane


# 4) Init & update
def init():
    for sc in scatters.values():
        sc._offsets3d = ([], [], [])
    for line in lines.values():
        line.set_data_3d([], [], [])
    return list(scatters.values()) + list(lines.values())


def update(frame):
    t = frame * 0.01
    for m, sc in scatters.items():
        x, y, z = coords[m][frame]
        sc._offsets3d = ([x], [y], [z])
    for (start, end), line in lines.items():
        x1, y1, z1 = coords[start][frame]
        x2, y2, z2 = coords[end][frame]
        line._verts3d = ([x1, x2], [y1, y2], [z1, z2])
    ax.set_title(f"Time = {t:.2f} s")
    return list(scatters.values()) + list(lines.values())


# 5) Slider for manual scrubbing
slider_ax = plt.axes([0.15, 0.02, 0.7, 0.03])
frame_slider = Slider(
    ax=slider_ax, label="Frame", valmin=0, valmax=len(df) - 1, valinit=0, valfmt="%0.0f"
)


def on_slider_change(val):
    update(int(val))
    fig.canvas.draw_idle()


frame_slider.on_changed(on_slider_change)

# draw first frame
init()
update(0)

# 6) (Optional) automatic animation
# ani = FuncAnimation(
#     fig, update,
#     frames=range(0, len(df), speed_factor),
#     init_func=init,
#     interval=10,
#     blit=False,
# )

# plt.tight_layout()
plt.show()
