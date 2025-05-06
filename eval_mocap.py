import math
import matplotlib.pyplot as plt
import os
import pandas as pd

# Cache for marker YZ data to avoid repeated Excel loads
_marker_yz_cache = {}


def _load_marker_yz_data(filename):
    """
    Load and cache all marker YZ data from the Excel file.
    Returns a dict mapping marker names to an (N_frames x 2) numpy array of [Y, Z].
    """
    if filename not in _marker_yz_cache:
        df_raw = pd.read_excel(filename, header=1).iloc[:, 1:]
        is_bone = df_raw.iloc[0] == "Bone Marker"
        is_pos = df_raw.iloc[3] == "Position"
        df_clean = df_raw.loc[:, is_bone & is_pos]
        df_clean.columns = (
            df_clean.iloc[1].str.split(":").str[1] + "." + df_clean.iloc[4].astype(str)
        )
        df = df_clean.iloc[5:].reset_index(drop=True).astype(float)
        markers = {}
        for m in set(c.split(".")[0] for c in df.columns):
            y_col = f"{m}.Y"
            z_col = f"{m}.Z"
            if y_col in df.columns and z_col in df.columns:
                markers[m] = df[[y_col, z_col]].to_numpy()
        _marker_yz_cache[filename] = markers
    return _marker_yz_cache[filename]


def get_marker_xy(filename, marker_name, frame):
    """
    Load the .xlsx and return the (X, Y) of a marker at a given frame.
    """
    import pandas as pd

    df_raw = pd.read_excel(filename, header=1).iloc[:, 1:]
    is_bone = df_raw.iloc[0] == "Bone Marker"
    is_pos = df_raw.iloc[3] == "Position"
    df_clean = df_raw.loc[:, is_bone & is_pos]
    df_clean.columns = (
        df_clean.iloc[1].str.split(":").str[1] + "." + df_clean.iloc[4].astype(str)
    )
    df = df_clean.iloc[5:].reset_index(drop=True).astype(float)

    x_col = f"{marker_name}.X"
    y_col = f"{marker_name}.Y"
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Marker '{marker_name}' not found in {filename}.")
    if not (0 <= frame < len(df)):
        raise IndexError(f"Frame {frame} out of range (0–{len(df) - 1}).")

    return df.at[frame, x_col], df.at[frame, y_col]


def get_marker_yz(filename, marker_name, frame):
    """
    Return the (Y, Z) of a marker at a given frame, using cached data.
    """
    markers = _load_marker_yz_data(filename)
    if marker_name not in markers:
        raise ValueError(f"Marker '{marker_name}' not found in {filename}.")
    data = markers[marker_name]
    if not (0 <= frame < len(data)):
        raise IndexError(f"Frame {frame} out of range (0–{len(data) - 1}).")
    y, z = data[frame]
    return y, z


def pose_analysis_mocap(position: str, filename: str, frame: int, prev_frame=None):
    fb = []

    def calculate_angle(A, B, C):
        """Return angle ABC in degrees using the law of cosines."""
        AB = math.dist(A, B)
        BC = math.dist(B, C)
        AC = math.dist(A, C)
        return math.degrees(math.acos((AB**2 + BC**2 - AC**2) / (2 * AB * BC)))

    def midpoint(P, Q):
        """Return the midpoint between two 2D points P and Q."""
        return ((P[0] + Q[0]) / 2, (P[1] + Q[1]) / 2)

    # — fetch current markers —
    head_L = get_marker_yz(filename, "LFHD", frame)
    head_R = get_marker_yz(filename, "RFHD", frame)
    head_mid = midpoint(head_L, head_R)

    hip_L = get_marker_yz(filename, "LASI", frame)
    hip_R = get_marker_yz(filename, "RASI", frame)
    hip_mid = midpoint(hip_L, hip_R)

    SHO_L = get_marker_yz(filename, "LSHO", frame)
    SHO_R = get_marker_yz(filename, "RSHO", frame)
    ELB_L = get_marker_yz(filename, "LELB", frame)
    ELB_R = get_marker_yz(filename, "RELB", frame)
    WR_L = midpoint(
        get_marker_yz(filename, "LWRA", frame), get_marker_yz(filename, "LWRB", frame)
    )
    WR_R = midpoint(
        get_marker_yz(filename, "RWRA", frame), get_marker_yz(filename, "RWRB", frame)
    )

    ANK_L = get_marker_yz(filename, "LANK", frame)
    ANK_R = get_marker_yz(filename, "RANK", frame)
    KNE_R = get_marker_yz(filename, "RKNE", frame)

    # — ADDRESS posture checks —
    if position == "Address":
        # 1) spine vs vertical
        # compare head→hip_mid to a vertical line at hip_mid
        spine_ang = calculate_angle(head_mid, hip_mid, (hip_mid[0], hip_mid[1] + 1))
        fb.append(
            f"Spine angle: {spine_ang:.1f}° → "
            + ("upright." if abs(spine_ang) <= 15 else "Too tilted.")
        )

        # 2) stance width vs shoulder width
        feet_w = math.dist(ANK_L, ANK_R)
        sh_w = math.dist(SHO_L, SHO_R)
        fb.append(
            f"Stance is {feet_w - sh_w:.1f} wider than shoulders → "
            + ("stable." if feet_w >= sh_w else "widen stance.")
        )

        # 3) left‐arm straightness
        left_arm_ang = calculate_angle(SHO_L, ELB_L, WR_L)
        fb.append(
            f"Left arm: {left_arm_ang:.1f}° → "
            + ("straight." if left_arm_ang >= 140 else "straighten more.")
        )

    # — TOE-UP, MID-BACKSWING, MID-DOWNSWING, IMPACT share left‑arm check —
    elif position in {
        "Toe-up",
        "Mid-backswing (arm parallel)",
        "Mid-downswing (arm parallel)",
        "Impact",
    }:
        ang = calculate_angle(SHO_L, ELB_L, WR_L)
        fb.append(
            f"Left arm: {ang:.1f}° → "
            + ("straight." if ang >= 140 else "straighten more.")
        )
        # Impact-specific hip alignment
        if position == "Impact":
            left_hip = get_marker_yz(filename, "LASI", frame)
            left_ankle = get_marker_yz(filename, "LANK", frame)
            hip_thresh = 25
            if abs(left_hip[0] - left_ankle[0]) > hip_thresh:
                fb.append(
                    "Left hip is not well aligned over the left foot. Focus on weight transfer."
                )
            else:
                fb.append("Left hip is well positioned over the left foot.")

    # — TOP posture —
    elif position == "Top":
        # left arm
        la = calculate_angle(SHO_L, ELB_L, WR_L)
        fb.append(
            f"Left arm: {la:.1f}° → "
            + ("straight." if la >= 135 else "straighten more.")
        )
        # right knee alignment
        offset = ANK_R[0] - KNE_R[0]
        knee_thresh = -25
        if offset > knee_thresh:
            fb.append("Right knee is shifted too far forward over the right foot.")
        else:
            fb.append("Right knee is well positioned relative to the right foot.")
        # spine angle vs vertical
        spine_ang = calculate_angle(head_mid, hip_mid, (hip_mid[0], hip_mid[1] + 1))
        tilt_direction = "right" if head_mid[0] > hip_mid[0] else "left"
        if spine_ang < 75 or spine_ang > 115:
            fb.append(
                f"Reverse spine angle detected at Top (spine angle: {spine_ang:.1f}°). It appears you're tilting to the {tilt_direction}. Work on keeping your spine aligned."
            )
        else:
            fb.append("Spine angle is within an acceptable range.")

    # — MID-FOLLOW-THROUGH —
    elif position == "Mid-follow-through (shaft parallel)":
        la = calculate_angle(SHO_L, ELB_L, WR_L)
        is_left_straight = la >= 140
        fb.append(
            f"Left arm: {la:.1f}° → "
            + ("straight." if is_left_straight else "straighten more.")
        )
        ra = calculate_angle(SHO_R, ELB_R, WR_R)
        is_right_straight = ra >= 150
        fb.append(
            f"Right arm: {ra:.1f}° → "
            + ("straight." if is_right_straight else "straighten more.")
        )

    # — FINISH posture —
    elif position == "Finish":
        # elbow vs shoulder height
        if ELB_L[1] >= SHO_L[1]:
            fb.append("Elbow at/above shoulder → great finish.")
        else:
            fb.append("Elbow below shoulder → lift elbow in finish.")

    # — head movement since prev_frame? —
    if prev_frame is not None:
        prev_head_L = get_marker_yz(filename, "LFHD", prev_frame)
        prev_head_R = get_marker_yz(filename, "RFHD", prev_frame)
        prev_mid = midpoint(prev_head_L, prev_head_R)
        move_dist = math.dist(head_mid, prev_mid)
        if move_dist > 20:
            fb.append(f"Head moved {move_dist:.1f} units since frame {prev_frame}.")

    return fb


def get_feedback_mocap(mocap_file, frames, visualize=False):
    positions = [
        "Address",
        "Toe-up",
        "Mid-backswing (arm parallel)",
        "Top",
        "Mid-downswing (arm parallel)",
        "Impact",
        "Mid-follow-through (shaft parallel)",
        "Finish",
    ]
    key_events = dict(zip(positions, frames))
    swing_feedbacks = {}
    prev_frame = None
    i = 0
    # Track initial head midpoint for backswing reference
    initial_head_mid = None
    for pos, frame in key_events.items():
        # 1) run your marker‐based analysis
        fb = pose_analysis_mocap(
            position=pos, filename=mocap_file, frame=frame, prev_frame=prev_frame
        )
        # For head movement tracking during backswing
        head_L = get_marker_yz(mocap_file, "LFHD", frame)
        head_R = get_marker_yz(mocap_file, "RFHD", frame)
        head_mid = ((head_L[0] + head_R[0]) / 2, (head_L[1] + head_R[1]) / 2)
        # Track head movement relative to address up through Impact
        if pos == "Address":
            initial_head_mid = head_mid
        elif (
            pos
            in {
                "Toe-up",
                "Mid-backswing (arm parallel)",
                "Top",
                "Mid-downswing (arm parallel)",
                "Impact",
            }
            and initial_head_mid is not None
        ):
            movement = math.dist(head_mid, initial_head_mid)
            fb.append(
                f"Head movement: {movement:.1f} units from address → "
                + (
                    "stable."
                    if movement <= 15
                    else "excessive movement; keep head steady."
                )
            )
        print(f"{pos} feedback at frame {frame}:")
        for f in fb:
            print(f"  - {f}")
        print()
        if fb:
            swing_feedbacks[pos] = fb
        prev_frame = frame

        if visualize:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            # Full-body skeleton visualization in the YZ plane
            # Fetch marker YZ coordinates
            head_L = get_marker_yz(mocap_file, "LFHD", frame)
            head_R = get_marker_yz(mocap_file, "RFHD", frame)
            SHO_L = get_marker_yz(mocap_file, "LSHO", frame)
            SHO_R = get_marker_yz(mocap_file, "RSHO", frame)
            ELB_L = get_marker_yz(mocap_file, "LELB", frame)
            ELB_R = get_marker_yz(mocap_file, "RELB", frame)
            WRA_L = get_marker_yz(mocap_file, "LWRA", frame)
            WRB_L = get_marker_yz(mocap_file, "LWRB", frame)
            WRA_R = get_marker_yz(mocap_file, "RWRA", frame)
            WRB_R = get_marker_yz(mocap_file, "RWRB", frame)
            HIP_L = get_marker_yz(mocap_file, "LASI", frame)
            HIP_R = get_marker_yz(mocap_file, "RASI", frame)
            KNE_L = get_marker_yz(mocap_file, "LKNE", frame)
            KNE_R = get_marker_yz(mocap_file, "RKNE", frame)
            ANK_L = get_marker_yz(mocap_file, "LANK", frame)
            ANK_R = get_marker_yz(mocap_file, "RANK", frame)

            # Compute midpoints
            head_mid = ((head_L[0] + head_R[0]) / 2, (head_L[1] + head_R[1]) / 2)
            shoulder_mid = ((SHO_L[0] + SHO_R[0]) / 2, (SHO_L[1] + SHO_R[1]) / 2)
            hip_mid = ((HIP_L[0] + HIP_R[0]) / 2, (HIP_L[1] + HIP_R[1]) / 2)
            WR_L = ((WRA_L[0] + WRB_L[0]) / 2, (WRA_L[1] + WRB_L[1]) / 2)
            WR_R = ((WRA_R[0] + WRB_R[0]) / 2, (WRA_R[1] + WRB_R[1]) / 2)

            # Assemble markers
            markers = {
                "Head_mid": head_mid,
                "Shoulder_mid": shoulder_mid,
                "Hip_mid": hip_mid,
                "Head_L": head_L,
                "Head_R": head_R,
                "Shoulder_L": SHO_L,
                "Shoulder_R": SHO_R,
                "Elbow_L": ELB_L,
                "Elbow_R": ELB_R,
                "Wrist_L": WR_L,
                "Wrist_R": WR_R,
                "Hip_L": HIP_L,
                "Hip_R": HIP_R,
                "Knee_L": KNE_L,
                "Knee_R": KNE_R,
                "Ankle_L": ANK_L,
                "Ankle_R": ANK_R,
            }

            # Define skeleton connections
            connections = [
                ("Head_mid", "Shoulder_mid"),
                ("Shoulder_mid", "Hip_mid"),
                ("Shoulder_L", "Elbow_L"),
                ("Elbow_L", "Wrist_L"),
                ("Shoulder_R", "Elbow_R"),
                ("Elbow_R", "Wrist_R"),
                ("Hip_L", "Knee_L"),
                ("Knee_L", "Ankle_L"),
                ("Hip_R", "Knee_R"),
                ("Knee_R", "Ankle_R"),
            ]

            fig, ax = plt.subplots()
            ax.axis("off")
            ax.set_aspect("equal", "box")
            ax.invert_xaxis()

            # Define colors for each body segment
            segment_colors = {
                "Head_mid": "red",
                "Shoulder_mid": "red",
                "Hip_mid": "red",
                "Head_L": "orange",
                "Head_R": "orange",
                "Shoulder_L": "blue",
                "Elbow_L": "blue",
                "Wrist_L": "blue",
                "Shoulder_R": "green",
                "Elbow_R": "green",
                "Wrist_R": "green",
                "Hip_L": "purple",
                "Knee_L": "purple",
                "Ankle_L": "purple",
                "Hip_R": "brown",
                "Knee_R": "brown",
                "Ankle_R": "brown",
            }

            # Plot markers
            for name, (y, z) in markers.items():
                color = segment_colors.get(name, "black")
                ax.scatter(y, z, color=color)
                # ax.text(y, z, name)
            # Plot skeleton lines
            for p1, p2 in connections:
                y1, z1 = markers[p1]
                y2, z2 = markers[p2]
                color = segment_colors.get(p1, "black")
                ax.plot([y1, y2], [z1, z2], linewidth=2, color=color)
            ax.set_xlabel("Y")
            ax.set_ylabel("Z")
            ax.set_title(f"{pos} Full-body YZ skeleton at frame {frame}")
            fig.savefig(
                os.path.join(output_dir, f"{i}_{pos}_frame_{frame}.png"), dpi=200
            )
            i += 1
            plt.close(fig)

    return swing_feedbacks


if __name__ == "__main__":
    jom_swing_1_mocap = [0, 263, 283, 309, 322, 336, 345, 425]
    jom_swing_2_mocap = [0, 215, 239, 271, 295, 310, 320, 351]
    get_feedback_mocap(
        "testswings/jom2.xlsx",
        jom_swing_2_mocap,
        visualize=True,
    )
