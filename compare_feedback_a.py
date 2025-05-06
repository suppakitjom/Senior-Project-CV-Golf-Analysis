import os
from feedback import get_feedback_toong
from eval_mocap import get_feedback_mocap

# 8 key‐event frame indices for each of 10 swings, per category
VIDEO_FRAME_SETS = {
    "jom": [
        [0, 42, 47, 53, 60, 66, 70, 90],
        [0, 11, 17, 26, 34, 40, 44, 59],
        [0, 42, 48, 57, 61, 67, 70, 89],
        [0, 41, 48, 56, 61, 67, 71, 88],
        [0, 41, 47, 56, 61, 68, 71, 76],
    ],
    "punn": [
        [0, 12, 17, 25, 32, 35, 41, 65],
        [0, 14, 18, 27, 33, 39, 41, 63],
        [0, 14, 19, 27, 34, 39, 42, 60],
        [0, 12, 16, 26, 32, 37, 40, 64],
        [0, 12, 17, 28, 32, 40, 42, 62],
    ],
}

MOCAP_FRAME_SETS = {
    "jom": [
        [0, 263, 283, 309, 322, 336, 345, 425],
        [0, 215, 239, 271, 295, 310, 320, 351],
        [0, 132, 162, 179, 217, 232, 240, 295],
        [0, 136, 162, 188, 217, 230, 241, 313],
        [0, 219, 241, 272, 290, 306, 319, 383],
    ],
    "punn": [
        [0, 135, 155, 187, 204, 219, 230, 301],
        [0, 204, 222, 251, 273, 290, 300, 390],
        [0, 164, 180, 204, 223, 238, 253, 334],
        [0, 168, 188, 215, 234, 247, 261, 328],
        [0, 218, 248, 278, 292, 307, 320, 392],
    ],
}

POSITIONS = [
    "Address",
    "Toe-up",
    "Mid-backswing (arm parallel)",
    "Top",
    "Mid-downswing (arm parallel)",
    "Impact",
    "Mid-follow-through (shaft parallel)",
    "Finish",
]

VIDEO_DIRS = {
    "punn": "./punnvids",
    "jom": "./jomvids",
}
MOCAP_DIR = "./testswings"
OUT_DIR = "./comparisons"


def compare_and_write(toong_fb, mocap_fb, out_path):
    with open(out_path, "w") as f:
        for pos in POSITIONS:
            f.write(f"--- {pos} ---\n")
            f.write("Video:\n")
            for line in toong_fb.get(pos, []):
                f.write(f"  • {line}\n")
            f.write("Motion Capture:\n")
            for line in mocap_fb.get(pos, []):
                f.write(f"  • {line}\n")
            f.write("\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for category, vdir in VIDEO_DIRS.items():
        vids = sorted([f for f in os.listdir(vdir) if f.endswith(".mp4")])
        max_runs = min(len(vids), len(VIDEO_FRAME_SETS[category]))

        for idx in range(max_runs):
            video_file = vids[idx]
            base = os.path.splitext(video_file)[0]
            video_path = os.path.join(vdir, video_file)

            # derive mocap filename from category and run index (e.g., jom1.xlsx)
            mocap_filename = f"{category}{idx + 1}.xlsx"
            mocap_path = os.path.join(MOCAP_DIR, mocap_filename)
            # fallback to .csv if xlsx not found
            if not os.path.isfile(mocap_path):
                mocap_filename_csv = f"{category}{idx + 1}.csv"
                mocap_path = os.path.join(MOCAP_DIR, mocap_filename_csv)
            if not os.path.isfile(mocap_path):
                print(
                    f"[SKIP] No mocap for {category} run #{idx + 1} ({mocap_filename} or csv)"
                )
                continue

            print(f"→ Processing {category} swing #{idx + 1} ({base})")
            toong_fb = get_feedback_toong(
                video_path, VIDEO_FRAME_SETS[category][idx], visualize=False
            )
            mocap_fb = get_feedback_mocap(
                mocap_path, MOCAP_FRAME_SETS[category][idx], visualize=False
            )

            out_file = os.path.join(OUT_DIR, f"{category}_comparison_{idx + 1:02d}.txt")
            compare_and_write(toong_fb, mocap_fb, out_file)
            print(f"   Saved: {out_file}")


if __name__ == "__main__":
    main()
