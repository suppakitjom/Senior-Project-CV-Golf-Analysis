import argparse
from feedback import get_feedback_toong
from eval_mocap import get_feedback_mocap


def compare_feedback(feedback_toong, feedback_mocap):
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
    for pos in positions:  # write to file
        print(f"--- {pos} ---")
        print("Video:")
        for line in feedback_toong.get(pos, []):
            print("  •", line)
        print("Motion Capture:")
        for line in feedback_mocap.get(pos, []):
            print("  •", line)
        print()

    with open("feedback_comparison.txt", "w") as f:
        for pos in positions:
            f.write(f"--- {pos} ---\n")
            f.write("Video:\n")
            for line in feedback_toong.get(pos, []):
                f.write(f"  • {line}\n")
            f.write("Motion Capture:\n")
            for line in feedback_mocap.get(pos, []):
                f.write(f"  • {line}\n")
            f.write("\n")


def main():
    # parser = argparse.ArgumentParser(
    #     description="Compare get_feedback_toong vs get_feedback_mocap outputs"
    # )
    # parser.add_argument("--video", required=True, help="path to swing video")
    # parser.add_argument("--mocap", required=True, help="path to mocap .xlsx")
    # parser.add_argument(
    #     "--video-frames",
    #     nargs=8,
    #     type=int,
    #     required=True,
    #     help="8 key‐event frame indices for video",
    # )
    # parser.add_argument(
    #     "--mocap-frames",
    #     nargs=8,
    #     type=int,
    #     required=True,
    #     help="8 key‐event frame indices for mocap",
    # )
    # args = parser.parse_args()

    video = "./punnvids/swing_002.mp4"
    video_frames = [0, 14, 18, 27, 33, 39, 41, 63]
    toong_fb = get_feedback_toong(video, video_frames, visualize=False)

    mocap = "./testswings/punn2.xlsx"
    mocap_frames = [0, 204, 222, 251, 273, 290, 300, 390]
    mocap_fb = get_feedback_mocap(mocap, mocap_frames, visualize=False)

    compare_feedback(toong_fb, mocap_fb)


if __name__ == "__main__":
    main()
