import argparse
import os
import subprocess
import pandas as pd


CLIPS_DIR = "auto_dataset/clips"


def run_ffmpeg_cut(video_path, start, duration, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-t",
        str(duration),
        "-c",
        "copy",
        output_path,
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    args = parser.parse_args()

    os.makedirs(CLIPS_DIR, exist_ok=True)

    df = pd.read_csv(args.metadata)

    saved_count = 0

    for _, row in df.iterrows():
        clip_id = row["clip_id"]
        start = float(row["start"])
        duration = float(row["duration"])

        output_path = os.path.join(CLIPS_DIR, f"{clip_id}.mp4")

        try:
            run_ffmpeg_cut(
                video_path=args.video,
                start=start,
                duration=duration,
                output_path=output_path
            )

            saved_count += 1
            print(f"[✓] Clip kaydedildi: {output_path}")

        except subprocess.CalledProcessError:
            print(f"[X] Clip kesilemedi: {clip_id}")

    print(f"\n[✓] Toplam kaydedilen clip: {saved_count}")


if __name__ == "__main__":
    main()