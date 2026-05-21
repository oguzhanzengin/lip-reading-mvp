import argparse
import os
from yt_dlp import YoutubeDL


VIDEOS_DIR = "auto_dataset/videos"
SUBTITLES_DIR = "auto_dataset/subtitles"


def download_video(url: str):
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(SUBTITLES_DIR, exist_ok=True)

    ydl_opts = {
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": f"{VIDEOS_DIR}/%(id)s.%(ext)s",

        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["tr", "tr-TR"],
        "subtitlesformat": "vtt",

        "skip_download": False,

        "postprocessors": [
            {
                "key": "FFmpegSubtitlesConvertor",
                "format": "vtt",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info.get("id")
    title = info.get("title")

    print("[✓] Video indirildi")
    print(f"ID: {video_id}")
    print(f"Title: {title}")
    print(f"Video klasörü: {VIDEOS_DIR}")
    print(f"Altyazı varsa: {VIDEOS_DIR}")

    return video_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube video URL")
    args = parser.parse_args()

    download_video(args.url)


if __name__ == "__main__":
    main()