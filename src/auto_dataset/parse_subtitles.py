import argparse
import os
import re
import pandas as pd


METADATA_DIR = "auto_dataset/metadata"


def timestamp_to_seconds(timestamp: str) -> float:
    timestamp = timestamp.replace(",", ".")

    parts = timestamp.split(":")

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds

    return float(parts[0])


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def parse_vtt(vtt_path: str, video_id: str):
    rows = []

    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content)

    clip_index = 0

    for block in blocks:
        lines = block.strip().splitlines()

        if not lines:
            continue

        timestamp_line = None

        for line in lines:
            if "-->" in line:
                timestamp_line = line
                break

        if timestamp_line is None:
            continue

        start_str, end_str = timestamp_line.split("-->")
        start_str = start_str.strip().split(" ")[0]
        end_str = end_str.strip().split(" ")[0]

        start = timestamp_to_seconds(start_str)
        end = timestamp_to_seconds(end_str)

        text_lines = [
            line for line in lines
            if "-->" not in line and not line.strip().isdigit()
        ]

        text = clean_text(" ".join(text_lines))

        if not text:
            continue

        duration = end - start

        if duration < 0.4:
            continue

        clip_id = f"{video_id}_clip_{clip_index:04d}"

        rows.append({
            "clip_id": clip_id,
            "video_id": video_id,
            "start": start,
            "end": end,
            "duration": duration,
            "text": text,
        })

        clip_index += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtt", required=True, help="Path to .vtt subtitle file")
    parser.add_argument("--video-id", required=True, help="YouTube video ID")
    args = parser.parse_args()

    os.makedirs(METADATA_DIR, exist_ok=True)

    rows = parse_vtt(args.vtt, args.video_id)

    df = pd.DataFrame(rows)

    output_path = os.path.join(
        METADATA_DIR,
        f"{args.video_id}_metadata.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"[✓] Metadata kaydedildi: {output_path}")
    print(f"[✓] Segment sayısı: {len(df)}")

    if len(df) > 0:
        print(df.head())


if __name__ == "__main__":
    main()