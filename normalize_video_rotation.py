#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def get_rotation_metadata(video_path):
    """Get rotation metadata from video file"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", 
        "-show_streams", "-select_streams", "v:0", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if "streams" in data and len(data["streams"]) > 0:
            stream = data["streams"][0]
            
            # Check for rotation in side_data
            if "side_data_list" in stream:
                for side_data in stream["side_data_list"]:
                    if side_data.get("side_data_type") == "Display Matrix":
                        rotation = side_data.get("rotation", 0)
                        return int(rotation)
            
            # Check for rotation in tags
            if "tags" in stream and "rotate" in stream["tags"]:
                return int(stream["tags"]["rotate"])
                
        return 0
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return 0


def normalize_video_rotation(video_path, backup=True):
    """Normalize video rotation by applying metadata rotation to pixel data"""
    rotation = get_rotation_metadata(video_path)
    
    if rotation == 0:
        print(f"No rotation needed for {video_path}")
        return False
    
    print(f"Found {rotation}° rotation metadata in {video_path}")
    
    if backup:
        backup_path = video_path + ".bk"
        if not os.path.exists(backup_path):
            os.rename(video_path, backup_path)
            source_path = backup_path
        else:
            source_path = video_path
    else:
        source_path = video_path
    
    # Determine ffmpeg filter based on rotation
    if rotation == 90 or rotation == -270:
        vf_filter = "transpose=1"  # 90° clockwise
    elif rotation == 180 or rotation == -180:
        vf_filter = "transpose=1,transpose=1"  # 180°
    elif rotation == 270 or rotation == -90:
        vf_filter = "transpose=2"  # 90° counter-clockwise
    else:
        print(f"Unsupported rotation angle: {rotation}°")
        return False
    
    cmd = [
        "ffmpeg", "-i", source_path,
        "-vf", vf_filter,
        "-metadata:s:v:0", "rotate=",
        "-c:a", "copy",
        "-f", "mp4",
        "-y", video_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Successfully normalized rotation for {video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to normalize {video_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Normalize video rotation metadata to pixel data")
    parser.add_argument("videos_dir", help="Directory containing video files")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        print(f"Directory not found: {videos_dir}")
        sys.exit(1)
    
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
    
    processed = 0
    for video_path in video_files:
        if normalize_video_rotation(str(video_path), backup=not args.no_backup):
            processed += 1
    
    print(f"Processed {processed}/{len(video_files)} videos")


if __name__ == "__main__":
    main()