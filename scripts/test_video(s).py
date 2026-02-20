import sys
from main import main as run_main

def run_test_video(video_path, output_path):
    global VIDEO_PATH, OUTPUT_PATH
    VIDEO_PATH = video_path
    OUTPUT_PATH = output_path
    run_main()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 test_video.py <video_path> <output_path>")
        sys.exit()

    video_path = sys.argv[1]
    output_path = sys.argv[2]
    run_test_video(video_path, output_path)