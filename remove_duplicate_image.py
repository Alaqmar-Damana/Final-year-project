import cv2
import os
from PIL import Image
import imagehash

def extract_frames(video_path, output_folder, interval_seconds=1):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(frame_rate * interval_seconds)

    # Loop through the video and extract frames
    while success:
        if count % interval_frames == 0:
            frame_path = os.path.join(output_folder, f"frame_{count // interval_frames}.jpg")
            cv2.imwrite(frame_path, image)  # Save frame as JPEG file
        success, image = vidcap.read()
        count += 1

    print(f"{count // interval_frames} frames extracted.")

def remove_duplicates(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dictionary to store image hashes
    hashes = {}

    # Loop through images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)

            # Calculate image hash
            with Image.open(image_path) as img:
                h = imagehash.average_hash(img)

            # Check for duplicate hash
            if h not in hashes:
                hashes[h] = image_path

    # Save unique images to the output folder
    for hash_val, image_path in hashes.items():
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        os.rename(image_path, output_path)

    print(f"{len(hashes)} unique images saved.")

if __name__ == "__main__":
    video_path = "oppenheimer.mp4"
    output_folder = "output_frames"
    unique_output_folder = "unique_frames"

    # Extract frames from the video
    extract_frames(video_path, output_folder)

    # Remove duplicate frames
    remove_duplicates(output_folder, unique_output_folder)
