import cv2
import os
from PIL import Image
import imagehash
from pytube import YouTube
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import pytesseract as tess
from pytesseract import image_to_string 
from io import BytesIO
import whisper
import subprocess


tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#function to download audio and video and return the paths
def load_video(url: str) -> (str, str):
    yt = YouTube(url)
    current_folder = os.getcwd()
    target_dir = os.path.join(current_folder, 'Youtube')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    audio_file_path = os.path.join(target_dir, f"{yt.title}_audio.mp3")
    video_file_path = os.path.join(target_dir, f"{yt.title}_video.mp4")

    # Check if files already exist
    if os.path.exists(audio_file_path) and os.path.exists(video_file_path):
        return audio_file_path, video_file_path

    try:
        # Download audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        print('----DOWNLOADING AUDIO FILE----')
        audio_stream.download(output_path=target_dir, filename=f"{yt.title}_audio.mp3")

        # Download video stream
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        print('----DOWNLOADING VIDEO FILE----')
        video_stream.download(output_path=target_dir, filename=f"{yt.title}_video.mp4")

    except Exception as e:
        print('Issue in Downloading video')

    return audio_file_path, video_file_path

#------------------------------------------------------------------------------------------------

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


if __name__=="__main__":
    url = "https://youtu.be/osUyjwDwjlg?si=IrZdeVeCh62KgL8C"
    audio_file_path, video_file_path = load_video(url)
    print(audio_file_path)
    print(video_file_path)

    output_folder = "output_frames"
    unique_output_folder = "unique_frames"

    # Extract frames from the video
    extract_frames(video_file_path, output_folder)

    # Remove duplicate frames
    remove_duplicates(output_folder, unique_output_folder)


    # Define the directory containing the images
    IMG_DIR = os.path.expanduser("E:\\Final-year-project\\unique_frames\\")

    # Loop through each image in the directory
    for img in os.listdir(IMG_DIR):
        if img.endswith(".jpg"):
            # Extract the base name of the image without extension
            base_name = os.path.splitext(img)[0]

            # Define the output file name based on the image name
            output_file = os.path.join(IMG_DIR, f"{base_name}.txt")

            # Define the shell command
            command = [
                "llama.cpp\\build\\bin\\Debug\\llava-cli.exe",
                "-m", "ggml-model-q5_k.gguf",
                "--mmproj", "mmproj-model-f16.gguf",
                "--temp", "0.1",
                "-p", "Describe the image in detail.",
                "--image", os.path.join(IMG_DIR, img)
            ]

            # Execute the command and save the output to the defined output file
            with open(output_file, "w") as f:
                subprocess.run(command, stdout=f)

    print("Process completed successfully.")



