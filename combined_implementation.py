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

tess.pytesseract.tesseract_cmd = r'C:\Users\alaqm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

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

def extract_frames(video_path, output_folder, interval_seconds=3):
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


def get_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    return [os.path.join(folder_path, file) for file in files]


def predict_caption(image_path):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
        
    
    pixel_values = feature_extractor(images=i_image, return_tensors="pt").pixel_values

    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    preds = [pred.strip() for pred in preds]
    print("Final Caption is: ",preds)
    return preds


def extract_text_with_pytesseract(image_path):    
    i_image = Image.open(image_path)
    text = tess.image_to_string(i_image)
    print(text)
    return text


if __name__=="__main__":
    url = "https://youtu.be/T7xyYCdapAA?si=qTHx--CzzKNglWqu"
    audio_file_path, video_file_path = load_video(url)
    print(audio_file_path)
    print(video_file_path)

    output_folder = "output_frames"
    unique_output_folder = "unique_frames"

    # Extract frames from the video
    extract_frames(video_file_path, output_folder)

    # Remove duplicate frames
    remove_duplicates(output_folder, unique_output_folder)

    #model initialization
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cpu")

    model.to(device)

    max_length = 32
    num_beams = 8
    gen_kwargs = {"max_length":max_length,"num_beams":num_beams}

    #get all the files(unique images) in a list from the unique frames folder
    folder_path = "unique_frames"
    files_in_folder = get_files_in_folder(folder_path)


    #for each image we call the functions for predicting caption and extracting text
    list_of_dictionaries = []

    for i in range(len(files_in_folder)):
        predicted_caption = predict_caption(files_in_folder[i])
        extracted_text = extract_text_with_pytesseract(files_in_folder[i])
        list_of_dictionaries.append({"caption": predicted_caption, "text": extracted_text})

    print(list_of_dictionaries)

