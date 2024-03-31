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
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# "https://youtu.be/osUyjwDwjlg?si=IrZdeVeCh62KgL8C"

class PeanutBot():
    def __init__(self, url) -> None:
        self.chat_history = []
        audio_file_path, video_file_path = self.load_video(url)
        print(audio_file_path)
        print(video_file_path)

        output_folder = "output_frames"
        unique_output_folder = "unique_frames"

        # Extract frames from the video
        self.extract_frames(video_file_path, output_folder)

        # Remove duplicate frames
        self.remove_duplicates(output_folder, unique_output_folder)

        #model initialization
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device("cpu")

        self.model.to(self.device)

        max_length = 32
        num_beams = 8
        self.gen_kwargs = {"max_length":max_length,"num_beams":num_beams}

        #get all the files(unique images) in a list from the unique frames folder
        folder_path = "unique_frames"
        files_in_folder = self.get_files_in_folder(folder_path)


        #for each image we call the functions for predicting caption and extracting text
        list_of_dictionaries = []

        for i in range(len(files_in_folder)):
            predicted_caption = self.predict_caption(files_in_folder[i])
            extracted_text = self.extract_text_with_pytesseract(files_in_folder[i])
            list_of_dictionaries.append({"caption": predicted_caption, "text": extracted_text})

        print(list_of_dictionaries)

        #audio part
        res = self.process_video(audio_file_path)
        audio_text = res['text']
        print(audio_text)

        combined_text = self.combine_list_of_dict_audio_text(list_of_dictionaries, audio_text)
        print(combined_text)
        print(type(combined_text))
        clean_combined_text = self.clean_text(combined_text)

        #chain
        self.make_chain(clean_combined_text)
    
    def generateResponse(self, query):
        result = self.question_answer(query, chain)
        print(result)
        print("Question: ", query)
        print("Answer: ", result)
        return result

    #function to download audio and video and return the paths
    def load_video(self, url: str) -> tuple[str, str]:
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

    def extract_frames(self, video_path, output_folder, interval_seconds=3):
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

    def remove_duplicates(self, input_folder, output_folder):
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


    def get_files_in_folder(self, folder_path):
        files = os.listdir(folder_path)
        return [os.path.join(folder_path, file) for file in files]


    def predict_caption(self, image_path):
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")


        pixel_values = self.feature_extractor(images=i_image, return_tensors="pt").pixel_values

        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        preds = [pred.strip() for pred in preds]
        print("Final Caption is: ",preds)
        return preds


    def extract_text_with_pytesseract(self, image_path):    
        i_image = Image.open(image_path)
        text = tess.image_to_string(i_image)
        print(text)
        return text


    #-------------AUDIO PART----------------------------
    def process_video(self, filepath):
        print('Transcribing Video with whisper base model')
        model = whisper.load_model("base").to('cpu')
        print("hello")
        result = model.transcribe(filepath,fp16=False)
        print("world")
        return result


    ##Combining list_of_dictionaries data with audio text:
    def combine_list_of_dict_audio_text(self, dict_list, audio_text):
        combined_text = ""
        for item in dict_list:
            caption = item.get("caption","")
            text = item.get("text","")
            combined_text += caption[0] + " " + text + " "
        combined_text += audio_text
        return combined_text


    def clean_text(self, text):
        text = text.replace("\n", " ")
        # Remove special characters except for ".", "?", and "!"
        text = re.sub(r"[^\w\s.?!]", "", text)
        print("Final Text: ")
        print(text)
        return text



    def make_chain(self, text):
        global chain
        vector_stores = Chroma.from_texts(texts = text, embedding=OpenAIEmbeddings())
        chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.5),
                                                      retriever=vector_stores.as_retriever(search_kwargs={"k":5}),
                                                      return_source_documents = True)
        return chain

    def question_answer(self, query, chain):
        global chat_history
        result = chain({"question":query, "chat_history":chat_history}, return_only_outputs=True)
        chat_history += [(query, result["answer"])]
        return result["answer"]

peanuts = PeanutBot("https://youtu.be/osUyjwDwjlg?si=IrZdeVeCh62KgL8C")

while True:
    q = input("Ask::")
    print(f'Answer::{peanuts.generateResponse(q)}')