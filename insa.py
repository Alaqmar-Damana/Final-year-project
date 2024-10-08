import cv2
import os, shutil
from PIL import Image
import imagehash
from moviepy.editor import VideoFileClip
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pytube import YouTube
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import pytesseract as tess
from pytesseract import image_to_string 
from io import BytesIO
import whisper
import re
import yt_dlp
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv, find_dotenv

#load_dotenv(find_dotenv())

tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# "https://youtu.be/osUyjwDwjlg?si=IrZdeVeCh62KgL8C"

class PeanutBot():
    def __init__(self, url=None, video=None) -> None:
        
        self.delete_folders()
        if url:
            print("url part executed")
            self.url = url
            audio_file_path, video_file_path = self.load_video(url)
        else:
            print("upload part executed")
            video_file_path = video
            audio_file_path = video[:-3]+"mp3"
            self.convert_video_to_audio(video,audio_file_path)
            
            
        
        self.chat_history = []
        
        print(audio_file_path)
        print(video_file_path)
        
        self.is_processing_complete = False

        output_folder = "output_frames"
        unique_output_folder = "unique_frames"

        # Extract frames from the video
        self.extract_frames(video_file_path, output_folder)

        # Remove duplicate frames
        self.remove_duplicates(output_folder, unique_output_folder)
        
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

        #model initialization
        # self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # self.device = torch.device("cpu")

        # self.model.to(self.device)

        # max_length = 32
        # num_beams = 8
        # self.gen_kwargs = {"max_length":max_length,"num_beams":num_beams}

        #get all the files(unique images) in a list from the unique frames folder
        folder_path = "unique_frames"
        files_in_folder = self.get_files_in_folder(folder_path)


        #for each image we call the functions for predicting caption and extracting text
        list_of_dictionaries = []

        for i in range(len(files_in_folder)):
            # predicted_caption = self.predict_caption(files_in_folder[i])
            extracted_text = self.extract_text_with_pytesseract(files_in_folder[i])
            list_of_dictionaries.append({"text": extracted_text})

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
        
    def delete_folders(self):
        if os.path.isdir("output_frames"):
            shutil.rmtree("output_frames")
        if os.path.isdir("unique_frames"):
            shutil.rmtree("unique_frames")
    
    def convert_video_to_audio(self, video_file, audio_file):
        try:
            video_clip = VideoFileClip(video_file)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_file)
            video_clip.close()
        except Exception as e:
            print("Error:", e)


    def generateResponse(self, query):
        result = self.qa.run(query)
        print(result)
        print("Question: ", query)
        print("Answer: ", result)
        return result

    def video_duration_in_seconds(self) -> int:
        return self.video_length_in_seconds

    #function to download audio and video and return the paths
    def load_video(self, url: str) -> tuple[str, str]:
        try:
            # Set up directories
            current_folder = os.getcwd()
            target_dir = os.path.join(current_folder, 'Youtube')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

            # Video options for yt-dlp
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',  # Best video and audio
                'outtmpl': os.path.join(target_dir, '%(title)s.%(ext)s'),  # Save path
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',  # Ensure conversion to mp4
                    'preferedformat': 'mp4'
                }]
            }

            # Audio options for yt-dlp
            ydl_audio_opts = {
                'format': 'bestaudio/best',  # Best audio
                'outtmpl': os.path.join(target_dir, '%(title)s_audio.%(ext)s'),  # Save path
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',  # Extract audio
                    'preferredcodec': 'mp3',  # Convert to mp3
                    'preferredquality': '192',  # Set bitrate
                }]
            }

            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print('----DOWNLOADING VIDEO FILE----')
                ydl.download([url])

            # Download audio
            with yt_dlp.YoutubeDL(ydl_audio_opts) as ydl:
                print('----DOWNLOADING AUDIO FILE----')
                ydl.download([url])

            # Set file paths based on the video title
            video_info = yt_dlp.YoutubeDL().extract_info(url, download=False)
            video_title = video_info['title']
            video_file_path = os.path.join(target_dir, f"{video_title}.mp4")
            audio_file_path = os.path.join(target_dir, f"{video_title}_audio.mp3")

            return audio_file_path, video_file_path

        except Exception as e:
            print(f"Issue in Downloading video: {e}")
            return None, None

    #------------------------------------------------------------------------------------------------

    def extract_frames(self, video_path, output_folder, interval_seconds=3):
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        vidcap = cv2.VideoCapture(video_path)
        
        # Check if the video was opened successfully
        if not vidcap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        # Get frame rate and check if it's valid
        frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            print(f"Error: Unable to get FPS for video {video_path}")
            return
        
        success, image = vidcap.read()
        count = 0
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


    # def predict_caption(self, image_path):
    #     i_image = Image.open(image_path)
    #     if i_image.mode != "RGB":
    #         i_image = i_image.convert(mode="RGB")


    #     pixel_values = self.feature_extractor(images=i_image, return_tensors="pt").pixel_values

    #     pixel_values = pixel_values.to(self.device)

    #     output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

    #     preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    #     preds = [pred.strip() for pred in preds]
    #     print("Final Caption is: ",preds)
    #     return preds


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
            #caption = item.get("caption","")
            text = item.get("text","")
            combined_text += text + " "
        combined_text += audio_text
        return combined_text


    def clean_text(self, text):
        text = text.replace("\n", " ")
        # Remove special characters except for ".", "?", and "!"
        text = re.sub(r"[^\w\s.?!]", "", text)
        print("Final Text: ")
        print(text)
        return text

    def isProcessed(self):
        return self.is_processing_complete

    def make_chain(self, text):
        file_name = "output1234.txt"

# Open the file in write mode ('w' mode)
        with open(file_name, 'w') as file:
            # Write the string to the file
            file.write(text)

        print("String has been written to the file:", file_name)
        loader = TextLoader("output1234.txt")
        docs = loader.load()
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_stores = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=api_key))
                # self.chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.5),
                #                                               retriever=vector_stores.as_retriever(search_kwargs={"k":5}),
                #                                               return_source_documents = True)
                
        retriever = vector_stores.as_retriever()
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
##YOU SHOULD ONLY ANSWER THE QUESTION IF IT IS IN GIVEN CONTEXT OTHERWISE DO NOT ANSWER##

{context}

Question: {question}
"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        self.qa = RetrievalQA.from_chain_type(
            self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs   # pass kwargs here
        )
        self.is_processing_complete = True
        
        print("Processing complete on the video")
        # return self.chain

    # def question_answer(self, query):
    #     print(self.chain)
    #     result = self.chain({"question":query, "chat_history":self.chat_history}, return_only_outputs=True)
    #     self.chat_history += [(query, result["answer"])]
    #     return result["answer"]

# peanuts = PeanutBot(video="Youtube\Blood Moon explained in 20 seconds_video.mp4")

# while True:
#     q = input("Ask::")
#     print(f'Answer::{peanuts.generateResponse(q)}')