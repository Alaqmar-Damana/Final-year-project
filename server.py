from flask import Flask, request, jsonify
from pytube import YouTube
from threading import Thread
from insa import PeanutBot
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip

app = Flask(__name__)
CORS(app, origins='*')

peanut_bot = None

# 0 represents YT Link, 1 represents file's name inside 'uploads' folder
def initialize_peanut_bot(link, filename):
    # print(link, type(link), filename, type(filename), init_type, type(init_type))
    # if link:
    #     print(link)
    # else:
    #     print(filename)
    # return
    global peanut_bot
    if link:
        peanut_bot = PeanutBot(url=link)
    else:
        peanut_bot = PeanutBot(video=f'uploads{os.sep}{filename}')

@app.route('/get_answer', methods=['GET'])
def get_answer():
    print("inside get answer")
    
    query = request.args.get('query')
    
    print("query::", query)
    
    response = {
        'answer': peanut_bot.generateResponse(query) if peanut_bot else 'Still processing.. please wait.'
    }

    return jsonify(response)

is_server_processing = False

@app.route('/is_processing', methods=['GET'])
def check_if_processing():
    global is_server_processing
    # 1 represents processed, anything else represents not processed
    
    is_processed = False
    
    if peanut_bot:
        is_processed = peanut_bot.is_processing_complete
    
    if is_server_processing:
        is_processed = False
    
    response = {
        'processed': 1 if is_processed else 0
    }
    
    print("processed::", is_processed)

    return jsonify(response)

@app.route('/upload', methods=['POST'])
def upload_file():
    global peanut_bot, is_server_processing
    is_server_processing = True
    print("inside upload file")
    
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        filepath = f'uploads{os.sep}{filename}'
        file.save(filepath)
        
        clip = VideoFileClip(filepath)
        length = clip.duration
        
        clip.close()
    
        if peanut_bot:
            del peanut_bot
            peanut_bot = None
        
        thread = Thread(target=initialize_peanut_bot, args=(None, filename))
        thread.start()
        
        response = {
            'length': length,
        }
        
        is_server_processing = False
        
        return jsonify(response), 200


@app.route('/update_vid_link', methods=['GET'])
def update_link():
    global peanut_bot, is_server_processing
    is_server_processing = True
    print("inside update link")
    
    link = request.args.get('link')
    print("link found::", link)
    
    if not link:
        return jsonify({'error': 'Missing parameters'}), 400
    
    if peanut_bot:
        del peanut_bot
        peanut_bot = None
    
    thread = Thread(target=initialize_peanut_bot, args=(link, None))
    thread.start()
    
    ytObj = YouTube(link)
    
    response = {
        'length': ytObj.length,
    }

    is_server_processing = False

    return jsonify(response)


if __name__ == '__main__':
    # thread = Thread(target=initialize_peanut_bot, args=("a", None, 0))
    # thread.start()
    # thread = Thread(target=initialize_peanut_bot, args=(None, "b", 1))
    # thread.start()
    app.run(port=5000, debug=True)
