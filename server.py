from flask import Flask, request, jsonify
from pytube import YouTube
from threading import Thread
from peanut import PeanutBot
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')

peanut_bot = None

def initialize_peanut_bot(link):
    global peanut_bot
    peanut_bot = PeanutBot(link)

@app.route('/get_answer', methods=['GET'])
def get_answer():
    print("inside get answer")
    
    query = request.args.get('query')
    
    print("query::", query)
    
    response = {
        'answer': peanut_bot.generateResponse(query) if peanut_bot else 'Still processing.. please wait.'
    }

    return jsonify(response)

@app.route('/is_processing', methods=['GET'])
def check_if_processing():
    # 1 represents processed, anything else represents not processed
    
    is_processed = False
    
    if peanut_bot:
        is_processed = peanut_bot.is_processing_complete
    
    response = {
        'processed': 1 if is_processed else 0
    }
    
    print("processed::", is_processed)

    return jsonify(response)

@app.route('/update_vid_link', methods=['GET'])
def update_link():
    print("inside update link")
    global peanut_bot
    
    link = request.args.get('link')
    print("link found::", link)
    
    if not link:
        return jsonify({'error': 'Missing parameters'}), 400
    
    if peanut_bot:
        del peanut_bot
        peanut_bot = None
    
    thread = Thread(target=initialize_peanut_bot, args=(link,))
    thread.start()
    
    ytObj = YouTube(link)
    
    response = {
        'length': ytObj.length,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
