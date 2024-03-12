import cv2
import os
from PIL import Image
import imagehash
from pytube import YouTube

def load_video(url:str) -> str:
    global result 

    yt = YouTube(url)
    target_dir = os.path.join('C:\\Users\\Lenovo\\Documents', 'Youtube')
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if os.path.exists(target_dir+'\\'+yt.title+'.mp4'):
        return target_dir+'\\'+yt.title+'.mp4'
    try:
        
        yt.streams.filter(only_audio=True)
        stream = yt.streams.get_audio_only()
        print('----DOWNLOADING AUDIO FILE----')
        stream.download(output_path=target_dir)
    except:
        raise gr.Error('Issue in Downloading video')
    #print(target_dir+'\\'+yt.title.replace('|','').replace('?','')+'.mp4')
    
    return target_dir+'\\'+yt.title.replace('|','').replace('?','').replace('\\','').replace('/','').replace(':','').replace('*','').replace('"','').replace('<','').replace('>','').replace('.','')+'.mp4'
