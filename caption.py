from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import pytesseract as tess
from pytesseract import image_to_string 
from io import BytesIO

tess.pytesseract.tesseract_cmd = r'C:\Users\alaqm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# device = torch.device("cpu")

# model.to(device)

# max_length = 32
# num_beams = 8
# gen_kwargs = {"max_length":max_length,"num_beams":num_beams}

# def predict_caption(image_paths):
#     images = []
#     for image_path in image_paths:
#         i_image = Image.open(image_path)
#         if i_image.mode != "RGB":
#             i_image = i_image.convert(mode="RGB")
#         images.append(i_image)
    
#     pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values

#     pixel_values = pixel_values.to(device)

#     output_ids = model.generate(pixel_values, **gen_kwargs)

#     preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

#     preds = [pred.strip() for pred in preds]
#     print("Final Caption is: ",preds)
#     return preds

def extract_text_with_pytesseract(image_paths):
    
    extracted_text = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        text = tess.image_to_string(i_image)
        print(text)
        extracted_text.append(text)
    return extracted_text


text_with_pytesseract = extract_text_with_pytesseract(["image1.jpeg","image2.jpeg","image3.jpeg"])

print(text_with_pytesseract)

#predict_caption(["image1.jpeg","image2.jpeg","image3.jpeg"])