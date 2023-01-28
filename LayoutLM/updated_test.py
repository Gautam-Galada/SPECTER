#### installing libs 


!pip install -q gradio
!pip install Pillow==9.1.0
!pip install datasets
!sudo apt install tesseract-ocr
!pip install -q pytesseract


import os
os.system('pip install git+https://github.com/huggingface/transformers.git --upgrade')
os.system('pip install pyyaml==5.1')
# workaround: install old version of pytorch since detectron2 hasn't released packages for pytorch 1.9 (issue: https://github.com/facebookresearch/detectron2/issues/3158)
os.system('pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')

# install detectron2 that matches pytorch 1.8
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
os.system('pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html')

## install PyTesseract
os.system('pip install -q pytesseract')

import gradio as gr
import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont


from transformers import LayoutLMv3ForTokenClassification
from transformers import LayoutLMv3Processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-cord")


dataset = load_dataset("nielsr/cord-layoutlmv3", split="test")





labels = dataset.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
print(labels)

label2color1 = {"menu.nm" : "blue" ,
                "menu.num" : 'green' , 
                'menu.unitprice':'black' ,
                'menu.cnt' : 'red',  
                'menu.discountprice':'red' ,
                'menu.price':'violet' ,
                'menu.itemsubtotal':'violet',
                'menu.vatyn':'violet',
                'menu.etc':'violet',
                'menu.sub_nm':'violet',
                'menu.sub_unitprice':'violet',
                'menu.sub_cnt':'violet',
                'menu.sub_price':'violet',
                'menu.sub_etc':'violet',
                'void_menu.nm':'violet',
                'void_menu.price':'violet',
                'sub_total.subtotal_price':'violet',
                'sub_total.discount_price':'violet',
                'sub_total.service_price':'violet',
                'sub_total.othersvc_price':'violet',
                'sub_total.tax_price':'violet',
                'sub_total.etc':'violet',
                'total.total_price':'violet',
                'total.total_etc':'violet',
                'total.cashprice':'violet' ,
                'total.changeprice':'violet',
                'total.creditcardprice':'violet',
                'total.emonryprice':'violet',
                'total.menutype_cnt':'violet',
                'total.menuqty_cnt':'violet',
                'menu.nm':'violet',
                'menu.num':'violet',
                'menu.unitprice':'violet',
                'menu.cnt':'violet',
                'menu.discountprice':'violet',
                'menu.price':'violet',
                'menu.itemsubtotal':'violet',
                'menu.vatyn':'violet',
                'menu.etc':'violet',
                'menu.sub_nm':'violet',
                'menu.sub_unitprice':'violet',
                'menu.sub_cnt':'violet',
                'menu.sub_price':'violet',
                'menu.sub_etc':'violet',
                'void_menu.nm':'violet',
                'void_menu.price':'violet',
                'sub_total.subtotal_price':'violet',
                'sub_total.discount_price':'violet',
                'sub_total.service_price':'violet',
                'sub_total.othersvc_price':'violet',
                'sub_total.tax_price':'violet',
                'sub_total.etc':'violet',
                'total.total_price':'violet',
                'total.total_etc':'violet',
                'total.cashprice':'violet',
                'total.changeprice':'violet',
                'total.creditcardprice':'violet',
                'total.emoneyprice':'violet',
                'total.menutype_cnt':'violet',
                'total.menuwty_cnt':'violet',
                'other':'black'

               }


def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label


def process_image(image):
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]



     # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color1[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color1[predicted_label], font=font)
    
    return image



def process_image(image):
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]



     # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color1[predicted_label])
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color1[predicted_label], font=font)
    
    return image



result = process_image(im)

print(result)



#### Using Gradio demo



title = "Interactive demo: LayoutLMv3"
description = "Demo for Microsoft's LayoutLMv3, a Transformer for state-of-the-art document image understanding tasks. This particular model is fine-tuned on FUNSD, a dataset of manually annotated forms. It annotates the words into QUESTION/ANSWER/HEADER/OTHER. To use it, simply upload an image or use the example image below. Results will show up in a few seconds."
article = "LayoutLMv3: Multi-modal Pre-training for Visually-Rich Document Understanding | Github Repo"
examples =[['bill2.jpg']]

css = """.output_image, .input_image {height: 600px !important}"""

iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Image(type="pil", label="annotated image"),
                     title=title,
                     description=description,
                     article=article,
                     examples=examples,
                     css=css)
                     
iface.launch(debug=True)


