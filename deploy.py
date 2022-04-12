import numpy as np
import cv2
import llie_zerodce
import time
import pandas as pd
import torch
import gradio as gr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import transformers
transformers.utils.logging.set_verbosity_error()
st = time.time()
args = {'model':'microsoft/trocr-small-printed'}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT_PATH = 'weights/best.pt'
yolov5 = torch.hub.load('ultralytics/yolov5', 'custom',
                        path=CKPT_PATH,
                        force_reload=True,
                        verbose=False)

model = VisionEncoderDecoderModel.from_pretrained(args['model'],cache_dir="./cache/"+args['model'])
processor = TrOCRProcessor.from_pretrained(args['model'],cache_dir="./cache/"+args['model']) 
model.eval()
model = model.to(device)

def frame_extract(vid_path):
    vidObj = cv2.VideoCapture(vid_path) 
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image

def detect(img):
    pred = yolov5(img, size=1280, augment=False)
    return pred.pandas().xyxy[0]

def gen_text(img):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def label_gen(img,llie):

    img = np.array(img)

    if llie:
        img = img/255.0
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        img = img.cuda().unsqueeze(0)

        DCE_net = llie_zerodce.enhance_net_nopool().cuda()
        DCE_net.load_state_dict(torch.load('weights/Epoch99.pth'))
        _,img,_ = DCE_net(img)
        img = np.array(torch.permute(img[0],(1,2,0)).cpu().detach().numpy()*255,np.uint8)

    data_img = detect(img)
    results = {'labels':[]}

    for i,row in data_img.iterrows():
        if row['confidence']>0.5:
            xmin,xmax,ymin,ymax = int(row['xmin']),int(row['xmax']),int(row['ymin']),int(row['ymax'])
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
            plate = img[ymin:ymax,xmin:xmax,:]
            plate = torch.from_numpy(plate)
            label = gen_text(plate)
            results['labels'].append(label)

    return pd.DataFrame(results),img 



if __name__=='__main__':
    lpr = gr.Interface(fn=label_gen, inputs=["image",gr.inputs.Checkbox(default=False, label="Low Light Image Enhancement")], outputs=["dataframe","image"],title="License Plate Recognition")
    lpr.launch(share=False)
