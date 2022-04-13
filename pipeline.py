import argparse
import llie_zerodce
from PIL import Image
import numpy as np
import cv2
import re
import time
import logging
import torch
from torchvision.utils import save_image
import os
import warnings
warnings.filterwarnings("ignore")
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import transformers
transformers.utils.logging.set_verbosity_error()

st = time.time()

parser = argparse.ArgumentParser(description='License Plate Recognition')
parser.add_argument("img_path",type=str, help="path of the image")
parser.add_argument("--show", type=bool, default=False, help="display image")
parser.add_argument("--save", type=bool, default=True, help="save annotated image")
parser.add_argument("--out_dir","--output_dir",type=str, default="./results",help="output dir for annotated image/video")
parser.add_argument("--llie",'--low_light_image_enhancement',type=bool, default=False,help="low light image enhancement")
parser.add_argument("--model",type=str,default="microsoft/trocr-small-printed",help="TrOCR model to use, you can use other models too")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT_PATH = 'weights/best.pt'
yolov5 = torch.hub.load('ultralytics/yolov5', 'custom',
                        path=CKPT_PATH,
                        force_reload=True,
                        verbose=False)

model = VisionEncoderDecoderModel.from_pretrained(args.model,cache_dir="./cache/"+args.model)
processor = TrOCRProcessor.from_pretrained(args.model,cache_dir="./cache/"+args.model) 
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


def label_gen(img_path,args):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)

    if args.llie:
        img = img/255.0
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        img = img.cuda().unsqueeze(0)

        DCE_net = llie_zerodce.enhance_net_nopool().cuda()
        DCE_net.load_state_dict(torch.load('weights/Epoch99.pth'))
        _,img,_ = DCE_net(img)
        img = np.array(torch.permute(img[0],(1,2,0)).cpu().detach().numpy()*255,np.uint8)

    # print(img.shape)

    data_img = detect(img)
    labels = []
    for i,row in data_img.iterrows():
        if row['confidence']>0.5:
            xmin,xmax,ymin,ymax = int(row['xmin']),int(row['xmax']),int(row['ymin']),int(row['ymax'])
           
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
            plate = img[ymin:ymax,xmin:xmax,:]
            plate = torch.from_numpy(plate)
            # print(plate.shape)
            label = gen_text(plate)
            label = re.sub(r'[^\w\s]', '',label)
            if args.save:
                save_image(plate.permute((2,0,1))/255.,args.out_dir+'/'+label+".png")
            labels.append(label)
    if args.show:
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.save:        
        im = Image.fromarray(img)
        im.save(args.out_dir+"/image.png")
    return labels,img


if __name__=='__main__':
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    labels,img = label_gen(args.img_path,args)
    flag=0
    for i in labels:
        logging.info(f"label:{i}")
        flag=1
    if flag==0:
        logging.info(f"no label")    
    print(f"Time: {time.time()-st}")