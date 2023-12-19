
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from inference_utils import inference, create_model
from typing import List, Optional
from pydantic import BaseModel
from fastapi import UploadFile, File
import io
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import base64
class Parameters(BaseModel):
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    netG:str = 'resnet_9blocks'
    norm: str = 'instance'
    init_type: str = 'xavier'
    init_gain: float = 0.02
    display_sides: int = 1
    num_classes: int = 1
    gpu_ids: List[int] = [0]
    isTrain: bool = False
    checkpoints_dir : str = r"C:\Users\AbdulRehman\Downloads\crack_detection\Concrete-Crack-Detection-Segmentation\models"
    name : str = "deepcrack"

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

def from_image_to_bytes(img):
    """
    pillow image to bytes
    """
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()

    encoded = base64.b64encode(imgByteArr)
    decoded = encoded.decode('ascii')
    return decoded

#python-multipart
@app.post("/predict/{dim}/{unit}")
async def predict(dim:str, unit:str, file: UploadFile = File(...)):
    bytesImg = await file.read()
    width, height = [float(x) for x in dim.split('-')]    
    parameters = Parameters()
    print("Priting the parameters: ",parameters)
    checkpoint_path = r"C:\Users\AbdulRehman\Downloads\crack_detection\Concrete-Crack-Detection-Segmentation\models\deepcrack\pretrained_net_G.pth"
    model = create_model(parameters,checkpoint_path)
    predicted_cntr, visuals = inference(model, bytesImg, (width, height), unit) # Outputs Predicted Masks

    cntr_converted = from_image_to_bytes(Image.fromarray(predicted_cntr))
    
    img_list = [cntr_converted]
    
    for index,k in enumerate(['fused', 'side1', 'side2', 'side3', 'side4', 'side5']):
        map_ = from_image_to_bytes(Image.fromarray(visuals[k]))
        img_list.append(map_)
        Image.fromarray(visuals[k]).save(k+".jpeg")
        print("type of response: ",type(map_))
    return True