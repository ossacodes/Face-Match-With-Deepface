from fastapi import FastAPI
import cv2
import urllib.request
import numpy as np
from mangum import Mangum
from deepface import DeepFace

app = FastAPI()
handler = Mangum(app)


@app.get("/")
def read_root():
    return {"Hello": "Welcome to face match API"}


@app.get("/verifyface/")
def verify_face(image_1: str, image_2: str):
    
    # Download the image from the URL
    req = urllib.request.urlopen(image_1)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

    # Convert the raw bytes to a NumPy array in BGR format
    img1 = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Download the image from the URL
    req2 = urllib.request.urlopen(image_2)
    arr2 = np.asarray(bytearray(req2.read()), dtype=np.uint8)

    # Convert the raw bytes to a NumPy array in BGR format
    img2 = cv2.imdecode(arr2, cv2.IMREAD_COLOR)

    result = DeepFace.verify(img1, img2, enforce_detection= True, model_name="DeepFace")
    print("Result: ", result)
    return {"data": result, 'verification': result['verified']}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}



