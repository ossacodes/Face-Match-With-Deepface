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
    # image1 = "data:image/jpeg;base64,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"
    # image2 = "https://library.sportingnews.com/2023-02/Lionel%20Messi%20PSG%20030922.jpg"

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

    result = DeepFace.verify(img1, img2, enforce_detection= True, model_name="Facenet")
    print("Result: ", result)
    return {"data": result, 'verification': result['verified']}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


