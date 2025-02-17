# upload/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os, sys
import io
import uvicorn
import cv2
import numpy as np
from PIL import Image, ImageDraw
from utils import recognize,get_databse,get_encoding_list,get_result
import uuid

IMAGEDIR = "images/"

TOLERANCE = 0.7

def init_app():
    app = FastAPI(
         title='Test FastAPI',
         description="Fast API Test",
         version="1.0.1")

    # Root route
    @app.get("/")
    def root():
        return {"message": "Welcome to the FastAPI root route!"}


    @app.post("/upload/")
    async def create_upload_file(file: UploadFile = File(...)):
        try:
            #file_name = os.getcwd() +"/"+ file.filename.replace(" ", "-")
            file.filename = f"{uuid.uuid4()}.jpg"
            contents = await file.read()

            # save the file
            tmpfilename = f"{IMAGEDIR}{file.filename}"
            with open(tmpfilename, "wb") as f:
                f.write(contents)
                f.close()

            img = Image.open(tmpfilename)
            known_encoding, id_info = get_encoding_list("select * from SHOPPER")
            result = recognize(tmpfilename, known_encoding, id_info)

            #print(f'Results:{result}')
            #cv2.imshow('WyzeCam', image)

            # save the file
            #with open(file_name,'wb+') as f:
            #    f.write(contents)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            raise HTTPException(status_code=500, detail=f'Something went wrong. Error:{e,exc_tb.tb_lineno}')
        finally:
            file.file.close()

        return result

    return app

app = init_app()

if __name__== '__main__':
    uvicorn.run("app:app", host="localhost", port=8888, reload=True)