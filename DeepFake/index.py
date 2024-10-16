from flask import Flask, render_template, request,flash
import sqlite3
from flask import session
from flask import Response
import os
import sys
import numpy as np
from Detection import fake_real_detection
from werkzeug.utils import secure_filename
import warnings
import cv2

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "1234"



@app.route('/')
def index():
    return render_template('index.html')


@app.route("/detection")
def detection():
    return render_template("detection.html")


@app.route("/detection_object",methods =["GET", "POST"])
def detection_object():
    image = request.files['file']

    image.save("C:/Users/shrey/Music/DeepFake/test_image/test_img.jpg")

    image_path="C:/Users/shrey/Music/DeepFake/test_image/test_img.jpg"

    result=fake_real_detection(image_path)
    print(result)
    
    return render_template("detection_result.html",result=result)
        

@app.route("/evaluation")
def evaluation():
    return render_template("model_evaluation.html")





if __name__ == '__main__':
    app.run(host="localhost", port=1122, debug=True)
