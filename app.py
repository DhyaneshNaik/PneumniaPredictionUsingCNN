from flask import *
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# classes of the Traffic Signs
classes = {
    0: "NORMAL",
    1: "PNEUMONIA"
}


def image_processing(img):
    model = load_model('./model/PNP.h5')
    data = []
    imgs = image.load_img(img, target_size=(150, 150), grayscale=True)
    #X = np.array(imgs)
    #X = np.expand_dims(X, axis=0)
    imgs = image.img_to_array(imgs)
    imgs = np.array(imgs)/255
    data.append(imgs)
    X = np.array(data)
    pred = model.predict_classes(X)
    return pred


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['imageUpload']
        file_name = secure_filename(f.filename)
        file_path = "./static/images/" + file_name
        f.save(file_path)

        result = image_processing(file_path)
        output = "Prediction is : " + classes[int(result)]
        # os.remove(file_path)
        data = {'image': file_path, 'output': output}
        return render_template('predict.html',
                               prediction=output,
                               path=file_name)
    return None


if __name__ == "__main__":
    app.run(debug=True)
