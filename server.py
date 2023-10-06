from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import json
import PIL

app = Flask(__name__)
model = load_model("https://github.com/Anshu-Patel-Vikalp/Server/blob/main/mobilenetv2.h5")

def preprocess_input(img):
    img = np.array(img).reshape((1, 224, 224, 3))
    img = img / 255.0
    return img

@app.route('/model', methods=['POST'])
def model():
    data = json.loads(request.data)
    img = data['img']
    img = np.array(img)
    img = PIL.Image.fromarray(np.uint8(img))
    img = preprocess_input(img)
    prediction = model.predict(img)
    return ("The prediction is {}".format(['Fabric', 'Glass', 'Leather', 'Metal', 'Paper', 'Plastic', 'Wood'][model.predict(img).argmax()]))

if __name__ == '__main__':
    model = load_model("https://github.com/Anshu-Patel-Vikalp/Server/blob/main/mobilenetv2.h5")
    app.run(debug=True)

