from flask import Flask, request, render_template, send_file
import cv2 as cv
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
prototxt = "models/colorization_deploy_v2.prototxt"
model = "models/colorization_release_v2.caffemodel"
pts = "models/pts_in_hull.npy"

net = cv.dnn.readNetFromCaffe(prototxt, model)
pts_in_hull = np.load(pts)

# Add cluster centers as 1x1 convolutions
class8_ab = net.getLayerId("class8_ab")
conv8_313_rh = net.getLayerId("conv8_313_rh")
pts = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8_ab).blobs = [pts.astype(np.float32)]
net.getLayer(conv8_313_rh).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read and resize image
    frame = cv.imread(filepath)
    h_orig, w_orig = frame.shape[:2]
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB).astype("float32") / 255.0
    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    l_channel = img_lab[:, :, 0]

    # Resize L channel for model
    l_rs = cv.resize(l_channel, (224, 224))
    l_rs -= 50  # Normalize as expected by model
    net_input = cv.dnn.blobFromImage(l_rs)

    net.setInput(net_input)
    ab_output = net.forward()[0, :, :, :].transpose((1, 2, 0))  # (224x224x2)
    ab_output_us = cv.resize(ab_output, (w_orig, h_orig))  # Match original size

    # Merge original L with predicted ab channels
    l_channel_orig = l_channel.reshape(h_orig, w_orig, 1)
    lab_output = np.concatenate((l_channel_orig, ab_output_us), axis=2)

    # Convert back to BGR
    img_bgr_out = cv.cvtColor(lab_output.astype("float32"), cv.COLOR_Lab2BGR)
    img_bgr_out = np.clip(img_bgr_out, 0, 1)
    img_bgr_out = (255 * img_bgr_out).astype("uint8")

    output_path = os.path.join(UPLOAD_FOLDER, "colorized_" + filename)
    cv.imwrite(output_path, img_bgr_out)

    return send_file(output_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
