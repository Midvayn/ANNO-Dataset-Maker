import os, cv2, base64, io, csv
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

app = Flask(__name__)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.onnx")
TAGS_PATH = os.path.join(MODELS_DIR, "wd14_tags.csv")

ort_session = None
tag_names = []

def load_ai():
    global ort_session, tag_names

    if os.path.exists(TAGS_PATH):
        try:
            with open(TAGS_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                tag_names = [row[1] for row in reader]
            print(f"[SYSTEM] Tags loaded: {len(tag_names)}")
        except Exception as e:
            print(f"[ERROR] Failed to load tags CSV: {e}")

    if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
            print("[SYSTEM] AI model ready")
        except Exception:
            try:
                ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
                print("[SYSTEM] AI model running on CPU")
            except Exception as e:
                print(f"[FATAL] Failed to start model: {e}")

def preprocess_image(image: Image.Image, size=448):
    w, h = image.size
    max_dim = max(w, h)
    pad_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    pad_img.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    img = pad_img.resize((size, size), Image.Resampling.BICUBIC)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np[..., ::-1]
    return np.expand_dims(img_np, 0)

load_ai()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze_heatmap", methods=["POST"])
def analyze_heatmap():
    try:
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        open_cv_image = np.array(img)[:, :, ::-1].copy()
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(open_cv_image)

        if success:
            saliency_map = (saliency_map * 255).astype("uint8")
            heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
            res_img = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        else:
            res_img = img

        buf = io.BytesIO()
        res_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({"status": "ok", "heatmap": f"data:image/png;base64,{b64}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/get_tags", methods=["POST"])
def get_tags():
    if not ort_session:
        return jsonify({"status": "error", "tags": "Error: model not loaded"})

    try:
        file = request.files["image"]
        threshold = float(request.form.get("threshold", 0.35))
        img = Image.open(file.stream).convert("RGB")
        input_data = preprocess_image(img)
        input_name = ort_session.get_inputs()[0].name
        probs = ort_session.run(None, {input_name: input_data})[0][0]

        found_tags = []
        upper = min(len(probs), len(tag_names))
        for i in range(4, upper):
            if probs[i] > threshold:
                found_tags.append(tag_names[i])

        return jsonify({"status": "ok", "tags": ", ".join(found_tags)})
    except Exception as e:
        return jsonify({"status": "error", "tags": f"Error: {str(e)}"})

if __name__ == "__main__":
    print("--- PALADIN READY ---")
    app.run(host="0.0.0.0", port=5000, debug=False)

# Powered by ChatGPT
