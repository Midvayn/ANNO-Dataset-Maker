import os, requests

MODEL_URL = "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx"
TAGS_URL = "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2/resolve/main/selected_tags.csv"
SAVE_DIR = "models"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_file(url, filename):
    path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(path):
        print(f"[SKIP] {filename} already exists")
        return

    print(f"[DOWNLOADING] {filename}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[OK] {filename} saved")
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")

if __name__ == "__main__":
    print("--- DOWNLOADING ANNO DATASET MAKER MODEL FILES ---")
    download_file(MODEL_URL, "model.onnx")
    download_file(TAGS_URL, "wd14_tags.csv")
    print("Done. Run run.bat")
    input("Press Enter to exit...")

# Powered by ChatGPT
