import os
import urllib.request
import sys

MODEL_FILENAME = "emotion-ferplus-8.onnx"
# List of potential URLs for the model
URLS = [
    "https://github.com/onnx/models/blob/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx?raw=true",
    "https://media.githubusercontent.com/media/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx", 
    "https://github.com/microsoft/onnxjs-demo/raw/master/docs/emotion_ferplus/model.onnx",
    "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.onnx"
]

def download_model():
    if os.path.exists(MODEL_FILENAME) and os.path.getsize(MODEL_FILENAME) > 1000:
        print(f"Model {MODEL_FILENAME} already exists and seems valid.")
        return True

    print(f"Downloading {MODEL_FILENAME}...")
    headers = {'User-Agent': 'Mozilla/5.0'}

    for url in URLS:
        print(f"Trying {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                if response.getcode() == 200:
                    with open(MODEL_FILENAME, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)
                    
                    if os.path.getsize(MODEL_FILENAME) > 1000:
                        print(f"Successfully downloaded from {url}")
                        return True
                    else:
                        print("File too small, possibly an error page.")
                        os.remove(MODEL_FILENAME)
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            if os.path.exists(MODEL_FILENAME):
                try:
                    os.remove(MODEL_FILENAME)
                except:
                    pass

    print("\nERROR: Could not automatically download the emotion model.")
    print(f"Please manually download the file 'emotion-ferplus-8.onnx' and place it in: {os.getcwd()}")
    print("You can try searching Google for 'emotion-ferplus-8.onnx download' or check the ONNX Model Zoo.")
    return False

if __name__ == "__main__":
    download_model()
