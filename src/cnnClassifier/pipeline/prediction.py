import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path


# Project root (parent of src/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
# Use model/ folder for deployment (can be committed to git)
MODEL_PATH = ROOT_DIR / "model" / "model.h5"


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model from model/ folder (deployment path)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Add model.h5 to the model/ folder (e.g. copy from artifacts/training/model.h5 after training)."
            )
        model = load_model(str(MODEL_PATH))

        # Load image from project root (same place app saves decoded image)
        image_path = ROOT_DIR / self.filename if not os.path.isabs(self.filename) else Path(self.filename)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}. Upload an image first.")

        test_image = image.load_img(str(image_path), target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        # Match training: training uses rescale=1./255
        test_image = test_image / 255.0
        result = np.argmax(model.predict(test_image, verbose=0), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]