import os
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras


class OCRPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path, compile=False)
        self.input_shape = self.model.input_shape
        _, self.img_height, self.img_width, self.channels = self.input_shape
        self.num_classes = int(self.model.output_shape[-1])

        # IMPORTANT:
        # Your model has 13 output classes.
        # Usually OCR with CTC means: len(charset) + 1 blank token = num_classes
        # I made charset configurable so you can match your training labels.
        default_charset = os.getenv("OCR_CHARSET", "0123456789.-")
        if len(default_charset) != self.num_classes - 1:
            fallback = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
            default_charset = fallback[: self.num_classes - 1]
        self.charset = default_charset

    def preprocess(self, image_file):
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))

        if self.channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        image = ImageOps.contain(image, (self.img_width, self.img_height))

        if self.channels == 1:
            canvas = Image.new("L", (self.img_width, self.img_height), color=255)
        else:
            canvas = Image.new("RGB", (self.img_width, self.img_height), color=(255, 255, 255))

        x = (self.img_width - image.size[0]) // 2
        y = (self.img_height - image.size[1]) // 2
        canvas.paste(image, (x, y))

        arr = np.array(canvas).astype("float32") / 255.0
        arr = 1.0 - arr

        if self.channels == 1:
            arr = np.expand_dims(arr, axis=-1)

        arr = np.expand_dims(arr, axis=0)
        return arr

    def decode_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        decoded, _ = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
        tokens = decoded[0].numpy()[0]

        chars = []
        for token in tokens:
            if token == -1:
                continue
            if 0 <= token < len(self.charset):
                chars.append(self.charset[int(token)])

        return "".join(chars).strip()

    def predict(self, image_file):
        x = self.preprocess(image_file)
        pred = self.model.predict(x, verbose=0)
        text = self.decode_prediction(pred)

        # fallback if ctc_decode returns empty
        if not text:
            argmax_tokens = pred[0].argmax(axis=-1)
            deduped = []
            prev = None
            blank_index = self.num_classes - 1

            for token in argmax_tokens:
                token = int(token)
                if token == prev:
                    continue
                prev = token
                if token == blank_index:
                    continue
                if 0 <= token < len(self.charset):
                    deduped.append(self.charset[token])

            text = "".join(deduped).strip()

        return {"prediction": text}
