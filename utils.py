import numpy as np

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
