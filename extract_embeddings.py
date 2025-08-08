import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Carrega o modelo pré-treinado sem a última camada (para extrair features)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = base_model.predict(img_array, verbose=0)
    return embedding.flatten()

dataset_path = "dataset"
embeddings = []
image_paths = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, file)
            vec = get_embedding(path)
            embeddings.append(vec)
            image_paths.append(path)
            print(f"Processado: {path}")

embeddings = np.array(embeddings)
df = pd.DataFrame({"path": image_paths, "embedding": list(embeddings)})
df.to_pickle("image_embeddings.pkl")

print(f"Extração concluída! {len(image_paths)} imagens processadas.")
