import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Carrega embeddings salvos
df = pd.read_pickle("image_embeddings.pkl")
embeddings = np.array(df["embedding"].to_list())
image_paths = df["path"].to_list()

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = base_model.predict(img_array, verbose=0)
    return embedding.flatten()

def search_similar(query_path, k=5):
    vec = get_embedding(query_path).reshape(1, -1)
    sims = cosine_similarity(vec, embeddings)[0]
    top_k_idx = sims.argsort()[-k:][::-1]
    return [image_paths[i] for i in top_k_idx]

def show_results(query_path, results):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(image.load_img(query_path))
    plt.title("Consulta")
    plt.axis("off")
    
    for i, res in enumerate(results, start=2):
        plt.subplot(1, len(results) + 1, i)
        plt.imshow(image.load_img(res))
        plt.title(f"Similar {i-1}")
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    query_img = "dataset/relogio/img1.jpg"  # imagem de teste
    results = search_similar(query_img, k=4)
    show_results(query_img, results)
