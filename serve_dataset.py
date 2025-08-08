import os
import requests
import zipfile

images = {
    "relogio": [
        "https://upload.wikimedia.org/wikipedia/commons/2/29/Analog_watch.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/e7/Wristwatch.jpg"
    ],
    "camiseta": [
        "https://upload.wikimedia.org/wikipedia/commons/8/8e/Tshirt.svg",
        "https://upload.wikimedia.org/wikipedia/commons/9/97/Tshirt_blue.svg"
    ],
    "bicicleta": [
        "https://upload.wikimedia.org/wikipedia/commons/0/0b/Bicycle.png",
        "https://upload.wikimedia.org/wikipedia/commons/d/d1/Bikeicon.svg"
    ],
    "sapato": [
        "https://upload.wikimedia.org/wikipedia/commons/1/17/Shoe_icon.png",
        "https://upload.wikimedia.org/wikipedia/commons/0/0f/Sneakers_icon.png"
    ]
}

os.makedirs("dataset", exist_ok=True)

for category, urls in images.items():
    os.makedirs(f"dataset/{category}", exist_ok=True)
    for i, url in enumerate(urls, start=1):
        img_path = f"dataset/{category}/{category}_{i}.jpg"
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(r.content)
            print(f"Baixado: {img_path}")
        else:
            print(f"Falhou: {url}")

zip_filename = "dataset.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for root, _, files in os.walk("dataset"):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=os.path.relpath(file_path, "dataset"))

print(f"Dataset compactado como: {zip_filename}")
