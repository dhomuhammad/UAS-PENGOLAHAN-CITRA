import numpy as np
import matplotlib.pyplot as plt
import cv2

def segment_image(image_path, k=3, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)):
    # Membaca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Gambar di {image_path} tidak ditemukan atau tidak bisa dibaca.")
        return None, None
    # Ubah warna ke RGB (dari BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Membentuk ulang gambar menjadi susunan piksel 2D dengan 3 nilai warna (RGB)
    pixel_vals = image.reshape((-1, 3))
    # Mengkonversikan ke tipe float
    pixel_vals = np.float32(pixel_vals)
    # Melakukan k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Mengonversi data menjadi nilai 8-bit
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # Membentuk ulang data menjadi dimensi gambar asli
    segmented_image = segmented_data.reshape((image.shape))
    return image, segmented_image

# Path untuk gambar
image_path1 = 'images/Mobil.jpg'
image_path2 = 'images/Mobil 2.jpg'
image_path3 = 'images/Mobil 3.jpg'

# Segmentasi gambar
original_image1, segmented_image1 = segment_image(image_path1)
original_image2, segmented_image2 = segment_image(image_path2)
original_image3, segmented_image3 = segment_image(image_path3)

# Menampilkan gambar asli dan gambar tersegmentasi secara berdampingan
if all(img is not None for img in [original_image1, segmented_image1, original_image2, segmented_image2, original_image3, segmented_image3]):
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 1)
    plt.imshow(original_image1)
    plt.title("Gambar Asli 1")

    plt.subplot(3, 2, 2)
    plt.imshow(segmented_image1)
    plt.title("Gambar Tersegmentasi 1")

    plt.subplot(3, 2, 3)
    plt.imshow(original_image2)
    plt.title("Gambar Asli 2")

    plt.subplot(3, 2, 4)
    plt.imshow(segmented_image2)
    plt.title("Gambar Tersegmentasi 2")

    plt.subplot(3, 2, 5)
    plt.imshow(original_image3)
    plt.title("Gambar Asli 3")

    plt.subplot(3, 2, 6)
    plt.imshow(segmented_image3)
    plt.title("Gambar Tersegmentasi 3")

    plt.show()
else:
    print("Satu atau lebih gambar tidak ditemukan atau tidak bisa diproses.")