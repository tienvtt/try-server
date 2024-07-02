from PIL import Image

try:
    image = Image.open('/mnt/d/tienvo/dataset/train/normal/video_569/frame0000.jpg')
    image.show()
except Exception as e:
    print(f"Error: {e}")
