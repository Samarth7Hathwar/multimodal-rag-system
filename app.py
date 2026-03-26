from PIL import Image

def load_image(path):
    try: 
        image = Image.open(path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def main():
    image_path = "images/test.jpg"
    image = load_image(image_path)

    if image:
        print("Image loaded successfully!")
        print(f"Format : {image.format}")
        print(f"Size : {image.size}")
        print(f"Mode : {image.mode}")
        image.show() # Opens image
    else: 
        print("Failed to load image")

if __name__ == "__main__":
    main()