from PIL import Image #USE PILLOW-SIMD, REGULAR IS SLOW
from pathlib import Path
from multiprocessing import Pool
import time

IMAGES_FOLDER = "./images"
CROPPED_FOLDER = "./cropped"

def clean():
    cropped_path = Path(CROPPED_FOLDER)
    cropped_path.mkdir(exist_ok=True)
    for file in cropped_path.glob("*.png"):
        file.unlink()

cropped_folder = Path(CROPPED_FOLDER)
cropped_folder.mkdir(exist_ok=True)

def process_image(args):
    image_path, index = args
    image = Image.open(image_path)
    x, y = image.size
    cropped = image.crop((0, 0, x, y * 0.10)) # Crop the top 10% of the image
    # cropped = image.crop((x - (x * 0.15), 0, x, y * 0.10)) CORNER CROP
    cropped.save(cropped_folder / image_path.name, compress_level=0)
    return index  # Return the index to track progress

if __name__ == "__main__":
    images_folder_path = Path(IMAGES_FOLDER)

    if not images_folder_path.exists():
        print(f"Path does not exist: {IMAGES_FOLDER}")
        quit()

    images = list(images_folder_path.glob("*.png"))
    clean()

    start_time = time.time()
    total_images = len(images)
    
    with Pool() as pool:
        # Use imap_unordered for iteration over results as they complete
        completed = 0
        for _ in pool.imap_unordered(process_image, [(image, i) for i, image in enumerate(images)]):
            completed += 1
            if completed % 1000 == 0:
                rate = completed // (time.time()-start_time)
                print(f"Processed {completed}/{total_images} images ({(completed/total_images)*100}%)")
                print(f"Current rate: {rate} images/sec")

    print(f"\nTotal images processed: {total_images}")
    print(f"Rate: {total_images // (time.time() - start_time)} images/sec")