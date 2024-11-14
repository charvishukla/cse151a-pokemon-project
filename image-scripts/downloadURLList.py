import aiohttp
import asyncio
import csv
from pathlib import Path

INPUT_CSV = 'TCG-image-urls.csv'
FAILED_CSV = 'failed_downloads.csv'
OUTPUT_FOLDER = 'images'
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)


# Function to download a single image
async def download_image(session, image_id, url):
    filename = Path(OUTPUT_FOLDER) / f"{image_id}.png"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, 'wb') as f:
                    f.write(await response.read())
                return True, image_id, url  # Success
            else:
                return False, image_id, url  # Failed
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False, image_id, url  # Failed

# Main function to handle asynchronous downloads
async def download_images(csv_file):
    # Read CSV file for URLs
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        urls = [(row['id'], row['url']) for row in reader]

    # Track failed downloads
    failed_downloads = []

    # Run downloads concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, image_id, url) for image_id, url in urls]
        results = await asyncio.gather(*tasks)

        # Record failed downloads
        for success, image_id, url in results:
            if not success:
                failed_downloads.append((image_id, url))

    # Write failed downloads to CSV
    with open(FAILED_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'url'])
        for image_id, url in failed_downloads:
            writer.writerow([image_id, url])

# Run the async function
asyncio.run(download_images(INPUT_CSV))