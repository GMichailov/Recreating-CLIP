from itertools import islice
from datasets import load_dataset
import aiohttp
from PIL import Image
import io
import asyncio
import csv
import os

class DataDownloader:
    def __init__(self, total_size, batch_size, save_location) -> None:
        self.total_size = total_size
        self.batch_size = batch_size
        self.processed = 0
        self.dataset = load_dataset("laion/laion400m", split="train", streaming=True)
        self.dataset_iterator = iter(self.dataset)
        self.save_location = save_location
        self.csv_path, self.images_path = self._create_storage_file()
        self.identifier_length = self._get_identifier_length()

    def _create_storage_file(self):
        os.makedirs(self.save_location, exist_ok=True)
        csv_path = os.path.join(self.save_location, "dataset.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "caption"])
        images_path = os.path.join(self.save_location, "images")
        os.makedirs(images_path, exist_ok=True)
        return csv_path, images_path

    def _get_identifier_length(self):
        length = 0
        total_size = self.total_size
        while total_size > 0:
            total_size //= 10
            length += 1
        return length

    def get_dataset(self):
        return asyncio.run(self._get_dataset_async())
    
    async def _get_dataset_async(self):
        connector = aiohttp.TCPConnector(limit=256)
        async with aiohttp.ClientSession(connector=connector) as session:
            while self.processed < self.total_size:
                raw_data = list(islice(self.dataset_iterator, self.batch_size))
                raw_data = [(x["url"], x["caption"]) for x in raw_data] # type: ignore
                batch = await self._resolve_images(session, raw_data)
                self._save_batch(batch)

    async def _resolve_images(self, session, batch):
        tasks = [
            self._fetch_and_process(session, url, caption) for url, caption in batch
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _fetch_and_process(self, session, url, caption):
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status != 200:
                    return None
                ctype = resp.headers.get("Content-Type", "")
                if "image/webp" in ctype.lower():
                    return None
                data = await resp.read()
        except Exception:
            return None
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return (img, caption)
        except Exception:
            return None

    def _save_batch(self, batch):
        batch = self._remove_missing_images_add_identifier(batch)
        self._save_imgs(batch)
        self._save_to_csv(batch)
        self.processed += len(batch)
        
    def _save_to_csv(self, batch):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for _, identifier, caption in batch:
                prepend = "0" * (self.identifier_length - len(str(identifier)))
                name = f"{prepend}{identifier}.jpg"
                writer.writerow([
                    name,
                    caption
                ])

    def _save_imgs(self, batch):
        for img, identifier, _ in batch:
            prepend = "0" * (self.identifier_length - len(str(identifier)))
            name = f"{prepend}{identifier}.jpg"
            save_path = os.path.join(self.images_path, name)
            img.save(save_path, "JPEG", quality=95)

    def _remove_missing_images_add_identifier(self, batch):
        batch = [pair for pair in batch if pair is not None and not isinstance(pair, Exception)]
        batch = [(img, self.processed + idx, caption) for idx, (img, caption) in enumerate(batch)]
        return batch


if __name__ == "__main__":
    data_downloader = DataDownloader(8, 4, os.path.dirname(os.path.abspath(__file__)))
    data_downloader.get_dataset()
    os._exit(0)