from itertools import islice
from datasets import load_dataset
import aiohttp
from PIL import Image
import io
import asyncio
import torchvision.transforms as T
import torch

class CustomDataLoader:
    def __init__(self, clip_text_tokenizer, image_size=224) -> None:
        self.tokenizer = clip_text_tokenizer
        self.dataset = load_dataset("laion/laion400m", split="train", streaming=True)
        self.ready_data = []
        self.device = "cuda"
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_batch(self, instances):
        return asyncio.run(self._get_batch_async(instances))
    
    async def _get_batch_async(self, instances):
        raw_data = list(islice(self.dataset, instances * 2))
        raw_data = [(x["url"], x["caption"]) for x in raw_data]
        batch = await self._process_raw_batch(raw_data)
        self.ready_data.extend(batch)
        while len(self.ready_data) < instances:
            raw_data = list(islice(self.dataset, instances * 2))
            raw_data = [(x["url"], x["caption"]) for x in raw_data]
            batch = await self._process_raw_batch(raw_data)
            self.ready_data.extend(batch)
            print(f"Iterating again. Only {len(self.ready_data)} instances and need {instances} instances.")
        final = self.ready_data[:instances]
        self.ready_data = self.ready_data[instances:]

        img_tensors = torch.stack([img for (img, tok_dict) in final], dim=0).to(self.device)
        text_tensors = self._collate_token_dicts([tok_dict for (img, tok_dict) in final])

        return img_tensors, text_tensors

    async def _process_raw_batch(self, batch):
        resolve_imgs_task = asyncio.create_task(self._resolve_images(batch))
        tokenize_task = asyncio.create_task(self._tokenize_captions(batch))
        img_tensors = await resolve_imgs_task
        caption_tokens = await tokenize_task
        processed = []
        for img_tensor, tokens in zip(img_tensors, caption_tokens):
            if img_tensor is not None and tokens is not None:
                processed.append((img_tensor, tokens))
        return processed

    async def _resolve_images(self, batch):
        connector = aiohttp.TCPConnector(limit=1024)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._fetch(session, url) for url, _ in batch
            ]
            byte_results = await asyncio.gather(*tasks)
        tensors = []
        for bytestr in byte_results:
            if bytestr is None:
                tensors.append(None)
                continue
            tensor = self._decode_to_tensor(bytestr)
            tensors.append(tensor)
        return tensors

    def _decode_to_tensor(self, image_bytes: bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(img)
            return tensor
        except Exception:
            return None

    async def _fetch(self, session, url):
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                return data
        except Exception:
            return None

    async def _tokenize_captions(self, batch):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [
                self.tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=32,
                    return_tensors="pt"
                ) for (_, caption) in batch
            ]
        )
    
    def _collate_token_dicts(self, token_dicts):
        # Tokenizer is outputing {'input_ids':(1,L), 'attention_mask': (1,L)}
        # Tokenize caption returns a list of these.
        batch = {}
        for key in token_dicts[0].keys():
            batch[key] = torch.cat([d[key] for d in token_dicts], dim=0).to(self.device) # turn into (batch, L)
        return batch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.model_utils import get_text_transformer_model
import time

tokenizer, _ = get_text_transformer_model("sentence-transformers/all-MiniLM-L6-v2")

dataloader = CustomDataLoader(tokenizer)
start = time.time()
i, t = dataloader.get_batch(512)
end = time.time()
print(end - start)
