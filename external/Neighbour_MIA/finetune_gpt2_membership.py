"""??? GPT-2 fine-tune??? Neighbour_MIA membership analysis?

??? train_members.txt ???? member ???? target model ??????
?????????????????????????
"""

import os
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

BASE_MODEL = '/root/autodl-tmp/models/gpt2'
TRAIN_FILE = '/root/autodl-tmp/Neighbour_MIA/data/train_members.txt'
OUTPUT_DIR = '/root/autodl-tmp/Neighbour_MIA/finetuned_gpt2'


class TextDataset(Dataset):
    """???????? GPT-2 causal LM ?????"""

    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 128):
        self.examples = []
        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids = encoded['input_ids'][0]
            attention_mask = encoded['attention_mask'][0]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    random.seed(7)
    torch.manual_seed(7)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    # ???? member ?????????????????? membership signal?
    train_texts = texts * 4
    random.shuffle(train_texts)

    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
    model.train()

    dataset = TextDataset(train_texts, tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    epochs = 2
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
            global_step += 1
            if global_step % 25 == 0:
                print(f'epoch={epoch+1} step={global_step} loss={loss.item():.4f}')
        print(f'epoch={epoch+1} mean_loss={total_loss / max(1, len(loader)):.4f}')

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f'saved finetuned model to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
