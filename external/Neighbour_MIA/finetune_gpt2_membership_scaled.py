"""100/200 scale GPT-2 fine-tune for membership evaluation.

??? member ???? target model?????????????????? membership signal?
"""
import json, os, random
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

BASE_MODEL = '/root/autodl-tmp/models/gpt2'
DATA_DIR = '/root/autodl-tmp/Neighbour_MIA/data'
META_FILE = os.path.join(DATA_DIR, 'membership_eval_scale.json')
with open(META_FILE, 'r', encoding='utf-8') as f:
    META = json.load(f)
SCALE = int(META['scale'])
TRAIN_FILE = os.path.join(DATA_DIR, f'train_members_{SCALE}.txt')
OUTPUT_DIR = f'/root/autodl-tmp/Neighbour_MIA/finetuned_gpt2_{SCALE}'
LOG_META = f'/root/autodl-tmp/Neighbour_MIA/finetuned_gpt2_{SCALE}_training_meta.json'

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, max_length: int = 128):
        self.examples = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
            labels = enc['input_ids'][0].clone()
            labels[enc['attention_mask'][0] == 0] = -100
            self.examples.append({'input_ids': enc['input_ids'][0], 'attention_mask': enc['attention_mask'][0], 'labels': labels})
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def main():
    random.seed(13); torch.manual_seed(13)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 2 if torch.cuda.is_available() else 1
    epochs = int(META.get('epochs', 3))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        texts = [x.strip() for x in f if x.strip()]
    # ???????????????? 3 epoch ???? membership signal?
    train_texts = texts[:]
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL).to(device)
    model.train()
    loader = DataLoader(TextDataset(train_texts, tokenizer), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    losses = []
    step = 0
    print(f'scale={SCALE} device={device} batch_size={batch_size} epochs={epochs} train_items={len(train_texts)}')
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            total += float(loss.item()); losses.append(float(loss.item()))
            step += 1
            if step % 50 == 0:
                print(f'epoch={epoch+1} step={step} loss={loss.item():.4f}')
        print(f'epoch={epoch+1} mean_loss={total/max(1,len(loader)):.4f}')
    model.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR)
    meta = {'scale': SCALE, 'train_items': len(train_texts), 'epochs': epochs, 'batch_size': batch_size, 'device': str(device), 'final_loss': losses[-1], 'mean_loss': sum(losses)/len(losses), 'output_dir': OUTPUT_DIR}
    with open(LOG_META, 'w', encoding='utf-8') as f: json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
