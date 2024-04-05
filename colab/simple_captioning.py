import torch, math
from torch.utils.data import DataLoader
from torch import nn, Tensor
from tokenizers import Tokenizer
from transformers import AutoProcessor, CLIPModel
import datasets
import matplotlib.pyplot as plt

### Let's build a simple image captioning model
### Step 1: Define a Transformer model that decodes an image embedding (CLIP) into a text description
### Step 2: Load the MSCOCO dataset and pre-compute the image and text embeddings
### Step 3: Train the model
###
### While this being a very minimal example, many improvements are possible:
### pretraining, weight tying, encoder network, frozen LLM, PEFT, fast attention,
### model compilation, loss function(s), speculative decoding...


## Step 1: Define the model

class SimpleImageCaptioner(nn.Module):
    def __init__(self,
                 num_decoder_layers: int, emb_size: int, nhead: int,
                 tokenizer: Tokenizer, max_seq_len: int=100,
                 dropout: float=0.1):
        super(SimpleImageCaptioner, self).__init__()
        self.emb_size = emb_size
        self.tgt_tok_emb = nn.Embedding(tokenizer.get_vocab_size(), emb_size)
        self.pos_emb = nn.Embedding(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead),
            num_layers=num_decoder_layers
        )
        self.generator = nn.Linear(emb_size, tokenizer.get_vocab_size())
        self.tokenizer = tokenizer
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_decoder_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, img_embs: Tensor, tgt_tokens: Tensor, tgt_mask: Tensor=None, tgt_key_padding_mask=None, tgt_out: Tensor=None):
        " This takes the image embeddings and the texts already generated and predict the next tokens "
        seq_len, batch_size = tgt_tokens.size(0), tgt_tokens.size(1)

        # Sum token and positional embeddings. Add dropout for regularization
        tgt_emb = self.tgt_tok_emb(tgt_tokens.long()) * math.sqrt(self.emb_size)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=tgt_emb.device)
        pos_emb = self.pos_emb(pos.unsqueeze(1).expand(-1, batch_size))
        tgt_emb = self.dropout(tgt_emb + pos_emb)

        # directly decode the image embeddings into texts
        outs = self.decoder(tgt_emb, img_embs, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # generate logits (+ loss if training)
        if tgt_out is not None:
            logits = self.generator(outs)
            loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        else:
            logits = self.generator(outs[[-1], :, :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, img_emb, max_len=50, start_token=101, end_token=102):
        " loop that runs one forward pass and pick the next word with the highest probability "
        img_emb = img_emb.unsqueeze(0) # batch of one
        seq = torch.ones(1, 1).fill_(start_token).type(torch.long).to(img_emb.device)
        for _ in range(max_len-1):
            logits, _ = self.forward(img_emb, seq)
            _, next_word = torch.max(logits[0][0], dim=0)
            seq = torch.cat([seq, next_word.unsqueeze(0).unsqueeze(0)], dim=0)
            if next_word == end_token:
                break

        return self.tokenizer.decode(seq.flatten().cpu().numpy().astype(int))


## Step 2: Load the dataset and pre-compute the embeddings for the images and the text

# use 5% of MSCOCO for a quick demo
dataset = datasets.DatasetDict({
    "train": datasets.load_dataset("HuggingFaceM4/COCO", split="train[:5%]", trust_remote_code=True),
    "validation": datasets.load_dataset("HuggingFaceM4/COCO", split="validation[:5%]", trust_remote_code=True),
})
# or full dataset: dataset = datasets.load_dataset("HuggingFaceM4/COCO", trust_remote_code=True)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
PAD_IDX = tokenizer.token_to_id("[PAD]")

def preprocess_embeddings(batch):
    " Pre-Generate img and text embeddings "
    # There are multiple caption samples per image. Uniqify before computing CLIP features
    images = {path: img for path, img in zip(batch['filepath'], batch['image'])}
    inputs = processor(images=list(images.values()), return_tensors="pt").to("cuda")
    features_map = {k: v for k, v in zip(images.keys(), clip_model.get_image_features(**inputs))}
    return {
        'img_emb': [[features_map[path]] for path in batch['filepath']],
        'tok_emb': [tokenizer.encode(s['raw']).ids for s in batch['sentences']],
    }
dataset = dataset.map(preprocess_embeddings, batched=True, batch_size=500)


## Step 3: Train the model

# Define helper functions for generating the training masks and batching data together
def create_mask(tgt):
    " Create causal mask for the tgt sequence. shape: (tgt_seq_len, tgt_seq_len). tgt_padding_mask indicates where the padding is."
    mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=tgt.shape[0]).to("cuda").float()
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).float()
    return mask, tgt_padding_mask

def collate_fn(batch):
    " Pad text sequence inside the batch . shape: (max_seq_len, batch_size, emb_size)"
    path_batch = [e['filepath'] for e in batch]
    img_batch = [e['image'] for e in batch]
    img_emb_batch = torch.Tensor([e['img_emb'] for e in batch]).transpose(0, 1)
    tgt_batch = torch.nn.utils.rnn.pad_sequence([torch.Tensor(e['tok_emb']) for e in batch], padding_value=PAD_IDX).long()
    tgt_str_batch = [e['sentences']['raw'] for e in batch]
    return path_batch, img_batch, img_emb_batch, tgt_batch, tgt_str_batch


# Init the model and start the training loop
model = SimpleImageCaptioner(num_decoder_layers=4, emb_size=512, nhead=8, tokenizer=tokenizer).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=0.00015, betas=(0.8, 0.98), eps=1e-9)
for epoch in range(1, 15):
    train_losses, val_losses = [], []
    # training
    model.train()
    for paths, imgs, img_embs, text_embs, texts in DataLoader(dataset['train'], batch_size=128, collate_fn=collate_fn):
        text_embs = text_embs.to("cuda")
        tgt_input = text_embs[:-1, :]
        tgt_mask, tgt_padding_mask = create_mask(tgt_input)
        outputs, loss = model(img_embs.to("cuda"), tgt_input, tgt_mask, tgt_padding_mask, text_embs[1:, :])

        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_losses.append(loss.item())

    # evaluation
    model.eval()
    for paths, imgs, img_embs, text_embs, texts in DataLoader(dataset['validation'], batch_size=128, collate_fn=collate_fn):
        img_embs, text_embs = img_embs.to("cuda"), text_embs.to("cuda")
        tgt_input = text_embs[:-1, :]
        tgt_mask, tgt_padding_mask = create_mask(tgt_input)
        outputs, loss = model(img_embs, tgt_input, tgt_mask, tgt_padding_mask, text_embs[1:, :])
        val_losses.append(loss.item())

    print(f"### Epoch {epoch}, Train Loss: {sum(train_losses) / len(train_losses):.3f}, Val Loss: {sum(val_losses) / len(val_losses):.3f}")


# display
for i, (paths, imgs, img_embs, text_embs, texts) in enumerate(DataLoader(dataset['validation'], batch_size=5, collate_fn=collate_fn)):
    if i > 5: break
    img_embs = img_embs.to('cuda')
    plt.axis('off'); plt.imshow(imgs[0]); plt.show()
    print(f"GT:{ texts[0] }\nPRED:{ model.generate(img_embs[0][0].unsqueeze(0)) }")