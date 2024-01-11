import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_tgt, src_lang, tgt_lang, seq_len, sliding_window):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sliding_window = sliding_window

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
    
     

        
        tgt_text = self.ds[idx]['content']
        # src_text = src_target_pair[self.src_lang]
        # tgt_text = src_target_pair[self.tgt_lang]
       
        

       
     

        # # Transform the text into tokens
        # enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        dec_input_tokens = dec_input_tokens[:self.seq_len]

        dec_num_padding_tokens = 0

        if len(dec_input_tokens) == 0:
        # Return None or any appropriate value to indicate skipping
            dec_num_padding_tokens = self.seq_len-1
        else:
        # # Add sos, eos and padding to each sentence
        # enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) 

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # # Add <s> and </s> token
        # encoder_input = torch.cat(
        #     [
        #         self.sos_token,
        #         torch.tensor(enc_input_tokens, dtype=torch.int64),
        #         self.eos_token,
        #         torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        #     ],
        #     dim=0,
        # )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens[:-1], dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens[1:], dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

       
        # # Double check the size of the tensors to make sure they are all seq_len long
        # assert encoder_input.size(0) == self.seq_len
        # print(f'decoder_input size: {decoder_input.size(0)} and seq {len(dec_input_tokens)}')
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        return {
            # 'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            # "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)) & sliding_mask(decoder_input.size(0),self.sliding_window), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
             
            # "src_text": src_text,
            "tgt_text": tgt_text,
        }

   
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
def sliding_mask(size, sliding_window):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=-(sliding_window-1)).type(torch.int)
    return mask == 1    
        