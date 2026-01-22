from tokenizers import Tokenizer, pre_tokenizers, decoders
import torch
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset
import pandas as pd
import os


class BPETokenizer(Dataset):
    def __init__(self, config):
        self.unk_word = '<UNK>'
        self.stop_word = '<EOS>'
        self.start_word = '<BOS>'
        self.pad_word = '<PAD>'
        self.mask_word = '<MASK>'
        self.df = None
        self.max_seq_len = config.max_seq_len
        if config.mode == 'Train':
            self.df = pd.read_csv(config.data_file_path, nrows=500)
        elif config.mode == 'Eval':
            self.df = pd.read_csv(config.data_file_path, skiprows=500, nrows=2)
        elif config.mode == 'Test':
            self.df = pd.read_csv(config.data_file_path, skiprows=1000, nrows=2)
        if not os.path.exists(config.tokenizer_path):
            self.tokenizer = Tokenizer(BPE(unk_token=self.unk_word))
            self.trainer = BpeTrainer(vocab_size=config.vocab_size, special_tokens=[self.unk_word, self.stop_word, self.start_word, self.pad_word, self.mask_word])
            self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
            self.tokenizer.decoder = decoders.Metaspace()
            self.tokenizer.train_from_iterator(self.df['Text'], self.trainer)
            self.tokenizer.save(config.tokenizer_path)
        else:
            self.tokenizer = Tokenizer.from_file(config.tokenizer_path)
            total_seq_data = []
            for line in self.df['Text']:
                line_tokens = self.tokenizer.encode(line).ids
                for i in range(0, len(line_tokens)-self.max_seq_len-1, self.max_seq_len):
                    total_seq_data.append(line_tokens[i:self.max_seq_len+i+1]) # len of each sample as config.max_seq_len + 1
            self.df = total_seq_data[:]
            del total_seq_data
        self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        self.start_token = self.tokenizer.token_to_id(self.start_word)
        self.stop_token = self.tokenizer.token_to_id(self.stop_word)
        self.pad_token = self.tokenizer.token_to_id(self.pad_word)
        self.unk_token = self.tokenizer.token_to_id(self.unk_word)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        current_data = self.df[idx]
        x = current_data[:self.max_seq_len]
        y = current_data[1:self.max_seq_len+1]
        mask_tokes_count = 0
        if len(x)<self.max_seq_len:
            mask_tokes_count = self.max_seq_len - len(x)
            x += [self.pad_token] * mask_tokes_count
            y += [self.pad_token] * mask_tokes_count
        x = torch.tensor(x)
        y = torch.tensor(y)
        mask = ~(x == self.pad_token)
        return x,  y, mask
    
    def decode(self, ids):
        text = self.tokenizer.decode(ids)
        return text
    
    def encode(self, text):
        ids = self.tokenizer.encode(text).ids
        return ids
        
        





if __name__ == '__main__':
    from config import ModelArgs
    ModelArgs.mode = 'Train'
    tokenizer = BPETokenizer(ModelArgs)
    print(tokenizer.vocab_size)
    import pdb; pdb.set_trace()


