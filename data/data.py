# data.py
import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pickle

class TextDataset(Dataset):
    """
    A custom PyTorch dataset to handle tokenized text data.
    Converts raw text into token IDs using a vocabulary and provides
    sequences of length `seq_len` for training.

    Parameters:
    - texts: list of text lines (strings)
    - vocab: pre-built vocabulary (optional)
    - seq_len: length of each training sequence
    """
    def __init__(self, texts, vocab_path=None, seq_len=32, use_saved_vocab=False):
        self.seq_len = seq_len

        # Make sure vocab is saved inside the same folder as this file
        if vocab_path is None:
            current_dir = os.path.dirname(__file__)
            vocab_path = os.path.join(current_dir, "vocab.pkl")

        self.vocab_path = vocab_path

        # Add <bos> and <eos> tokens to each sentence
        texts = [f"<bos> {line.strip()} <eos>" for line in texts if line.strip()]
        self.tokens = [token for line in texts for token in line.split()]

        if use_saved_vocab and os.path.exists(self.vocab_path):
            with open(self.vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            self.vocab = self.build_vocab(self.tokens)
            with open(self.vocab_path, "wb") as f:
                pickle.dump(self.vocab, f)

        self.token_ids = [
            self.vocab['stoi'].get(token, self.vocab['stoi']['<unk>']) 
            for token in self.tokens
        ]

    def build_vocab(self, tokens):
        """
        Builds a vocabulary from the list of tokens.

        Returns:
        - vocab: a dict with 'stoi' (string-to-index) and 'itos' (index-to-string)
        """
        vocab = sorted(set(tokens))  # unique and sorted list of tokens
        stoi = {token: i + 4 for i, token in enumerate(vocab)}  # shift for reserved tokens
        stoi.update({'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3})
        itos = {i: token for token, i in stoi.items()}
        return {'stoi': stoi, 'itos': itos}

    def __len__(self):
        """
        Number of sequences that can be extracted from token_ids
        """
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx):
        """
        Retrieves a sequence of token IDs from index `idx` to `idx + seq_len`
        """
        chunk = self.token_ids[idx:idx + self.seq_len]
        return torch.tensor(chunk, dtype=torch.long)


def get_dataset(name="wikitext", subset="wikitext-2-raw-v1"):
    """
    Loads and returns a cleaned list of text lines from HuggingFace datasets.

    Parameters:
    - name: dataset name
    - subset: specific version of the dataset

    Returns:
    - List of non-empty text strings
    """
    dataset = load_dataset(name, subset)
    train_texts = [line for line in dataset['train']['text'] if line.strip()]
    return train_texts
