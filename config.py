from dataclasses import dataclass
@dataclass
class ModelArgs:
    dim: int = 512
    n_layer: int = 8
    n_head: int = 8 # for queries
    n_kv_head: int = 4
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: int = int((2/3)*4)
    norm_eps: float = 1e-5
    rope_base = 10000
    assert dim % n_head == 0
    head_dim = dim//n_head
    dropout: float = 0.1


    # for kv cache
    max_batch_size: int = 16
    max_seq_len: int = 256
    device: str = None
    mode: str = None

    # data
    data_file_path = '/Users/dvidyasagar/Desktop/code/NN/datasets/gutenberg_data_modified.csv'
    tokenizer_path = 'bpe_tokenizer.json'

    # training
    num_epochs = 20
    lr = 1e-5
    model_folder = "./checkpoints"