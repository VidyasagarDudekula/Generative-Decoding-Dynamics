from dataclasses import dataclass
@dataclass
class ModelArgs:
    dim: int = 1024
    n_layer: int = 8
    n_head: int = 16 # for queries
    n_kv_head: int = 8
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: int = 4
    norm_eps: float = 1e-5
    rope_base = 10000
    assert dim % n_head == 0
    head_dim = dim//n_head
    dropout: float = 0.1


    # for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 1024
    device: str = None
    mode: str = None

    # data
    data_file_path = '/scratch/vdudekul/checkpoints/datasets_collections/gutenberg_data_modified.csv'
    tokenizer_path = 'bpe_tokenizer.json'

    # training
    num_epochs = 50
    lr = 1e-5
    model_folder = "/scratch/vdudekul/checkpoints/llama2-train-2"