from tokenizers import Tokenizer, models, pre_tokenizers, trainers

class TokenizerManager:
    def __init__(self, kmer_size, vocab_size=None):
        self.kmer_size = kmer_size
        self.vocab_size = vocab_size

    def setup_kmer_tokenizer(self, sequences, kmers):
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        vocab = {kmer: i for i, kmer in enumerate(kmers)}
        special_token_ids = {token: len(vocab) + i for i, token in enumerate(special_tokens)}
        vocab.update(special_token_ids)

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        for token in special_tokens:
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"{token} token was not properly initialized in the tokenizer.")
            print(f"{token} token index: {token_id}")

        return tokenizer

    def save_tokenizer(self, tokenizer, save_path):
        tokenizer.save(save_path)

    def load_kmer_tokenizer(self, load_path):
        tokenizer = Tokenizer.from_file(load_path)

        if tokenizer.token_to_id("[UNK]") is None:
            raise ValueError("Loaded tokenizer does not recognize the [UNK] token.")

        return tokenizer

    def bpe_encode_sequences_genes(self, sequences, tokenizer):
        encoded_sequences = []
        for gene in sequences:
            encoded_sequence = []
            for kmer in gene:
                encoded = tokenizer.encode(kmer).ids
                unk_id = tokenizer.token_to_id("[UNK]")
                if unk_id in encoded:
                    print(f"[UNK] token detected in the encoded sequence")
                    encoded = [tokenizer.token_to_id("[UNK]")]  # Wrap the int in a list
                    encoded_sequence.extend(encoded)
                else:
                    encoded_sequence.extend(encoded)
            encoded_sequences.append(encoded_sequence)
        return encoded_sequences
