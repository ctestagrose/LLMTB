from tokenizers import Tokenizer, models, pre_tokenizers

class TokenizerManager:
    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size

    def setup_tokenizer(self, genes):
        """
        Sets up the gene tokenizer

        :param genes: a list of all unique gene/intergenic region sequences
        :return: tokenizer object
        """
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        vocab = {gene: i for i, gene in enumerate(genes)}
        special_token_ids = {token: len(vocab) + i for i, token in enumerate(special_tokens)}
        vocab.update(special_token_ids)

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        for token in special_tokens:
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"{token} token was not properly initialized in the tokenizer.")

        return tokenizer

    def save_tokenizer(self, tokenizer, save_path):
        """
        Saves a tokenizer object

        :param tokenizer: the tokenizer object
        :param save_path: the path where the tokenizer will be saved
        """
        tokenizer.save(save_path)

    def load_tokenizer(self, load_path):
        """
        Loads a predefined tokenizer object

        :param load_path: location of the saved tokenizer
        :return: returns the loaded tokenizer object
        """
        tokenizer = Tokenizer.from_file(load_path)

        if tokenizer.token_to_id("[UNK]") is None:
            raise ValueError("Loaded tokenizer does not recognize the [UNK] token.")

        return tokenizer

    def encode_sequences_genes(self, sequences, tokenizer):
        """
        Takes an isolate's loaded gene/intergenic region sequences and encodes/tokenizes them

        :param sequences: The loaded isolate gene/intergenic sequences
        :param tokenizer: The tokenizer object
        :return: returns an encoded isolate sequence
        """

        encoded_sequences = []
        for genes in sequences:
            encoded_sequence = []
            for gene in genes:
                encoded = tokenizer.encode(gene).ids
                unk_id = tokenizer.token_to_id("[UNK]")
                if unk_id in encoded:
                    encoded = [tokenizer.token_to_id("[UNK]")]
                    encoded_sequence.extend(encoded)
                else:
                    encoded_sequence.extend(encoded)
            encoded_sequences.append(encoded_sequence)
        return encoded_sequences