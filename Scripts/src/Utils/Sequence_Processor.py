import random

class SequenceProcessor:
    def __init__(self, items=[]):
        self.l = items

    def handle_ns(self, sequence, strategy="nothing"):
        if strategy == "trim":
            return sequence.replace("N", "")
        elif strategy == "filter":
            return sequence if sequence.count('N') / len(sequence) < 0.20 else "XXXXXXXXXXXXXXXXXX"
        elif strategy == "substitute":
            return ''.join(random.choice('ATCG') if nucleotide == 'N' else nucleotide for nucleotide in sequence)
        elif strategy == "mask":
            return sequence.replace('N', 'X')
        elif strategy == "nothing":
            return sequence
        else:
            raise ValueError("Unsupported strategy")

    def extract_and_prep_genes(self, sequences, labels):
        unique_mers = set()
        prepped_seqs = []
        prepped_labels = []

        for item, label in zip(sequences, labels):
            prepped_item = []
            for sequence in item:
                prepped_seq = []
                if "N" in sequence:
                    sequence = self.handle_ns(sequence)
                prepped_seq.append(sequence)
                unique_mers.add(sequence)
                prepped_item.append(prepped_seq)
            prepped_seqs.append(prepped_item)
            prepped_labels.append(label)
        return unique_mers, prepped_seqs, prepped_labels