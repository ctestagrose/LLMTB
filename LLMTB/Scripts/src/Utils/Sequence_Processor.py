import random

class SequenceProcessor:
    def __init__(self, kmer_size, stride):
        self.kmer_size = kmer_size
        self.stride = stride

    def handle_ns(self, sequence, strategy="nothing"):
        if strategy == "trim":
            return sequence.replace("N", "")
        elif strategy == "filter":
            return sequence if sequence.count('N')/len(sequence) < 0.20 else "XXXXXXXXXXXXXXXXXX"
        elif strategy == "substitute":
            return ''.join(random.choice('ATCG') if nucleotide == 'N' else nucleotide for nucleotide in sequence)
        elif strategy == "mask":
            return sequence.replace('N', 'X')
        elif strategy == "nothing":
            return sequence
        else:
            raise ValueError("Unsupported strategy")

    def extract_and_prep_genes(self, sequences, labels):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        unique_mers = set()
        prepped_seqs = []
        prepped_labels = []
        minimizer_size = 0
        
        for item, label in zip(sequences, labels):
            prepped_item = []
            for sequence in item:
                prepped_seq = []
                if "N" in sequence:
                    sequence = self.handle_ns(sequence)
                if minimizer_size > 0:
                    rev = ''.join(complement.get(nuc, '') for nuc in reversed(sequence))
                    L = len(sequence)
                    for i in range(0, L - self.kmer_size + 1, self.stride):
                        sub_f = sequence[i:i + self.kmer_size]
                        sub_r = rev[L - self.kmer_size - i:L - i]
                        min_f, min_r = "ZZZZZZZZZZZZZ", "ZZZZZZZZZZZZZ"
                        for j in range(self.kmer_size - minimizer_size + 1):
                            sub2_f = sub_f[j:j + minimizer_size]
                            sub2_r = sub_r[j:j + minimizer_size]
                            if sub2_f < min_f:
                                min_f = sub2_f
                            if sub2_r < min_r:
                                min_r = sub2_r
                        minimizer = min(min_f, min_r)
                        prepped_seq.append(minimizer)
                        unique_mers.add(minimizer)
                else:
                    pos = 0
                    while pos <= (len(sequence) - (self.kmer_size - 1)):
                        mer = sequence[pos:pos + self.kmer_size]
                        prepped_seq.append(mer)
                        unique_mers.add(mer)
                        pos += self.stride
                    if pos < len(sequence):
                        mer = sequence[pos:]
                        prepped_seq.append(mer)
                        unique_mers.add(mer)
                prepped_item.append(prepped_seq)
            prepped_seqs.append(prepped_item)
            prepped_labels.append(label)
        return unique_mers, prepped_seqs, prepped_labels