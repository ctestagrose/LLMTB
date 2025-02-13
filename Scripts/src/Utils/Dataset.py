import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random

class SeqDataset(Dataset):
    def __init__(self, sequences, labels, seq_ids, classification_type='multi'):
        self.sequences = sequences
        self.labels = labels
        self.seq_ids = seq_ids
        self.classification_type = classification_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        seq_id = self.seq_ids[idx]
        if self.classification_type == 'binary':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.long)
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': label,
            'seq_id': seq_id
        }

def dynamic_masking(inputs, mask_prob=0.15, mask_token_id=0, vocab_size=None, max_span_length=5, max_attempts=100):
    batch_size, seq_len = inputs.size()
    num_masked_tokens = int(mask_prob * seq_len)

    masked_inputs = inputs.clone()

    for i in range(batch_size):
        spans = []
        total_masked = 0
        attempts = 0

        while total_masked < num_masked_tokens and attempts < max_attempts:
            span_length = random.randint(1, max_span_length)
            start_idx = random.randint(0, seq_len - span_length)
            end_idx = min(start_idx + span_length, seq_len)

            if torch.any(masked_inputs[i, start_idx:end_idx] == 0):
                attempts += 1
                continue

            if total_masked + (end_idx - start_idx) > num_masked_tokens:
                end_idx = start_idx + (num_masked_tokens - total_masked)

            spans.append((start_idx, end_idx))
            total_masked += (end_idx - start_idx)
            attempts += 1


        for start_idx, end_idx in spans:
            mask_choice = torch.rand(end_idx - start_idx, device=inputs.device)
            for j in range(start_idx, end_idx):
                if mask_choice[j - start_idx] < 0.8:
                    masked_inputs[i, j] = mask_token_id
                elif mask_choice[j - start_idx] < 0.9 and vocab_size is not None:
                    masked_inputs[i, j] = torch.randint(1, vocab_size, (1,), device=inputs.device)

    mask = (masked_inputs != inputs)

    return masked_inputs, mask


# Collate function for data loader
def collate_fn(batch, classification_type='multi', mask_prob=0.20, MASK_TOKEN=1, PAD_TOKEN=0, VOCAB_SIZE=None):
    sequences = [item['sequence'] for item in batch]
    labels = [item['label'] for item in batch]
    seq_ids = [item['seq_id'] for item in batch]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_TOKEN)
    attention_masks = torch.zeros_like(sequences_padded, dtype=torch.bool)
    attention_masks[sequences_padded != 0] = 1
    masked_inputs, dynamic_mask = dynamic_masking(sequences_padded, mask_prob, mask_token_id=MASK_TOKEN,
                                                  vocab_size=VOCAB_SIZE)

    attention_masks = attention_masks & (~dynamic_mask)

    if classification_type == 'binary':
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = torch.stack(labels)

    return masked_inputs, attention_masks, labels, seq_ids
