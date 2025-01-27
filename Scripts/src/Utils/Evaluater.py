import json
import os
import torch
import numpy as np
from .Logger_Manager import LoggerManager
from .Metric_Calculator import MetricsCalculator
from .Model_Manager import ModelManager
from .Dataset import SeqDataset, collate_fn
from torch.utils.data import DataLoader


class Evaluater:
    def __init__(self, args, sequence_processor, tokenizer_manager, gene_manager):
        self.args = args
        self.sequence_processor = sequence_processor
        self.tokenizer_manager = tokenizer_manager
        self.gene_manager = gene_manager

    def tokenize_sets(self, prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, args, fold):
        encoded_sequences = []
        gene_mapping = {}
        token_to_gene = {}  # Add reverse mapping

        for index, sequence in enumerate(prepped_seqs):
            encoded_sequence = self.tokenizer_manager.encode_sequences_genes(sequence, tokenizer)
            combined_list = [item for sublist in encoded_sequence for item in sublist]
            encoded_sequences.append([combined_list])
            genes_in_this_isolate = genes_list[index]

            # Create mappings
            for token, gene in zip(encoded_sequence, genes_in_this_isolate):
                token_id = token[0]
                if token_id == 0:  # Skip padding tokens
                    continue

                # Gene to token mapping
                if gene not in gene_mapping:
                    gene_mapping[gene] = []
                if token_id not in gene_mapping[gene]:
                    gene_mapping[gene].append(token_id)

                # Token to gene mapping
                token_to_gene[token_id] = {
                    'gene': gene,
                    'is_intergenic': gene.endswith('_ir_before') or gene.endswith('_ir_after')
                }

        # Save both mappings
        mappings = {
            'gene_to_token': gene_mapping,
            'token_to_gene': token_to_gene
        }

        return list(zip(encoded_sequences, prepped_labels, seq_ids))

    def create_dataset(self, zipped_data, classification_type):
        seqs_ = []
        labs = []
        seq_ids = []
        for seqs, label, seq_id in zipped_data:
            for seq in seqs:
                seqs_.append(seq)
                labs.append(label)
                seq_ids.append(seq_id)
        return SeqDataset(seqs_, labs, seq_ids, classification_type)

    def evaluate(self, zipped_data, mode):
        args = self.args
        sequence_processor = self.sequence_processor
        tokenizer_manager = self.tokenizer_manager
        metrics_calculator = MetricsCalculator()

        target_format = "binary" if "multi-cat" not in args.antibiotic else "multi-cat"

        save_path = os.path.join(args.save_path, args.antibiotic)

        tokenizer_path = f"{args.save_path}/{args.antibiotic}/tokenizer.json"
        print(tokenizer_path)
        tokenizer = tokenizer_manager.load_tokenizer(tokenizer_path)

        sequences = [seq for seq, _, _, _ in zipped_data]
        labels = [label for _, label, _, _ in zipped_data]
        seq_ids = [seq_id for _, _, seq_id, _ in zipped_data]
        genes_list = [genes for _, _, _, genes in zipped_data]

        unique_n_mers, prepped_seqs, prepped_labels = sequence_processor.extract_and_prep_genes(sequences, labels)
        test_data = self.tokenize_sets(prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, args, 0)
        print(test_data[0])
        test_dataset = self.create_dataset(test_data, target_format)

        batch_size = args.batch_size
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, classification_type=target_format),
            num_workers=0,
            pin_memory=True,
        )

        for fold in range(5):
            model_path = os.path.join(save_path, f"best_model_fold_{fold + 1}.pth")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            config = json.load(open(args.model_config))
            vocab_size = len(tokenizer.get_vocab())
            model_manager = ModelManager(vocab_size=vocab_size, config=config)
            model, threshold = model_manager.load_model(model_path, device_ids=[0])
            best_epoch, best_metric, best_precision, best_recall = 0, 0, 0, 0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            num_gpus = torch.cuda.device_count()
            device_ids = list(range(num_gpus))

            logger = LoggerManager(args.antibiotic, fold + 1, save_path, train=False)
            logger.log("Evaluating on Test Set...")

            model.eval()
            all_predictions = []
            all_probabilities = []
            all_labels = []

            with torch.no_grad():
                for sequence, attention_masks, labels, seq_ids in test_loader:
                    sequence, attention_masks, labels = (
                        sequence.to(device),
                        attention_masks.to(device),
                        labels.to(device),
                    )
                    outputs, _ = model(sequence, mask=attention_masks)
                    outputs = outputs.squeeze(1)
                    probabilities = torch.sigmoid(outputs)
                    probabilities = np.array(probabilities.cpu())
                    if isinstance(threshold, (float, int)):
                        predictions = (probabilities >= threshold).astype(int)
                    else:
                        predictions = (probabilities >= threshold[np.newaxis, :]).astype(int)
                    all_predictions.extend(predictions)
                    all_probabilities.extend(probabilities)
                    all_labels.extend(labels.cpu().numpy())

            mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            peak_mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

            logger.log(f"GPU Memory Allocated: {mem_allocated:.2f} MB")
            logger.log(f"GPU Memory Reserved: {mem_reserved:.2f} MB")
            logger.log(f"Peak GPU Memory Allocated: {peak_mem_allocated:.2f} MB")
            logger.log(f"Peak GPU Memory Reserved: {peak_mem_reserved:.2f} MB")

            f1, accuracy, hamming, jaccard, precision, recall, auc, confusion, class_report = metrics_calculator.calculate_metrics(
                all_labels, all_predictions, all_probabilities, target_format
            )

            metrics_calculator.print_eval_metrics(
                logger,
                accuracy,
                f1,
                best_metric,
                best_epoch,
                hamming,
                jaccard,
                precision,
                recall,
                best_precision,
                best_recall,
                auc,
                confusion,
                class_report,
            )