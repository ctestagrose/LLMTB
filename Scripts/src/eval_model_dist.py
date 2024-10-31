import json
import os
import numpy as np
import torch
from collections import defaultdict
from Utils.Logger_Manager import LoggerManager
from Utils.Metric_Calculator import MetricsCalculator
from Utils.Model_Manager import ModelManager
from Utils.Dataset import SeqDataset, collate_fn
from torch.utils.data import DataLoader


def _evaluate(data_loader, model_path, target_format, metrics_calculator, model_manager, args, fold):
    save_path = os.path.join(args.save_path, args.antibiotic)
    logger = LoggerManager(args.antibiotic, fold + 1, save_path, train=False)
    logger.log("Evaluating on Test Set...")
    best_epoch, best_metric, best_precision, best_recall = 0, 0, 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold = model_manager.load_model_threshold(model_path)
    model.to(device)
    model.eval()

    seq_predictions = defaultdict(list)
    seq_probabilities = defaultdict(list)
    seq_labels = {}
    token_importance_dict = {}
    with torch.no_grad():
        all_seq_ids = []
        all_predictions = []
        all_probabilities = []
        all_labels = []
        for sequence, attention_masks, labels, seq_ids in data_loader:
            sequence, attention_masks, labels = sequence.to(device), attention_masks.to(device), labels.to(device)
            outputs, attention_weights = model(sequence, mask=attention_masks)
            outputs = outputs.squeeze(1)
            avg_attn_weights = torch.mean(torch.stack([torch.mean(attn, dim=1) for attn in attention_weights]),
                                          dim=0)
            for batch_idx in range(sequence.size(0)):
                if sequence[batch_idx].size(0) == 1:
                    token = sequence[batch_idx][0].item()
                    token_importance = avg_attn_weights[batch_idx].squeeze()
                    token_importance_dict[token] = token_importance.item()
                else:
                    token_importance = torch.mean(avg_attn_weights[batch_idx], dim=0).squeeze()
                    sorted_importance, indices = torch.sort(token_importance, descending=True)
                    for idx, importance in zip(indices, sorted_importance):
                        token = sequence[batch_idx][idx].item()
                        if token == 0:
                            continue
                        if token in token_importance_dict:
                            token_importance_dict[token] += importance.item()
                        else:
                            token_importance_dict[token] = importance.item()
            probabilities = torch.sigmoid(outputs)
            threshold = torch.tensor(threshold).to(device) if not isinstance(threshold, torch.Tensor) else threshold
            predictions = (probabilities > threshold).long() if target_format != "multi_cat" else probabilities > 0.5
            all_seq_ids.extend(seq_ids)
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        for seq_id, prediction, label, probability in zip(all_seq_ids, all_predictions, all_labels, all_probabilities):
            seq_predictions[seq_id].append(prediction)
            seq_probabilities[seq_id].append(probability)
            if seq_id not in seq_labels:
                seq_labels[seq_id] = label
    final_predictions = []
    final_probabilities = []
    for seq_id in seq_labels:
        final_predictions.append(max(seq_predictions[seq_id], key=seq_predictions[seq_id].count))
        final_probabilities.append(sum(seq_probabilities[seq_id]) / len(seq_probabilities[seq_id]))
    final_labels = [seq_labels[seq_id] for seq_id in seq_labels]
    important_tokens = sorted(token_importance_dict.items(), key=lambda x: x[1], reverse=True)

    f1, accuracy, hamming, jaccard, precision, recall, auc, confusion, class_report = metrics_calculator.calculate_metrics(
        final_labels, final_predictions, final_probabilities, target_format)

    metrics_calculator.print_eval_metrics(logger, accuracy, f1, best_metric,
                                          best_epoch, hamming, jaccard, precision, recall, best_precision, best_recall,
                                          auc, confusion, class_report)

    correct = []
    wrong = []
    important_tokens = [(token, importance) for token, importance in important_tokens if importance > 1]
    if target_format == "multi-cat":
        for isolate in seq_labels:
            if np.any(seq_labels[isolate] == 1):
                if not np.array_equal(seq_labels[isolate], seq_predictions[isolate]):
                    wrong.append(isolate)
                else:
                    correct.append(isolate)
    else:
        for isolate in seq_labels:
            if seq_labels[isolate] == 1:
                if seq_labels[isolate] != seq_predictions[isolate]:
                    wrong.append(isolate)
                else:
                    correct.append(isolate)
    if target_format != "multi-cat":
        logger.log(f"CORRECT RESISTANT ISOLATES {correct}")
        logger.log(f"INCORRECT RESISTANT ISOLATES {wrong}")
    logger.log(f"IMPORTANT TOKENS {important_tokens}")
    logger.close()

def tokenize_sets(prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, tokenizer_manager, args, fold):
    encoded_sequences = []
    gene_mapping = {}
    for index, sequence in enumerate(prepped_seqs):
        encoded_sequence = tokenizer_manager.bpe_encode_sequences_genes(sequence, tokenizer)
        combined_list = [item for sublist in encoded_sequence for item in sublist]
        encoded_sequences.append([combined_list])
        genes_in_this_isolate = genes_list[index]
        for token, gene in list(zip(encoded_sequence, genes_list[index])):
            if gene not in gene_mapping:
                gene_mapping[gene] = []
            if token[0] not in gene_mapping[gene]:
                gene_mapping[gene].append(token[0])

    zipped_data = list(zip(encoded_sequences, prepped_labels, seq_ids))

    return zipped_data

def create_dataset(zipped_data, classification_type):
    seqs_ = []
    labs = []
    seq_ids = []
    for seqs, label, seq_id in zipped_data:
        for seq in seqs:
            seqs_.append(seq)
            labs.append(label)
            seq_ids.append(seq_id)
    return SeqDataset(seqs_, labs, seq_ids, classification_type)
    
def evaluate(zipped_data, args, sequence_processor, tokenizer_manager, gene_manager, mode="Train"):

    if "multi-cat" not in args.antibiotic:
        target_format = "binary"
    else:
        target_format = "multi-cat"
        
    sequences_te = [seq for seq, _, _, _ in zipped_data]
    labels_te = [label for _, label, _, _ in zipped_data]
    seq_ids_te = [seq_id for _, _, seq_id, _ in zipped_data]
    genes_list_te = [genes for _, _, _, genes in zipped_data]

    for fold in range(5):
        
        if mode == "Evaluate":
            print(f"{args.save_path}/{args.antibiotic}/tokenizer.json")
            # tokenizer = tokenizer_manager.load_kmer_tokenizer(f"{args.save_path}/{args.antibiotic}/tokenizer_{fold+1}.json")
            tokenizer = tokenizer_manager.load_kmer_tokenizer(f"{args.save_path}/{args.antibiotic}/tokenizer.json")
        
        unique_n_mers = tokenizer.get_vocab()
        
        vocab_size = len(unique_n_mers)-4
        batch_size = int(args.batch_size)
        config = json.load(open(args.model_config))
        save_path = os.path.join(args.save_path, args.antibiotic)

        unique_n_mers_te, prepped_seqs_te, prepped_labels_te = sequence_processor.extract_and_prep_genes(
                sequences_te, labels_te)
        
        test_data_prepped = tokenize_sets(prepped_seqs_te, prepped_labels_te, seq_ids_te, genes_list_te, tokenizer,
                                     tokenizer_manager, args, fold)
        
        print(vocab_size)
        
        model_manager = ModelManager(vocab_size, config)
        metrics_calculator = MetricsCalculator()
        test_dataset = create_dataset(test_data_prepped , target_format)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: collate_fn(batch, classification_type=target_format),
                                     num_workers=0, pin_memory=True)
        _evaluate(test_loader, os.path.join(save_path, f"best_model_fold_{fold + 1}.pth"), target_format, metrics_calculator,
                 model_manager, args, fold)

