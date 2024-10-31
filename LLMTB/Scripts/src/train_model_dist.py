import json
import os
import gc
import torch
import numpy as np
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter, defaultdict
from Utils.Logger_Manager import LoggerManager
from Utils.Metric_Calculator import MetricsCalculator
from Utils.Model_Manager import ModelManager
from Utils.Dataset import SeqDataset, collate_fn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import random
from functools import partial


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
class CombinedFocalLabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, reduction='mean'):
        super(CombinedFocalLabelSmoothingLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        BCE_loss = F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction='none')
        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def setup_and_run(rank, world_size, train_dataset, val_dataset, batch_size, *args):
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '8989')

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size, shuffle=False)

    collate_fn_partial = partial(collate_fn, classification_type=args[3])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn_partial,
        num_workers=6,  
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=2,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn_partial,  
        num_workers=6, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=2,
        sampler=val_sampler
    )

    _train_model(rank, world_size, train_loader, val_loader, *args)
 
 
def evaluate(data_loader, model_path, target_format, metrics_calculator, model_manager, args, fold):
    save_path = os.path.join(args.save_path, args.antibiotic)
    logger = LoggerManager(args.antibiotic, fold + 1, save_path, train=False)
    logger.log("Evaluating on Test Set...")
    best_epoch, best_metric, best_precision, best_recall = 0, 0, 0, 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))

    model, threshold = model_manager.load_model_threshold(model_path, device_ids=[0])
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
                    token_importance = avg_attn_weights[batch_idx].sum(dim=0).squeeze()
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
    
    f1, accuracy, hamming, jaccard, precision, recall, auc, confusion, class_report = metrics_calculator.calculate_metrics(final_labels, final_predictions, final_probabilities, target_format)

    metrics_calculator.print_eval_metrics(logger, accuracy, f1, best_metric,
                                              best_epoch, hamming, jaccard, precision, recall, best_precision, best_recall, auc, confusion, class_report)

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
    

def adversarial_training(model, loss_fn, sequence, attention_masks, labels, epsilon=1e-5):
    param_states = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and 'embedding' in name:
            param_states[name] = param.data.clone()
    
    grad_norm = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and 'embedding' in name:
            grad_norm += (param.grad.detach() ** 2).sum()
    grad_norm = grad_norm.sqrt()
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and 'embedding' in name:
            param.data.add_(epsilon * param.grad / grad_norm)
    
    outputs, _ = model(sequence, mask=attention_masks)
    outputs = outputs.squeeze(1)
    adv_loss = loss_fn(outputs, labels.float())
    
    for name, param in model.named_parameters():
        if name in param_states:
            param.data = param_states[name]
    
    return adv_loss


def _train_model(rank, world_size, train_loader, val_loader, save_path, target_format, fold,
                 labels_for_weighting, num_epochs, model_manager, optimizer, scheduler, metrics_calculator, args):

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"Rank {rank}: Starting training", flush=True)

    model = model_manager.model.to(device)
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model = DDP(model, device_ids=[rank])
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if rank == 0:
        logger = LoggerManager(args.antibiotic, fold + 1, save_path, train=True)
        logger.log(args)
        logger.log(f"Training on Rank {rank} for {num_epochs} epochs...")

    if target_format != "multi-cat":
        label_counts = Counter(labels_for_weighting)
        # pos_weight_value = label_counts[0] / label_counts[1]
        # if rank == 0:
        #     logger.log(f"Positive class weight: {pos_weight_value}")
        # pos_weight = torch.tensor([pos_weight_value], device=device)
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        negative_counts = label_counts[0]
        positive_counts = label_counts[1]

        alpha = negative_counts / (negative_counts + positive_counts)
        imbalance_ratio = negative_counts / positive_counts
        if imbalance_ratio < 10:
            gamma = 2.0  # Moderate imbalance
        elif imbalance_ratio < 100:
            gamma = 3.0  # High imbalance
        else:
            gamma = 5.0  # Severe imbalance
            
        if rank == 0:
            logger.log(f"alpha: {alpha}")
            logger.log(f"gamma: {gamma}")

        # loss_fn = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
        loss_fn = CombinedFocalLabelSmoothingLoss(alpha=alpha, gamma=gamma, reduction='mean', smoothing=0.1)
    else:
        labels_array = np.array(labels_for_weighting)
        positive_counts = labels_array.sum(axis=0)
        negative_counts = (1 - labels_array).sum(axis=0)
        epsilon = 1e-6
        positive_weights = [
            neg_count / (pos_count + epsilon)
            for neg_count, pos_count in zip(negative_counts, positive_counts)
        ]
        pos_weight_tensor = torch.tensor(positive_weights, device=device)
        if rank == 0:
            logger.log(f"Positive class weights: {positive_weights}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    best_metric = 0
    best_epoch = 0
    best_threshold = None
    max_norm = 1.0
    
    scaler = GradScaler()
    
    if rank == 0:
        logger.log(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        current_lr = scheduler.get_last_lr()[0]
        if rank == 0:
            logger.log(f"{'-' * 20}\nEpoch {epoch + 1}, Learning Rate: {current_lr}")
        total_loss = 0
        model.train()
        for batch in train_loader:
            sequence, attention_masks, labels, _ = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs, _ = model(sequence, mask=attention_masks)
                outputs = outputs.squeeze(1)
                loss = loss_fn(outputs, labels.float())
                
            scaler.scale(loss).backward()

            # Adversarial training
            adv_loss = adversarial_training(model, loss_fn, sequence, attention_masks, labels, epsilon=1e-4)
            scaler.scale(adv_loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() + adv_loss.item()
        dist.barrier()
        scheduler.step()
        average_train_loss = total_loss / len(train_loader)
        if rank == 0:
            logger.log(f"Train Loss: {average_train_loss}")
        
        model.eval()
        total_eval_loss = 0
        local_seq_ids = []
        local_seq_probabilities = []
        local_seq_labels = []
        token_importance_dict = {}
        token_counts = defaultdict(int)
        with torch.no_grad():
            for sequence, attention_masks, labels, seq_ids in val_loader:
                sequence = sequence.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                with autocast():
                    outputs, attention_weights = model(sequence, mask=attention_masks)
                    outputs = outputs.squeeze(1)
                    loss = loss_fn(outputs, labels.float())
                    
                # outputs, attention_weights = model(sequence, mask=attention_masks)
                # outputs = outputs.squeeze(1)
                # loss = loss_fn(outputs, labels.float())
        
                total_eval_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                local_seq_ids.extend(seq_ids)
                local_seq_probabilities.extend(probabilities.cpu().numpy())
                local_seq_labels.extend(labels.cpu().numpy())
                

                # avg_attn_weights = torch.stack(attention_weights).mean(dim=0).mean(dim=1)
                # token_importance_scores = avg_attn_weights.sum(dim=1)
                
                # avg_attn_weights = torch.stack(attention_weights).mean(dim=0)
                # token_importance_scores = avg_attn_weights.sum(dim=2)
                
                stacked_attention = torch.stack(attention_weights)
                avg_attention = stacked_attention.mean(dim=0)
                token_importance_scores = avg_attention.sum(dim=1)   # Changed from dim=1 to dim=2
                
                for batch_idx in range(sequence.size(0)):
                    token_ids = sequence[batch_idx]
                    attn_scores = token_importance_scores[batch_idx]
                    for token_id, score in zip(token_ids.cpu().tolist(), attn_scores.cpu().tolist()):
                        if token_id == 0:
                            continue  # Skip padding tokens
                        if token_id in token_importance_dict:
                            token_importance_dict[token_id] += score
                            token_counts[token_id] += 1
                        else:
                            token_importance_dict[token_id] = score
                            token_counts[token_id] += 1
                
                            
        gathered_token_importance = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_token_importance, token_importance_dict)
        
        gathered_token_counts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_token_counts, token_counts)

        total_eval_loss_tensor = torch.tensor([total_eval_loss], device=device)
        all_eval_loss_tensors = [torch.zeros_like(total_eval_loss_tensor) for _ in range(world_size)]
        dist.all_gather(all_eval_loss_tensors, total_eval_loss_tensor)
        combined_eval_loss = sum([tensor.item() for tensor in all_eval_loss_tensors])
        average_eval_loss = combined_eval_loss / (len(val_loader) * world_size)
        
        local_data = list(zip(local_seq_ids, local_seq_probabilities, local_seq_labels))
        gathered_data = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_data, local_data)
        
        if rank == 0:
            all_seq_ids = []
            all_probabilities = []
            all_labels = []
            
            for rank_data in gathered_data:
                for seq_id, prob, label in rank_data:
                    all_seq_ids.append(seq_id)
                    all_probabilities.append(prob)
                    all_labels.append(label)

            combined_seq_probabilities = defaultdict(list)
            combined_seq_labels = {}
            
            for seq_id, prob, label in zip(all_seq_ids, all_probabilities, all_labels):
                combined_seq_probabilities[seq_id].append(prob)
                combined_seq_labels[seq_id] = label
            
            final_probabilities = [
                np.mean(combined_seq_probabilities[seq_id], axis=0)
                for seq_id in combined_seq_labels
            ]
            final_labels = [combined_seq_labels[seq_id] for seq_id in combined_seq_labels]

            total_token_importance = defaultdict(float)
            total_token_counts = defaultdict(int)
            
            # Combine importance scores and counts from all processes
            for rank_importance_dict, rank_counts_dict in zip(gathered_token_importance, gathered_token_counts):
                for token_id, importance in rank_importance_dict.items():
                    total_token_importance[token_id] += importance
                for token_id, count in rank_counts_dict.items():
                    total_token_counts[token_id] += count
            
            # Compute average importance
            average_token_importance = {}
            for token_id in total_token_importance:
                average_importance = total_token_importance[token_id] / total_token_counts[token_id]
                average_token_importance[token_id] = average_importance
        
            # Proceed with logging or further analysis
            # For example, sort and log the most important tokens
            important_tokens = sorted(
                average_token_importance.items(), key=lambda x: x[1], reverse=True
            )
            important_tokens = [(token_id, score) for token_id, score in important_tokens if score > 1]

            metrics_results = metrics_calculator.calculate_metrics_threshold(
                final_labels, final_predictions=None, final_probabilities=final_probabilities, target_format=target_format
            )
            f1, accuracy, hamming, jaccard, precision, recall, auc, confusion, class_report, current_best_threshold = metrics_results
            
            metrics_calculator.print_metrics_threshold(
                logger, average_eval_loss, accuracy, f1, best_metric, best_epoch, hamming,
                jaccard, precision, recall, auc, confusion, class_report, current_best_threshold
            )
            
            if f1 > best_metric or epoch == 0:
                best_metric = f1
                best_epoch = epoch + 1
                best_threshold = current_best_threshold
                if target_format != "multi-cat":
                    logger.log(f"Best Threshold: {best_threshold}")
                else:
                    logger.log(f"Best Thresholds: {best_threshold}")
                logger.log(f"Important Tokens: {important_tokens}")
                model_save_path = os.path.join(save_path, f"best_model_fold_{fold + 1}.pth")
                model_manager.save_model_threshold(model_save_path, best_threshold, logger)
    if rank == 0:
        logger.log(f"Training Finished")
    dist.destroy_process_group()
    
    
def create_folds(train_val_data, labels, target_format, n_splits=5, seed=42):
    folds = []

    if target_format == "multi-cat":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(train_val_data):
            train_fold = [train_val_data[i] for i in train_idx]
            val_fold = [train_val_data[i] for i in val_idx]
            folds.append((train_fold, val_fold))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(train_val_data, labels):
            train_fold = [train_val_data[i] for i in train_idx]
            val_fold = [train_val_data[i] for i in val_idx]
            folds.append((train_fold, val_fold))

    return folds


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
    with open(f"{args.save_path}/{args.antibiotic}/Gene_to_Token_{fold+1}.json", "w") as j:
        json.dump(gene_mapping, j, indent=3)

    zipped_data = list(zip(encoded_sequences, prepped_labels, seq_ids))

    return zipped_data
    

def oversample_minority_class(dataset):
    # Separate majority and minority class samples
    majority_samples = [sample for sample in dataset if sample[1] == 0]
    minority_samples = [sample for sample in dataset if sample[1] == 1]

    # Calculate how many times to duplicate minority samples
    majority_count = len(majority_samples)
    minority_count = len(minority_samples)
    duplication_factor = majority_count // minority_count

    # Duplicate minority samples
    oversampled_minority_samples = minority_samples * duplication_factor
    # Optionally add remaining samples to match majority count
    remaining = majority_count - len(oversampled_minority_samples)
    oversampled_minority_samples += minority_samples[:remaining]

    # Combine and shuffle
    oversampled_dataset = majority_samples + oversampled_minority_samples
    random.shuffle(oversampled_dataset)

    return oversampled_dataset


def train(zipped_data, args, sequence_processor, tokenizer_manager, gene_manager, mode):

    save_path = os.path.join(args.save_path, args.antibiotic)
    os.makedirs(save_path, exist_ok=True)

    if "multi-cat" not in args.antibiotic:
        target_format = "binary"
    else:
        target_format = "multi-cat"

    if args.use_holdout:
        if target_format != "multi-cat":
            labels = [label for _, label, _ , _ in zipped_data]
            train_val_data, test_data = train_test_split(zipped_data, test_size=0.15, stratify=labels, random_state=42)
        else:
            train_val_data, test_data = train_test_split(zipped_data, test_size=0.15, random_state=42)
        test_seq_ids = [seq_id for _, _, seq_id, _ in test_data]

        with open(os.path.join(save_path, 'test_seq_ids.json'), 'w') as f:
            json.dump(test_seq_ids, f, indent=3)
    else:
        train_val_data = zipped_data

    include_val_in_tokenizer = True
    
    labels = [label for _, label, _, _ in train_val_data]
    
    if include_val_in_tokenizer:
        sequences = [seq for seq, _, _, _ in train_val_data]
        labels = [label for _, label, _, _ in train_val_data]
        seq_ids = [seq_id for _, _, seq_id, _ in train_val_data]
        genes_list = [genes for _, _, _, genes in train_val_data]
        
        unique_n_mers, prepped_seqs, prepped_labels = sequence_processor.extract_and_prep_genes(sequences, labels)
        
        if mode == "Evaluate":
            print(f"{args.save_path}/{args.antibiotic}/tokenizer.json")
            tokenizer = tokenizer_manager.load_kmer_tokenizer(f"{args.save_path}/{args.antibiotic}/tokenizer.json")
        else:
            tokenizer = tokenizer_manager.setup_kmer_tokenizer(prepped_seqs, unique_n_mers)
    
        if mode == "Train":
                tokenizer_manager.save_tokenizer(tokenizer, f"{args.save_path}/{args.antibiotic}/tokenizer.json")
                
        train_val_data = tokenize_sets(prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, tokenizer_manager, args, 0)

        vocab_size = len(unique_n_mers) + 1
        batch_size = int(args.batch_size)
        config = json.load(open(args.model_config))
        print(f"Vocab Size: {vocab_size}")
        print(train_val_data[0])

    folds = create_folds(train_val_data, labels, target_format)

    for fold, (train_fold, val_fold) in enumerate(folds):
        if not include_val_in_tokenizer:
            sequences, sequences_val = [seq for seq, _, _, _ in train_fold], [seq for seq, _, _, _ in val_fold]
            labels, labels_val = [label for _, label, _, _ in train_fold], [label for _, label, _, _ in val_fold]
            seq_ids, seq_ids_val = [seq_id for _, _, seq_id, _ in train_fold], [seq_id for _, _, seq_id, _ in val_fold]
            genes_list, genes_list_val = [genes for _, _, _, genes in train_fold], [genes for _, _, _, genes in val_fold]
    
            unique_n_mers, prepped_seqs, prepped_labels = sequence_processor.extract_and_prep_genes(sequences, labels)
            unique_n_mers_val, prepped_seqs_val, prepped_labels_val = sequence_processor.extract_and_prep_genes(sequences_val, labels_val)

            if mode == "Evaluate":
                print(f"{args.save_path}/{args.antibiotic}/tokenizer.json")
                tokenizer = tokenizer_manager.load_kmer_tokenizer(f"{args.save_path}/{args.antibiotic}/tokenizer_{fold+1}.json")
            else:
                tokenizer = tokenizer_manager.setup_kmer_tokenizer(prepped_seqs, unique_n_mers)
    
            if mode == "Train":
                tokenizer_manager.save_tokenizer(tokenizer, f"{args.save_path}/{args.antibiotic}/tokenizer_{fold+1}.json")

            train_fold = tokenize_sets(prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, tokenizer_manager, args, fold)
            val_fold = tokenize_sets(prepped_seqs_val, prepped_labels_val, seq_ids_val, genes_list_val, tokenizer, tokenizer_manager, args, fold)

            vocab_size = len(unique_n_mers) + 1
            batch_size = int(args.batch_size)
            config = json.load(open(args.model_config))

        if args.use_gen:
            generated = []
            real = []
            for item in zipped_data:
                if "Generated_Isolate" in item[2] or "Augmented_Isolate" in item[2]:
                    generated.append(item)
                else:
                    real.append(item)
            zipped_data = real

        model_manager = ModelManager(vocab_size, config)
        optimizer = AdamW(model_manager.model.parameters(), lr=config["learning_rate"], weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-8)
        metrics_calculator = MetricsCalculator()

        # if "multi-cat" not in target_format:
        #     label_for_weighting = [label for _, label, _ in train_fold]
        # else:
        #     label_for_weighting = [label for _, label, _ in train_fold]


        # Save the isolate numbers for each set for reproduction purposes
        train_seq_ids = [seq_id for _, _, seq_id in train_fold]
        val_seq_ids = [seq_id for _, _, seq_id in val_fold]
        with open(os.path.join(save_path, f'train_seq_ids_fold_{fold + 1}.json'), 'w') as f:
            json.dump(train_seq_ids, f, indent=3)
        with open(os.path.join(save_path, f'val_seq_ids_fold_{fold + 1}.json'), 'w') as f:
            json.dump(val_seq_ids, f, indent=3)

        # train_dataset = create_dataset(train_fold, target_format)
        oversampled_train_data = oversample_minority_class(train_fold)
        if "multi-cat" not in target_format:
            label_for_weighting = [label for _, label, _ in oversampled_train_data]
        else:
            label_for_weighting = [label for _, label, _ in oversampled_train_data]
        train_dataset = create_dataset(oversampled_train_data, target_format)
        val_dataset = create_dataset(val_fold, target_format)

        world_size = torch.cuda.device_count()

        args_l = (train_dataset, val_dataset, batch_size, save_path, target_format, fold,
                  label_for_weighting, args.num_epochs, model_manager, optimizer, scheduler, metrics_calculator, args)

        mp.spawn(setup_and_run, args=(world_size, *args_l), nprocs=world_size, join=True)

        if args.use_holdout:
            sequences_te = [seq for seq, _, _, _ in test_data]
            labels_te = [label for _, label, _, _ in test_data]
            seq_ids_te = [seq_id for _, _, seq_id, _ in test_data]
            genes_list_te = [genes for _, _, _, genes in test_data]

            unique_n_mers_te, prepped_seqs_te, prepped_labels_te = sequence_processor.extract_and_prep_genes(
                sequences_te, labels_te)

            test_data_prepped = tokenize_sets(prepped_seqs_te, prepped_labels_te, seq_ids_te, genes_list_te, tokenizer,
                                     tokenizer_manager, args, fold)

            test_dataset = create_dataset(test_data_prepped , target_format)

            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: collate_fn(batch, classification_type=target_format),
                                     num_workers=0, pin_memory=True)
            evaluate(test_loader, os.path.join(save_path, f"best_model_fold_{fold + 1}.pth"), target_format,
                     metrics_calculator, model_manager, args, fold)

        gc.collect()
        torch.cuda.empty_cache()