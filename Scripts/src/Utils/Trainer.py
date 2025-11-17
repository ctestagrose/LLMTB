import json
import os
import gc
import torch
import numpy as np
from functools import partial
from collections import Counter, defaultdict
from .Logger_Manager import LoggerManager
from .Metric_Calculator import MetricsCalculator
from .Model_Manager import ModelManager
from .Dataset import SeqDataset, collate_fn
from .Loss_Functions import (CombinedFocalLabelSmoothingLoss,
                             CombinedFocalLabelSmoothingLossMultiCat)
from .Evaluater import Evaluater
from torch.optim import AdamW
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split
from tqdm import tqdm


class Trainer:
    def __init__(self, args, sequence_processor, tokenizer_manager, gene_manager):
        self.args = args
        self.sequence_processor = sequence_processor
        self.tokenizer_manager = tokenizer_manager
        self.gene_manager = gene_manager

    def _get_device(self, rank: int = 0):
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            return torch.device(f"cuda:{rank}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def setup_and_run(self, rank, world_size, train_dataset, val_dataset,
                      batch_size, tokenizer, distributed, *args):
        if distributed:
            master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
            master_port = os.environ.get('MASTER_PORT', '8989')
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port

            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        device = self._get_device(rank)

        if distributed:
            train_sampler = DistributedSampler(train_dataset, rank=rank,
                                               num_replicas=world_size, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, rank=rank,
                                             num_replicas=world_size, shuffle=False)
            shuffle_train = False
            shuffle_val = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle_train = True
            shuffle_val = False

        collate_fn_partial = partial(
            collate_fn,
            classification_type=args[3],
            MASK_TOKEN=tokenizer.token_to_id("[MASK]"),
            PAD_TOKEN=tokenizer.token_to_id("[PAD]"),
            VOCAB_SIZE=self.tokenizer_manager.vocab_size,
        )

        num_cpus = torch.get_num_threads()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_fn_partial,
            num_workers=num_cpus,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
            sampler=train_sampler,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_val,
            collate_fn=collate_fn_partial,
            num_workers=num_cpus,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
            sampler=val_sampler,
        )

        self._train_model(
            rank, world_size, train_loader, val_loader,
            *args,
            distributed=distributed,
            device=device,
        )

        self._train_model(rank, world_size, train_loader, val_loader, *args)

    def tokenize_sets(self, prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, args, fold):
        encoded_sequences = []
        gene_mapping = {}
        token_to_gene = {}

        for index, sequence in enumerate(prepped_seqs):
            encoded_sequence = self.tokenizer_manager.encode_sequences_genes(sequence, tokenizer)
            combined_list = [item for sublist in encoded_sequence for item in sublist]
            encoded_sequences.append([combined_list])
            genes_in_this_isolate = genes_list[index]

            for token, gene in zip(encoded_sequence, genes_in_this_isolate):
                token_id = token[0]
                if token_id == 0:
                    continue

                if gene not in gene_mapping:
                    gene_mapping[gene] = []
                if token_id not in gene_mapping[gene]:
                    gene_mapping[gene].append(token_id)

                token_to_gene[token_id] = {
                    'gene': gene,
                    'is_intergenic': gene.endswith('_ir_before') or gene.endswith('_ir_after')
                }

        mappings = {
            'gene_to_token': gene_mapping,
            'token_to_gene': token_to_gene
        }

        with open(f"{args.save_path}/{args.antibiotic}/Gene_Token_Mapping.json", "w") as j:
            json.dump(mappings, j, indent=3)
        # with open(f"{args.save_path}/{args.antibiotic}/Gene_Token_Mapping_{fold+1}.json", "w") as j:
        #     json.dump(mappings, j, indent=3)

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

    def oversample_minority_class(self, dataset):
        majority_samples = [sample for sample in dataset if sample[1] == 0]
        minority_samples = [sample for sample in dataset if sample[1] == 1]

        majority_count = len(majority_samples)
        minority_count = len(minority_samples)
        duplication_factor = majority_count // minority_count

        oversampled_minority_samples = minority_samples * duplication_factor
        remaining = majority_count - len(oversampled_minority_samples)
        oversampled_minority_samples += minority_samples[:remaining]

        oversampled_dataset = majority_samples + oversampled_minority_samples
        return oversampled_dataset

    def create_folds(self, train_val_data, labels, target_format, n_splits=5, val_size=0.2, seed=42):
        folds = []

        if n_splits == 1:
            if target_format == "multi-cat":
                train_data, val_data = train_test_split(train_val_data,
                                                        test_size=val_size,
                                                        random_state=seed)
            else:
                train_data, val_data = train_test_split(train_val_data,
                                                        test_size=val_size,
                                                        stratify=labels,
                                                        random_state=seed)
            folds.append((train_data, val_data))
        else:
            if target_format == "multi-cat":
                splitter = ShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)
                for train_idx, val_idx in splitter.split(train_val_data):
                    train_fold = [train_val_data[i] for i in train_idx]
                    val_fold = [train_val_data[i] for i in val_idx]
                    folds.append((train_fold, val_fold))
            else:
                splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=seed)
                for train_idx, val_idx in splitter.split(train_val_data, labels):
                    train_fold = [train_val_data[i] for i in train_idx]
                    val_fold = [train_val_data[i] for i in val_idx]
                    folds.append((train_fold, val_fold))

        return folds

    def adversarial_training(self, model, loss_fn, sequence, attention_masks, labels, epsilon=1e-5):
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

    def train(self, zipped_data, mode):
        args = self.args
        sequence_processor = self.sequence_processor
        tokenizer_manager = self.tokenizer_manager

        save_path = os.path.join(args.save_path, args.antibiotic)
        os.makedirs(save_path, exist_ok=True)

        target_format = "binary" if "multi-cat" not in args.antibiotic else "multi-cat"

        if args.use_holdout:
            labels = [label for _, label, _, _ in zipped_data]
            if target_format != "multi-cat":
                train_val_data, test_data = train_test_split(zipped_data, test_size=0.15, stratify=labels,
                                                             random_state=42)
            else:
                train_val_data, test_data = train_test_split(zipped_data, test_size=0.15, random_state=42)
            test_seq_ids = [seq_id for _, _, seq_id, _ in test_data]

            with open(os.path.join(save_path, 'test_seq_ids.json'), 'w') as f:
                json.dump(test_seq_ids, f, indent=3)
        else:
            train_val_data = zipped_data

        sequences = [seq for seq, _, _, _ in train_val_data]
        labels = [label for _, label, _, _ in train_val_data]
        seq_ids = [seq_id for _, _, seq_id, _ in train_val_data]
        genes_list = [genes for _, _, _, genes in train_val_data]

        unique_n_mers, prepped_seqs, prepped_labels = sequence_processor.extract_and_prep_genes(sequences, labels)

        if mode == "Evaluate":
            tokenizer_path = f"{args.save_path}/{args.antibiotic}/tokenizer.json"
            tokenizer = tokenizer_manager.load_tokenizer(tokenizer_path)
        else:
            tokenizer = tokenizer_manager.setup_tokenizer(unique_n_mers)
            tokenizer_manager.save_tokenizer(tokenizer, f"{args.save_path}/{args.antibiotic}/tokenizer.json")

        train_val_data = self.tokenize_sets(prepped_seqs, prepped_labels, seq_ids, genes_list, tokenizer, args, 0)

        vocab_size = len(unique_n_mers) + 5
        self.tokenizer_manager.vocab_size = len(unique_n_mers) + 5
        batch_size = int(args.batch_size)
        config = json.load(open(args.model_config))

        folds = self.create_folds(train_val_data, labels, target_format, n_splits=1, val_size=0.20)

        for fold, (train_fold, val_fold) in enumerate(folds):
            model_manager = ModelManager(vocab_size, config)
            optimizer = AdamW(model_manager.model.parameters(), lr=config["learning_rate"], weight_decay=1e-3)
            scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-8)
            metrics_calculator = MetricsCalculator()

            # Save the isolate numbers for each set for reproduction purposes
            train_seq_ids = [seq_id for _, _, seq_id in train_fold]
            val_seq_ids = [seq_id for _, _, seq_id in val_fold]
            with open(os.path.join(save_path, f'train_seq_ids_fold_{fold + 1}.json'), 'w') as f:
                json.dump(train_seq_ids, f, indent=3)
            with open(os.path.join(save_path, f'val_seq_ids_fold_{fold + 1}.json'), 'w') as f:
                json.dump(val_seq_ids, f, indent=3)

            if "multi" not in args.antibiotic:
                if args.oversample:
                    oversampled_train_data = self.oversample_minority_class(train_fold)
                    label_for_weighting = [label for _, label, _ in oversampled_train_data]
                    train_dataset = self.create_dataset(oversampled_train_data, target_format)
                else:
                    label_for_weighting = [label for _, label, _ in train_fold]
                    train_dataset = self.create_dataset(train_fold, target_format)
                val_dataset = self.create_dataset(val_fold, target_format)
            else:
                label_for_weighting = []

                for idx in range(len(train_fold)):
                    sample = train_fold[idx]
                    label = sample[1]
                    label_for_weighting.append(label)

                train_dataset = self.create_dataset(train_fold, target_format)
                val_dataset = self.create_dataset(val_fold, target_format)
                # print(len(train_dataset[0]["sequence"]))
                # print(len(train_dataset[0]))

            num_gpus = torch.cuda.device_count()
            use_distributed = (
                args.distributed
                and torch.cuda.is_available()
                and num_gpus > 1
            )

            args_l = (
                train_dataset, val_dataset, batch_size, tokenizer, save_path,
                target_format, fold, label_for_weighting, args.num_epochs,
                model_manager, optimizer, scheduler, metrics_calculator,
                args, tokenizer
            )

            if use_distributed:
                world_size = num_gpus
                mp.spawn(
                    self.setup_and_run,
                    args=(world_size, *args_l, True),
                    nprocs=world_size,
                    join=True,
                )
            else:
                rank = 0
                world_size = 1
                device = self._get_device(rank)

                collate_fn_partial = partial(
                    collate_fn,
                    classification_type=target_format,
                    MASK_TOKEN=tokenizer.token_to_id("[MASK]"),
                    PAD_TOKEN=tokenizer.token_to_id("[PAD]"),
                    VOCAB_SIZE=self.tokenizer_manager.vocab_size,
                )
                num_cpus = torch.get_num_threads()

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn_partial,
                    num_workers=num_cpus,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=True,
                    prefetch_factor=2,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn_partial,
                    num_workers=num_cpus,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=True,
                    prefetch_factor=2,
                )

                self._train_model(
                    rank, world_size,
                    train_loader, val_loader,
                    save_path, target_format,
                    fold, label_for_weighting, args.num_epochs,
                    model_manager, optimizer, scheduler, metrics_calculator,
                    args, tokenizer,
                    distributed=False,
                    device=device,
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.use_holdout:
            evaluater = Evaluater(args, sequence_processor, tokenizer_manager, self.gene_manager)
            evaluater.evaluate(test_data, mode="Evaluate", fold=fold)


    def _train_model(
        self, rank, world_size, train_loader, val_loader, save_path, target_format,
        fold, labels_for_weighting, num_epochs, model_manager, optimizer, scheduler,
        metrics_calculator, args, tokenizer,
        distributed: bool = False,
        device: torch.device | None = None,
    ):
        global stop_monitor
        if device is None:
            device = self._get_device(rank)
        device_type = "cuda" if device.type == "cuda" else (
            "mps" if device.type == "mps" else "cpu"
        )

        if rank == 0:
            print(f"Rank {rank}: Starting training on {device}", flush=True)

        model = model_manager.model.to(device)
        if distributed:
            model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)


        if device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if rank == 0:
            logger = LoggerManager(args.antibiotic, fold + 1, save_path, train=True)
            logger.log(args)
            logger.log(f"Training on Rank {rank} for {num_epochs} epochs...")

        if target_format != "multi-cat":
            label_counts = Counter(labels_for_weighting)
            negative_counts = label_counts[0]
            positive_counts = label_counts[1]
            total_counts = negative_counts + positive_counts

            alpha = positive_counts / total_counts
            alpha = 1 - alpha
            alpha = np.clip(alpha, 0.0, 1.0)
            imbalance_ratio = negative_counts / positive_counts
            gamma = max(0, 1 + np.log10(imbalance_ratio))

            if rank == 0:
                logger.log(f"alpha: {alpha}")
                logger.log(f"gamma: {gamma}")

            loss_fn = CombinedFocalLabelSmoothingLoss(alpha=alpha, gamma=gamma, reduction='mean', smoothing=0.5)
        else:
            labels_tensor = torch.tensor(labels_for_weighting)
            positive_counts = torch.sum(labels_tensor, dim=0)
            negative_counts = labels_tensor.size(0) - positive_counts
            total_counts = labels_tensor.size(0)

            epsilon = 1e-7
            positive_freq = positive_counts / (total_counts + epsilon)
            negative_freq = negative_counts / (total_counts + epsilon)

            alpha = 1 - positive_freq
            alpha = torch.clamp(alpha, min=0.0, max=1.0)

            imbalance_ratio = negative_freq / (positive_freq + epsilon)
            gamma = torch.clamp_min(1 + torch.log10(imbalance_ratio + epsilon), min=0)
            if rank == 0:
                logger.log(f"alpha: {alpha}")
                logger.log(f"gamma: {gamma}")
            alpha = alpha.to(device)
            gamma = gamma.to(device)
            loss_fn = CombinedFocalLabelSmoothingLossMultiCat(alpha=alpha, gamma=gamma, reduction='mean', smoothing=0.0)

        best_metric = 0
        best_epoch = 0

        scaler = GradScaler()

        if rank == 0:
            ordering = ["RIF", "INH", "EMB", "AMI", "KAN",
                        "RFB", "LEV", "MXF", "ETH", "LZD",
                        "CFZ", "DLM", "BDQ"]
            logger.log(f"Ordering: {ordering}")
            logger.log(f"Training for {num_epochs} epochs...")


        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            current_lr = scheduler.get_last_lr()[0]
            if rank == 0:
                logger.log(f"{'-' * 20}\nEpoch {epoch + 1}, Learning Rate: {current_lr}")
            total_loss = 0
            model.train()
            for batch in train_loader:
                sequence, attention_masks, labels, _ = [
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
                ]
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type):
                    outputs, _ = model(sequence, mask=attention_masks)
                    outputs = outputs.squeeze(1)
                    loss = loss_fn(outputs, labels.float())

                scaler.scale(loss).backward()

                if args.adversarial_training:
                    adv_loss = self.adversarial_training(model, loss_fn, sequence, attention_masks, labels, epsilon=1e-2)
                    scaler.scale(adv_loss).backward()

                scaler.step(optimizer)
                scaler.update()

                if args.adversarial_training:
                    total_loss += loss.item() + adv_loss.item()
                else:
                    total_loss += loss.item()
            if distributed:
                dist.barrier()
            scheduler.step()
            average_train_loss = total_loss / len(train_loader)

            if rank == 0:
                logger.log(f"Train Loss: {average_train_loss}")

            model.eval()
            total_eval_loss = 0
            local_seq_data = []
            with torch.no_grad():
                for sequence, attention_masks, labels, seq_ids in val_loader:
                    sequence = sequence.to(device)
                    attention_masks = attention_masks.to(device)
                    labels = labels.to(device)
                    with autocast(device_type):
                        outputs, attention_weights = model(sequence, mask=attention_masks)
                        outputs = outputs.squeeze(1)
                        loss = loss_fn(outputs, labels.float())

                    total_eval_loss += loss.item()
                    probabilities = torch.sigmoid(outputs)

                    for batch_idx in range(sequence.size(0)):
                        seq_id = seq_ids[batch_idx]
                        token_ids = sequence[batch_idx]
                        # Average attention weights across all layers and heads
                        attn_scores = torch.stack(attention_weights).mean(dim=0)[batch_idx]
                        attn_scores = attn_scores.sum(dim=0).cpu().tolist()  # Sum over sequence length

                        label = labels[batch_idx]
                        probability = probabilities[batch_idx]

                        if target_format != "multi-cat":
                            local_seq_data.append({
                                'seq_id': seq_id,
                                'probability': probability.cpu().item(),
                                'label': label.cpu().item(),
                                'token_ids': token_ids.cpu().tolist(),
                                'attn_scores': attn_scores
                            })
                        else:
                            local_seq_data.append({
                                'seq_id': seq_id,
                                'probability': probability.cpu(),
                                'label': label.cpu(),
                                'token_ids': token_ids.cpu().tolist(),
                                'attn_scores': attn_scores
                            })

            if distributed:
                total_eval_loss_tensor = torch.tensor([total_eval_loss], device=device)
                all_eval_loss_tensors = [torch.zeros_like(total_eval_loss_tensor) for _ in range(world_size)]
                dist.all_gather(all_eval_loss_tensors, total_eval_loss_tensor)
                combined_eval_loss = sum([tensor.item() for tensor in all_eval_loss_tensors])
                average_eval_loss = combined_eval_loss / (len(val_loader) * world_size)

                gathered_seq_data = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_seq_data, local_seq_data)

                if rank == 0:
                    all_seq_data = []
                    for rank_data in gathered_seq_data:
                        all_seq_data.extend(rank_data)
            else:
                average_eval_loss = total_eval_loss / len(val_loader)

            if rank == 0:
                if distributed:
                    all_seq_data = []
                    for rank_data in gathered_seq_data:
                        all_seq_data.extend(rank_data)
                else:
                    all_seq_data = local_seq_data

                final_labels = [item['label'] for item in all_seq_data]
                final_probabilities = [item['probability'] for item in all_seq_data]

                metrics_results = metrics_calculator.calculate_metrics_threshold(
                    final_labels,
                    final_probabilities=final_probabilities,
                    target_format=target_format,
                )
                (
                    f1,
                    accuracy,
                    hamming,
                    jaccard,
                    precision,
                    recall,
                    auc,
                    confusion,
                    class_report,
                    current_best_threshold,
                ) = metrics_results

                metrics_calculator.print_metrics_threshold(
                    logger,
                    average_eval_loss,
                    accuracy,
                    f1,
                    best_metric,
                    best_epoch,
                    hamming,
                    jaccard,
                    precision,
                    recall,
                    auc,
                    confusion,
                    class_report,
                    current_best_threshold,
                )

                final_probabilities = np.array(final_probabilities)  # Shape: [num_samples, num_classes]

                if isinstance(current_best_threshold, (float, int)):
                    final_predictions = (final_probabilities >= current_best_threshold).astype(int)
                else:
                    final_predictions = (final_probabilities >= current_best_threshold[np.newaxis, :]).astype(int)

                total_token_importance = defaultdict(float)
                total_token_counts = defaultdict(int)
                placeholder_token_ids = set()
                x_placeholder = tokenizer.encode("XXXXXXXXXXXXXXXXXX").ids[0]
                y_placeholder = tokenizer.encode("YYYYYYYYYYYY").ids[0]
                placeholder_token_ids.add(x_placeholder)
                placeholder_token_ids.add(y_placeholder)
                wrong_resistant = []

                for idx, item in enumerate(all_seq_data):
                    label = item['label']
                    prediction = final_predictions[idx]
                    token_ids = item['token_ids']
                    attn_scores = item['attn_scores']
                    if target_format != "multi-cat":
                        if label == 1 and prediction == 0:
                            wrong_resistant.append(item['seq_id'])
                    for token_id, importance in zip(token_ids, attn_scores):
                        if token_id in placeholder_token_ids or token_id == 0:
                            continue
                        total_token_importance[token_id] += importance
                        total_token_counts[token_id] += 1

                average_token_importance = {}
                for token_id in total_token_importance:
                    average_importance = total_token_importance[token_id] / total_token_counts[token_id]
                    average_token_importance[token_id] = average_importance

                if f1 > best_metric or epoch == 0:
                    best_metric = f1
                    best_epoch = epoch + 1
                    best_threshold = current_best_threshold
                    if target_format != "multi-cat":
                        logger.log(f"Best Threshold: {best_threshold}")
                    else:
                        logger.log(f"Best Thresholds: {best_threshold}")
                    # logger.log(f"Important Tokens: {important_tokens}")
                    with open(f"{args.save_path}/{args.antibiotic}/Gene_Token_Mapping.json", "r") as f:
                        mappings = json.load(f)
                        token_to_gene = mappings['token_to_gene']

                    # Analyze important tokens
                    gene_importance_sums = defaultdict(float)
                    gene_token_counts = defaultdict(int)
                    intergenic_importance_sums = defaultdict(float)
                    intergenic_token_counts = defaultdict(int)

                    for token_id, importance in average_token_importance.items():
                        token_id = str(token_id)
                        if token_id in token_to_gene:
                            info = token_to_gene[token_id]
                            gene_name = info['gene']
                            if info['is_intergenic']:
                                intergenic_importance_sums[gene_name] += importance
                                intergenic_token_counts[gene_name] += 1
                            else:
                                gene_importance_sums[gene_name] += importance
                                gene_token_counts[gene_name] += 1

                    # Compute average importance per gene
                    gene_importance = {}
                    for gene_name in gene_importance_sums:
                        gene_importance[gene_name] = gene_importance_sums[gene_name] / gene_token_counts[gene_name]

                    # Compute average importance per intergenic region
                    intergenic_importance = {}
                    for region_name in intergenic_importance_sums:
                        intergenic_importance[region_name] = intergenic_importance_sums[region_name] / \
                                                             intergenic_token_counts[region_name]

                    def normalize_importance(importance_dict):
                        values = np.array(list(importance_dict.values()))
                        mean = np.mean(values)
                        std = np.std(values)
                        return {
                            key: (value - mean) / std
                            for key, value in importance_dict.items()
                        }

                    # Calculate statistical significance
                    gene_scores = np.array(list(gene_importance.values()))
                    gene_mean = np.mean(gene_scores)
                    gene_std = np.std(gene_scores, ddof=1)  # Use ddof=1 for sample standard deviation

                    z_scores_genes = {
                        gene: (score - gene_mean) / gene_std
                        for gene, score in gene_importance.items()
                    }

                    # Identify significantly important genes (z-score > 1.96)
                    significant_genes = {
                        gene: z for gene, z in z_scores_genes.items() if z > 1.96
                    }

                    # Compute z-scores for intergenic importance
                    intergenic_scores = np.array(list(intergenic_importance.values()))
                    intergenic_mean = np.mean(intergenic_scores)
                    intergenic_std = np.std(intergenic_scores, ddof=1)

                    z_scores_intergenic = {
                        region: (score - intergenic_mean) / intergenic_std
                        for region, score in intergenic_importance.items()
                    }

                    # Identify significantly important intergenic regions (z-score > 1.96)
                    significant_intergenic = {
                        region: z for region, z in z_scores_intergenic.items() if z > 1.96
                    }

                    # Sort the significant genes and intergenic regions by z-score in descending order
                    sorted_significant_genes = sorted(significant_genes.items(), key=lambda x: x[1], reverse=True)
                    sorted_significant_intergenic = sorted(significant_intergenic.items(), key=lambda x: x[1],
                                                           reverse=True)

                    # Log statistically significant genes
                    if sorted_significant_genes:
                        logger.log("\nStatistically significant genes (z-score > 2):")
                        for gene, z in sorted_significant_genes:
                            logger.log(f"{gene}: z-score = {z:.4f}")
                    else:
                        logger.log("\nNo statistically significant genes found (z-score > 2).")

                    # Log statistically significant intergenic regions
                    if sorted_significant_intergenic:
                        logger.log("\nStatistically significant intergenic regions (z-score > 2):")
                        for region, z in sorted_significant_intergenic:
                            logger.log(f"{region}: z-score = {z:.4f}")

                    # Save most important genes to a file
                    genes_file_path = os.path.join(save_path, f'important_genes_fold_{fold + 1}.json')
                    with open(genes_file_path, 'w') as f:
                        json.dump(gene_importance, f, indent=4)

                    # Save most important intergenic regions to a file
                    intergenic_file_path = os.path.join(save_path, f'important_intergenic_regions_fold_{fold + 1}.json')
                    with open(intergenic_file_path, 'w') as f:
                        json.dump(intergenic_importance, f, indent=4)

                    # logger.log(f"Important Tokens: {important_tokens}")
                    model_save_path = os.path.join(save_path, f"best_model_fold_{fold + 1}.pth")
                    model_manager.save_model(model_save_path, best_threshold, logger)
                    logger.log(f"Wrong Resistant Isolates: {wrong_resistant}")
            if rank == 0:
                logger.log(f"Training Finished")

        if distributed:
            dist.destroy_process_group()
