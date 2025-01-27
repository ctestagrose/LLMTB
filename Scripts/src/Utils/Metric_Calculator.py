from sklearn.metrics import (
    f1_score, accuracy_score, hamming_loss, jaccard_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix,
    classification_report
)
import numpy as np

import math


class MetricsCalculator:

    def find_best_threshold(self, final_labels, final_probabilities, target_format="binary"):
        final_labels = np.array(final_labels)
        final_probabilities = np.array(final_probabilities)

        # thresholds = np.arange(0.0, 1.0, 0.01)
        # thresholds = np.arange(0.35, 0.65, 0.05)
        thresholds = np.arange(0.05, 0.95, 0.01)

        if target_format == "multi-cat":
            num_classes = final_labels.shape[1]
            best_thresholds = [0.5] * num_classes
            best_scores = [0.0] * num_classes

            for class_idx in range(num_classes):
                for threshold in thresholds:
                    preds = (final_probabilities[:, class_idx] >= threshold).astype(int)
                    score = f1_score(final_labels[:, class_idx], preds)
                    if score > best_scores[class_idx]:
                        best_scores[class_idx] = score
                        best_thresholds[class_idx] = threshold
            return np.array(best_thresholds), np.array(best_scores).mean()
        else:
            # Single-label case
            best_threshold = 0.5
            best_score = 0.0
            for threshold in thresholds:
                preds = (np.array(final_probabilities) >= threshold).astype(int)
                score = f1_score(final_labels, preds)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            return best_threshold, best_score


    def calculate_metrics_threshold(self, final_labels, final_predictions=None, final_probabilities=None,
                                    target_format="binary"):
        if target_format == "multi-cat":
            average_type = "macro"
        else:
            average_type = "binary"

        # If predictions are not provided, compute them using the best threshold
        if final_predictions is None and final_probabilities is not None:
            if target_format == "multi-cat":
                best_thresholds, best_score = self.find_best_threshold(final_labels, final_probabilities, target_format)
                final_predictions = (final_probabilities >= best_thresholds).astype(int)
                best_threshold = best_thresholds
            else:
                best_threshold, best_score = self.find_best_threshold(final_labels, final_probabilities, target_format)
                final_predictions = (np.array(final_probabilities) >= best_threshold).astype(int)

        f1 = f1_score(final_labels, final_predictions, average=average_type)
        accuracy = accuracy_score(final_labels, final_predictions)
        hamming = hamming_loss(final_labels, final_predictions)
        jaccard = jaccard_score(final_labels, final_predictions, average=average_type)
        precision = precision_score(final_labels, final_predictions, average=average_type)
        recall = recall_score(final_labels, final_predictions, average=average_type)
        roc_auc = None
        if final_probabilities is not None:
            if target_format == "multi-cat":
                roc_auc = roc_auc_score(final_labels, final_probabilities, multi_class='ovr', average=average_type)
            else:
                roc_auc = roc_auc_score(final_labels, final_probabilities)
        if target_format == "multi-cat":
            final_labels_np = np.array(final_labels)
            final_predictions_np = np.array(final_predictions)
            num_labels = final_labels_np.shape[1]
            confusion_matrices = []
            for i in range(num_labels):
                label_confusion = confusion_matrix(final_labels_np[:, i], final_predictions_np[:, i])
                confusion_matrices.append(label_confusion)
            confusion = confusion_matrices
        else:
            confusion = confusion_matrix(final_labels, final_predictions)
        class_report = classification_report(final_labels, final_predictions, digits=4)

        # Return the best threshold along with the metrics
        return f1, accuracy, hamming, jaccard, precision, recall, roc_auc, confusion, class_report, best_threshold


    def print_metrics_threshold(self, logger, average_eval_loss, accuracy, f1, best_metric, best_epoch,
                                hamming, jaccard, precision, recall, auc, confusion, class_report, best_threshold):
        logger.log(f"Average Validation Loss: {average_eval_loss}")
        logger.log(f"Validation Accuracy: {accuracy:.4f}")
        logger.log(f"Validation F1 Score: {f1:.4f}")
        logger.log(f"Precision: {precision:.4f} Recall: {recall:.4f}")
        if type(best_threshold) == str:
            logger.log(f"Threshold: {best_threshold:.2f}")
        else:
            logger.log(f"Threshold: {best_threshold}")
        if auc is not None:
            logger.log(f"AUC: {auc:.4f}")
        logger.log(f"Best F1 Score: {best_metric:.4f} at Epoch: {best_epoch}")
        logger.log(f"Hamming Loss: {hamming:.4f}")
        logger.log(f"Jaccard Score: {jaccard:.4f}")
        if isinstance(confusion, list):
            for i, cm in enumerate(confusion):
                logger.log(f"Confusion matrix for label {i}:\n{cm}\n")
        else:
            logger.log(f"Confusion Matrix:\n{confusion}")
        logger.log(f"Classification Report:\n{class_report}")


    def calculate_metrics(self, final_labels, final_predictions, final_probabilities=None, target_format="binary"):
        if target_format == "multi-cat":
            average_type = "macro"
        else:
            average_type = "binary"
        f1 = f1_score(final_labels, final_predictions, average=average_type)
        accuracy = accuracy_score(final_labels, final_predictions)
        hamming = hamming_loss(final_labels, final_predictions)
        jaccard = jaccard_score(final_labels, final_predictions, average=average_type)
        precision = precision_score(final_labels, final_predictions, average=average_type)
        recall = recall_score(final_labels, final_predictions, average=average_type)
        roc_auc = None
        if final_probabilities is not None:
            if target_format == "multi-cat":
                roc_auc = roc_auc_score(final_labels, final_probabilities, average=average_type, multi_class='ovo')
            else:
                roc_auc = roc_auc_score(final_labels, final_probabilities)
        if target_format == "multi-cat":
            final_labels_np = np.array(final_labels)
            final_predictions_np = np.array(final_predictions)
            num_labels = final_labels_np.shape[1]
            confusion_matrices = []
            for i in range(num_labels):
                label_confusion = confusion_matrix(final_labels_np[:, i], final_predictions_np[:, i])
                confusion_matrices.append(label_confusion)

            # Optionally, sum all confusion matrices
            # confusion = np.sum(confusion_matrices, axis=0)
            confusion = confusion_matrices
        else:
            confusion = confusion_matrix(final_labels, final_predictions)
        class_report = classification_report(final_labels, final_predictions, digits=4)
        return f1, accuracy, hamming, jaccard, precision, recall, roc_auc, confusion, class_report


    def print_eval_metrics(self, logger, accuracy, f1, best_metric, best_epoch,
                           hamming, jaccard, precision, recall, best_val_precision,
                           best_val_recall, auc, confusion, class_report):
        logger.log("-" * 20)
        logger.log("RESULTS ON TEST SET")
        logger.log(f"Validation Accuracy: {accuracy:.4f}")
        logger.log(f"Test F1 Score: {f1:.4f}")
        logger.log(f"Precision: {precision:.4f} Recall: {recall:.4f}")
        if auc is not None:
            logger.log(f"AUC: {auc:.4f}")
        logger.log(f"Best Validation F1 Score: {best_metric:.4f} at Epoch: {best_epoch}")
        logger.log(f"Best Validation Precision: {best_val_precision:.4f} Best Validation Recall: {best_val_recall:.4f}")
        logger.log(f"Hamming Loss: {hamming:.4f}")
        logger.log(f"Jaccard Score: {jaccard:.4f}")
        if len(confusion) > 2:
            for i in range(len(confusion)):
                logger.log(f"Confusion matrix for label {i}:\n{confusion[i]}\n")
        else:
            logger.log(f"Confusion Matrix:\n{confusion}")
        logger.log(f"Classification Report:\n{class_report}")