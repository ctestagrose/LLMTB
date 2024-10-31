import torch
from torch import nn
from ..BERT.BERT import BERT
import torch.nn.functional as F

class BERT_Ensemble(nn.Module):
    def __init__(self, vocab_size, bert_config, aggregation='mean'):
        super(BERT_Ensemble, self).__init__()

        num_models = len(bert_config['hidden_dim'])

        self.models = nn.ModuleList([
            BERT(
                vocab_size=vocab_size,
                config={ 
                    'hidden_dim': bert_config['hidden_dim'][i],
                    'num_heads': bert_config['num_heads'][i],
                    'ff_dim': bert_config['ff_dim'][i],
                    'num_blocks': bert_config['num_blocks'][i],
                    'block_size': bert_config['block_size'][i],
                    'num_layers': bert_config['num_layers'][i],
                    'num_class': bert_config['num_class'][i],
                    'dropout': bert_config['dropout'][i],
                    'include_adjacent': bert_config['include_adjacent'][i]
                }
            )
            for i in range(num_models)
        ])
        self.aggregation = aggregation

    def forward(self, x, mask=None):
        outputs = []

        for model in self.models:
            model_out, attention_weights = model(x, mask)
            outputs.append(model_out)

        outputs = torch.stack(outputs, dim=0)

        if self.aggregation == 'mean':
            output = outputs.mean(dim=0)
        elif self.aggregation == 'sum':
            output = outputs.sum(dim=0)
        elif self.aggregation == 'max':
            output, _ = outputs.max(dim=0)
        elif self.aggregation == 'min':
            output, _ = outputs.min(dim=0)
        elif self.aggregation == 'majority':
            output = torch.mean(outputs, dim=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return output
        
        
class BERT_Ensemble_Boosting(nn.Module):
    def __init__(self, vocab_size, bert_config, aggregation='weighted_sum'):
        super(BERT_Ensemble_Boosting, self).__init__()

        num_models = len(bert_config['hidden_dim'])
        self.models = nn.ModuleList([
            BERT(
                vocab_size=vocab_size,
                config={
                    'hidden_dim': bert_config['hidden_dim'][i],
                    'num_heads': bert_config['num_heads'][i],
                    'ff_dim': bert_config['ff_dim'][i],
                    'num_blocks': bert_config['num_blocks'][i],
                    'block_size': bert_config['block_size'][i],
                    'num_layers': bert_config['num_layers'][i],
                    'num_class': bert_config['num_class'][i],
                    'dropout': bert_config['dropout'][i],
                    'include_adjacent': bert_config['include_adjacent'][i]
                }
            )
            for i in range(num_models)
        ])
        self.aggregation = aggregation
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, mask=None):
        outputs = []
        attention_weights_first_model = None

        for idx, model in enumerate(self.models):
            output, attention_weights = model(x, mask)
            outputs.append(output)
            if idx == 0:
                attention_weights_first_model = attention_weights

        # Apply softmax to weights for normalization
        normalized_weights = self.softmax(self.model_weights)
        weighted_outputs = torch.stack([weight * output for weight, output in zip(normalized_weights, outputs)], dim=0)
        output = weighted_outputs.sum(dim=0)

        return output, attention_weights_first_model

        # return output, all_preds

    # def forward(self, x, mask=None):
    #     residuals = None
    #     outputs = []

    #     for i, model in enumerate(self.models):
    #         if i == 0:
    #             output = model(x, mask)
    #         else:
    #             output = model(x, mask) + residuals

    #         outputs.append(output)
    #         residuals = output - torch.mean(torch.stack(outputs), dim=0)

    #     outputs = torch.stack(outputs, dim=0)

    #     if self.aggregation == 'weighted_sum':
    #         weighted_outputs = outputs * self.model_weights.view(-1, 1, 1)
    #         output = weighted_outputs.sum(dim=0)
    #     elif self.aggregation == 'mean':
    #         output = outputs.mean(dim=0)
    #     else:
    #         raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

    #     return output
