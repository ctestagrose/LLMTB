import torch
from ..Models.Ensemble.BERT_Ensemble import BERT_Ensemble_Boosting as Ensemble
from ..Models.BERT.BERT_Updated import BERT

class ModelManager:
    def __init__(self, vocab_size, config):
        # Set Default to BERT if not "ensemble"
        self.model_type = config.get('model_type', 'BERT')
        self.model = self.initialize_model(vocab_size, config)

    def initialize_ensemble_model(self, vocab_size, config):
        return Ensemble(vocab_size, config)

    def initialize_model(self, vocab_size, config):
        if self.model_type == 'ensemble':
            return self.initialize_ensemble_model(vocab_size, config)
        else:
            return BERT(vocab_size, config)

    def save_model(self, save_path, logger):
        torch.save(self.model.state_dict(), save_path)
        logger.log("NEW MODEL SAVED")

    def load_model(self, model_path, device_ids=None):
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        if device_ids and len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device_ids[0] if device_ids else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self.model

    def save_model_threshold(self, save_path, best_threshold, logger):
        # Prepare the state dictionary
        state = {
            'model_state_dict': self.model.state_dict(),
            'best_threshold': best_threshold
        }
        torch.save(state, save_path)
        logger.log(f"Model and threshold saved to {save_path}")

    def load_model_threshold(self, model_path, device_ids=None):
        state = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state['model_state_dict'])
        if device_ids and len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device_ids[0] if device_ids else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        best_threshold = state.get('best_threshold', None)
        return self.model, best_threshold