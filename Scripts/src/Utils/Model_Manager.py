import torch
from ..Models.BERT.BERT import BERT

class ModelManager:
    def __init__(self, vocab_size, config):
        # Set Default to BERT if not "ensemble"
        self.model = self.initialize_model(vocab_size, config)

    def initialize_model(self, vocab_size, config):
        return BERT(vocab_size, config)


    def save_model(self, save_path, best_threshold, logger):
        # Prepare the state dictionary
        state = {
            'model_state_dict': self.model.state_dict(),
            'best_threshold': best_threshold
        }
        torch.save(state, save_path)
        logger.log(f"Model and threshold saved to {save_path}")


    def load_model(self, model_path, device_ids=None):
        state = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state['model_state_dict'])
        if device_ids and len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device_ids[0] if device_ids else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        best_threshold = state.get('best_threshold', None)
        return self.model, best_threshold