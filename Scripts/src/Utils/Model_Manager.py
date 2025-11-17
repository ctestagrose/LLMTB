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

    def load_model(self, model_path, device=None, device_ids=None):
        state = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = state.get("model_state_dict", state)
        best_threshold = state.get("best_threshold", state.get("threshold", None))
        if device is None:
            if torch.cuda.is_available():
                if device_ids and len(device_ids) > 0:
                    device = torch.device(f"cuda:{device_ids[0]}")
                else:
                    device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.model.to(device)
        self.model.load_state_dict(state_dict)
        if device.type == "mps":
            for p in self.model.parameters():
                p.data = p.data.clone().to(device)
        if device.type == "cuda" and device_ids and len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.model.eval()
        return self.model, best_threshold