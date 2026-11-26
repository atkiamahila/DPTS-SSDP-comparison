# inferenceKit/models/prm/qwen.py

import torch
from transformers import AutoTokenizer, AutoModel

from .base import BaseProcessRewardModel
from inferenceKit.utils import InferenceConfig
import copy

class QwenPRM(BaseProcessRewardModel):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)

        self.step_token = self.config.step_token
        # Qwen2.5-Math-PRM uses a dedicated separator token, not a good/bad token pair.
        self.step_token_id = self.tokenizer.encode(self.step_token, add_special_tokens=False)[0]

    def _default_config(self, base_config=None):
        config = super()._default_config(base_config)
        config.step_token = "<extra_0>"
        return config

    def _load_model(self, model_path):
        model = AutoModel.from_pretrained(
            model_path,
            device_map=self.device_map,
            torch_dtype=self.config.dtype,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer


    @torch.inference_mode()
    def score(self, query, inference_config: InferenceConfig):
        if not inference_config.config.mini_step:
            for step_token in inference_config.llm_config.step_tokens:
                query = list(map(
                    lambda x: x.replace(step_token, step_token + inference_config.prm_config.step_token),
                    copy.deepcopy(query)
                ))

            input_ids = self.tokenizer(query, padding=True, return_tensors="pt")["input_ids"].to(self.device)
            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs[0]
            probs = logits.softmax(dim=-1)[:, :, 1]

            step_scores = []
            for prob, input_id in zip(probs.unbind(0), input_ids.unbind(0)):
                step_score = prob[input_id == self.step_token_id]
                step_score = step_score.to(torch.float32).detach().cpu().numpy().tolist()
                step_scores.append(step_score)
            return step_scores
        else:
            query = list(map(lambda x: x + inference_config.prm_config.step_token, copy.deepcopy(query)))
            input_ids = self.tokenizer(query, padding=True, return_tensors="pt")["input_ids"].to(self.device)
            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs[0]
            probs = logits.softmax(dim=-1)[:, :, 1]

            step_scores = []
            for prob in probs.unbind(0):
                step_score = prob.to(torch.float32).detach().cpu().numpy().tolist()
                step_scores.append(step_score)
            return step_scores

