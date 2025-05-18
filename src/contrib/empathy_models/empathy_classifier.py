import torch
import numpy as np


import numpy as np


from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset

from models.models import BiEncoderAttentionWithRationaleClassification


class EmpathyClassifier():

    def __init__(self,
                 model_name,
                 device,
                 ER_model_path='output/sample.pth',
                 IP_model_path='output/sample.pth',
                 EX_model_path='output/sample.pth',
                 batch_size=1,
                 max_len=128):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=True)
        self.batch_size = batch_size
        self.device = device

        self.model_ER = BiEncoderAttentionWithRationaleClassification(
            model_name)
        self.model_IP = BiEncoderAttentionWithRationaleClassification(
            model_name)
        self.model_EX = BiEncoderAttentionWithRationaleClassification(
            model_name)

        ER_weights = torch.load(ER_model_path)
        self.model_ER.load_state_dict(ER_weights)

        IP_weights = torch.load(IP_model_path)
        self.model_IP.load_state_dict(IP_weights)

        EX_weights = torch.load(EX_model_path)
        self.model_EX.load_state_dict(EX_weights)

        self.model_ER.to(self.device)
        self.model_IP.to(self.device)
        self.model_EX.to(self.device)
        self.max_len = max_len

    def predict_empathy(self, seeker_posts, response_posts):

        input_ids_SP = []
        attention_masks_SP = []

        for sent in seeker_posts:

            encoded_dict = self.tokenizer.encode_plus(
                sent,                      # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                # Pad & truncate all sentences.
                max_length=self.max_len,
                truncation=True, padding='max_length',
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',     # Return pytorch tensors.
            )

            input_ids_SP.append(encoded_dict['input_ids'])
            attention_masks_SP.append(encoded_dict['attention_mask'])

        input_ids_RP = []
        attention_masks_RP = []

        for sent in response_posts:
            encoded_dict = self.tokenizer.encode_plus(
                sent,                      # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                # Pad & truncate all sentences.
                max_length=self.max_len,
                truncation=True, padding='max_length',
                return_attention_mask=True,   # Construct attn. masks.
                return_tensors='pt',     # Return pytorch tensors.
            )

            input_ids_RP.append(encoded_dict['input_ids'])
            attention_masks_RP.append(encoded_dict['attention_mask'])

        input_ids_SP = torch.cat(input_ids_SP, dim=0)
        attention_masks_SP = torch.cat(attention_masks_SP, dim=0)

        input_ids_RP = torch.cat(input_ids_RP, dim=0)
        attention_masks_RP = torch.cat(attention_masks_RP, dim=0)

        dataset = TensorDataset(
            input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP)

        dataloader = DataLoader(
            dataset,  # The test samples.
            # Pull out batches sequentially.
            sampler=SequentialSampler(dataset),
            batch_size=self.batch_size  # Evaluate with this batch size.
        )

        self.model_ER.eval()
        self.model_IP.eval()
        self.model_EX.eval()

        for batch in dataloader:
            b_input_ids_SP = batch[0].to(self.device)
            b_input_mask_SP = batch[1].to(self.device)
            b_input_ids_RP = batch[2].to(self.device)
            b_input_mask_RP = batch[3].to(self.device)

            with torch.no_grad():
                (logits_empathy_ER, logits_rationale_ER,) = self.model_ER(input_ids_SP=b_input_ids_SP,
                                                                          input_ids_RP=b_input_ids_RP,
                                                                          token_type_ids_SP=None,
                                                                          token_type_ids_RP=None,
                                                                          attention_mask_SP=b_input_mask_SP,
                                                                          attention_mask_RP=b_input_mask_RP)

                (logits_empathy_IP, logits_rationale_IP,) = self.model_IP(input_ids_SP=b_input_ids_SP,
                                                                          input_ids_RP=b_input_ids_RP,
                                                                          token_type_ids_SP=None,
                                                                          token_type_ids_RP=None,
                                                                          attention_mask_SP=b_input_mask_SP,
                                                                          attention_mask_RP=b_input_mask_RP)

                (logits_empathy_EX, logits_rationale_EX,) = self.model_EX(input_ids_SP=b_input_ids_SP,
                                                                          input_ids_RP=b_input_ids_RP,
                                                                          token_type_ids_SP=None,
                                                                          token_type_ids_RP=None,
                                                                          attention_mask_SP=b_input_mask_SP,
                                                                          attention_mask_RP=b_input_mask_RP)

            logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy().tolist()
            predictions_ER = np.argmax(logits_empathy_ER, axis=1).flatten()

            logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy().tolist()
            predictions_IP = np.argmax(logits_empathy_IP, axis=1).flatten()

            logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy().tolist()
            predictions_EX = np.argmax(logits_empathy_EX, axis=1).flatten()

            logits_rationale_ER = logits_rationale_ER.detach().cpu().numpy()
            predictions_rationale_ER = np.argmax(logits_rationale_ER, axis=2)

            logits_rationale_IP = logits_rationale_IP.detach().cpu().numpy()
            predictions_rationale_IP = np.argmax(logits_rationale_IP, axis=2)

            logits_rationale_EX = logits_rationale_EX.detach().cpu().numpy()
            predictions_rationale_EX = np.argmax(logits_rationale_EX, axis=2)

        return (logits_empathy_ER, predictions_ER,
                logits_empathy_IP, predictions_IP,
                logits_empathy_EX, predictions_EX,
                logits_rationale_ER, predictions_rationale_ER,
                logits_rationale_IP, predictions_rationale_IP,
                logits_rationale_EX, predictions_rationale_EX)
