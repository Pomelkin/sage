import torch.nn as nn

from .multilabel import T5ForConditionalGenerationTokenMultilabel
from .encoder_task import T5ForConditionalGenerationEncoderTask


class T5ForConditionalGenerationTokenMultilabelLM(T5ForConditionalGenerationTokenMultilabel):
    def compute_encoder_loss(
            self,
            input_ids=None,
            attention_mask=None,
            label_attention_mask=None,
            encoder_outputs=None,
            labels=None,
            encoder_labels=None,
            label_ids=None,
            encoder_lm_labels=None
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = encoder_outputs[0]
        token_logits = self.clf_head(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        encoder_loss = None
        if encoder_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            encoder_labels = encoder_labels.to(token_logits.device)
            encoder_loss = loss_fct(token_logits, encoder_labels.float())

        if encoder_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            encoder_lm_labels = encoder_lm_labels.to(lm_logits.device)
            encoder_loss += loss_fct(lm_logits.view(-1, lm_logits.size(-1)), encoder_lm_labels.view(-1))
        return encoder_outputs, encoder_loss, token_logits


class T5ForConditionalGenerationLM(T5ForConditionalGenerationEncoderTask):
    def compute_encoder_loss(
            self,
            input_ids=None,
            attention_mask=None,
            label_attention_mask=None,
            encoder_outputs=None,
            labels=None,
            encoder_labels=None,
            label_ids=None,
            encoder_lm_labels=None
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = encoder_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        encoder_lm_labels = encoder_lm_labels.to(lm_logits.device)
        encoder_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), encoder_lm_labels.view(-1))
        return encoder_outputs, encoder_loss, None
