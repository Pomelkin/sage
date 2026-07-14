import torch.nn as nn

from .encoder_task import T5ForConditionalGenerationEncoderTask


class T5ForConditionalGenerationTokenMultilabel(T5ForConditionalGenerationEncoderTask):
    def __init__(self, config):
        super().__init__(config)
        self.clf_head = nn.Linear(config.d_model, 4, bias=False)

    def compute_encoder_loss(
            self,
            input_ids=None,
            attention_mask=None,
            label_attention_mask=None,
            encoder_outputs=None,
            labels=None,
            encoder_labels=None,
            encoder_lm_labels=None,
            label_ids=None
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = encoder_outputs[0]
        token_logits = self.clf_head(hidden_states)
        encoder_loss = None
        if encoder_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            encoder_labels = encoder_labels.to(token_logits.device)
            encoder_loss = loss_fct(token_logits, encoder_labels.float())
        return encoder_outputs, encoder_loss, token_logits
