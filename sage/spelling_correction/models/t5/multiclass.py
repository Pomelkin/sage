import torch.nn as nn

from .encoder_task import T5ForConditionalGenerationEncoderTask


class T5ForConditionalGenerationTokenMulticlass(T5ForConditionalGenerationEncoderTask):
    def __init__(self, config):
        super().__init__(config)
        self.clf_head = nn.Linear(config.d_model, 4, bias=False)

    def compute_encoder_loss(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            labels=None,
            encoder_labels=None,
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
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            encoder_labels = encoder_labels.to(token_logits.device)
            encoder_loss = loss_fct(token_logits.view(-1, token_logits.size(-1)), encoder_labels.view(-1))
        return encoder_loss
