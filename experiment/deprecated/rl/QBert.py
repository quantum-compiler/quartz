import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler


class QBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_embeddings = nn.Embedding(
            config.vocab_size,
            int(config.hidden_size / 2),
            padding_idx=config.pad_token_id,
        )
        self.qubit_embeddings = nn.Embedding(
            config.max_position_embeddings, int(config.hidden_size / 4)
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.gate_embeddings(input_ids)

        qubit_embeds = self.qubit_embeddings(token_type_ids)
        qubit2_embeds = self.qubit_embeddings(position_ids)

        qubits_embeds = torch.cat([qubit_embeds, qubit2_embeds], dim=-1)
        embeddings = torch.cat([inputs_embeds, qubits_embeds], dim=-1)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = QBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()


class QBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = QBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
