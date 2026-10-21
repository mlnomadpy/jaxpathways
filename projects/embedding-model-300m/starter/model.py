"""Learner starter for the bidirectional SciFact encoder."""


def initialize(key, config):
    """TODO: create the complete tied-MLM parameter tree."""
    raise NotImplementedError("implement bidirectional encoder initialization")


def parameter_count(config, *, include_mlm=False):
    """TODO: count the trainable leaves implied by config."""
    raise NotImplementedError("count the encoder and optional tied MLM head")


def hidden_states(params, input_ids, config, valid_mask=None, *, dropout_key=None):
    """TODO: return final pre-norm Transformer token states."""
    raise NotImplementedError("implement masked bidirectional RoPE attention")


def mlm_logits(params, input_ids, config, valid_mask=None, *, dropout_key=None):
    """TODO: return tied-vocabulary masked-token logits."""
    raise NotImplementedError("implement the dense-GELU-norm tied MLM head")


def pool(params, input_ids, config, valid_mask=None, *, dropout_key=None):
    """TODO: return normalized content-token mean vectors."""
    raise NotImplementedError("implement special-token-aware mean pooling")


def masked_token_loss(params, input_ids, labels, loss_mask, config, dropout_key=None):
    """TODO: return mean NLL, selected-token NLL sum, and selected count."""
    raise NotImplementedError("implement selected-token MLM cross entropy")
