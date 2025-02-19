import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

#
# 1) 量子化用クラス
#

class MinMaxObserver(nn.Module):
    """
    入力テンソルの min/max を追跡し、FakeQuantize が利用する scale, zero_point を更新する Observer。
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            min_x, max_x = x.min(), x.max()
            self.min_val = torch.minimum(self.min_val, min_x)
            self.max_val = torch.maximum(self.max_val, max_x)
        return x


class FakeQuantize(nn.Module):
    """
    非対称量子化: [0 ~ 2^bit - 1]
    学習時に Observer で min_val,max_val を更新しつつ FakeQuant を行う。
    推論時は学習した min_val,max_val で量子化する。
    """
    def __init__(self, observer: MinMaxObserver, bit_width=8):
        super().__init__()
        self.observer = observer
        self.bit_width = bit_width

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0))

        self.quant_min = 0
        self.quant_max = 2**bit_width - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.observer(x)
        min_val = self.observer.min_val
        max_val = self.observer.max_val

        if (max_val - min_val) < 1e-8:
            return x  # レンジが極小なら量子化しない

        scale = (max_val - min_val) / float(self.quant_max - self.quant_min)
        zero_point = self.quant_min - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, self.quant_min, self.quant_max)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

        x_int = torch.clamp(torch.round(x / scale + zero_point), self.quant_min, self.quant_max)
        x_q = (x_int - zero_point) * scale
        return x_q


class ChannelWiseMinMaxObserver(nn.Module):
    """
    weightを行方向 (out_features方向) 単位で min/max を追跡する Observer。
    """
    def __init__(self, out_channels: int, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.register_buffer("min_vals", torch.full((out_channels,), float("inf")))
        self.register_buffer("max_vals", torch.full((out_channels,), float("-inf")))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.training:
            mins = w.min(dim=1)[0]
            maxs = w.max(dim=1)[0]
            self.min_vals = torch.minimum(self.min_vals, mins)
            self.max_vals = torch.maximum(self.max_vals, maxs)
        return w


class ChannelWiseFakeQuantize(nn.Module):
    """
    重みを対称量子化 (-2^(bit-1) ~ 2^(bit-1)-1) で行方向単位に量子化する。
    """
    def __init__(self, observer: ChannelWiseMinMaxObserver, bit_width=8):
        super().__init__()
        self.observer = observer
        self.bit_width = bit_width

        out_channels = observer.min_vals.shape[0]
        self.register_buffer("scale", torch.ones(out_channels))
        self.register_buffer("zero_point", torch.zeros(out_channels))

        self.quant_min = -(2 ** (bit_width - 1))
        self.quant_max = (2 ** (bit_width - 1)) - 1

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w = self.observer(w)

        min_vals = self.observer.min_vals
        max_vals = self.observer.max_vals

        ranges = torch.maximum(min_vals.abs(), max_vals.abs())
        scales = ranges / float(self.quant_max)
        scales = torch.clamp(scales, min=1e-8)
        zero_points = torch.zeros_like(scales)

        self.scale.copy_(scales)
        self.zero_point.copy_(zero_points)

        w_q_list = []
        for i in range(w.shape[0]):
            scale_i = scales[i]
            zp_i = zero_points[i]
            w_i = w[i]
            w_int = torch.clamp(torch.round(w_i / scale_i + zp_i), self.quant_min, self.quant_max)
            w_q_i = (w_int - zp_i) * scale_i
            w_q_list.append(w_q_i)
        w_q = torch.stack(w_q_list, dim=0)
        return w_q


class AdvancedBitLinear(nn.Module):
    """
    - 重み: ChannelWiseFakeQuantize
    - 活性化(入力/出力): FakeQuantize
    - バイアスはFP32のまま
    """
    def __init__(self, in_features: int, out_features: int, bit_width=8, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.weight_observer = ChannelWiseMinMaxObserver(out_features)
        self.weight_fake_quant = ChannelWiseFakeQuantize(self.weight_observer, bit_width=bit_width)

        self.act_in_observer = MinMaxObserver()
        self.act_in_fake_quant = FakeQuantize(self.act_in_observer, bit_width=bit_width)

        self.act_out_observer = MinMaxObserver()
        self.act_out_fake_quant = FakeQuantize(self.act_out_observer, bit_width=bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_in_fake_quant(x)
        w_q = self.weight_fake_quant(self.weight)
        out = F.linear(x, w_q, self.bias)
        out = self.act_out_fake_quant(out)
        return out


#
# 2) BERT風の設定クラス
#

class BitNetConfig:
    """
    BERT相当の構成を想定。BertConfigに相当。
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        bit_width=8
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.bit_width = bit_width


#
# 3) BERT Embeddings
#

class BitBertEmbeddings(nn.Module):
    """
    BERT同様: word embedding + position embedding + token_type embedding
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, seq_len)

        if token_type_ids is None:
            token_type_ids = torch.zeros((bsz, seq_len), dtype=torch.long, device=input_ids.device)

        word_embed = self.word_embeddings(input_ids)
        pos_embed = self.position_embeddings(position_ids)
        tok_embed = self.token_type_embeddings(token_type_ids)

        embeddings = word_embed + pos_embed + tok_embed
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


#
# 4) エンコーダ (Self-Attention + FeedForward)
#

class BitBertSelfAttention(nn.Module):
    """
    簡易的に nn.MultiheadAttention を用いて Self-Attention
    入出力に FakeQuant を挟んで量子化をシミュレートする例。
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=False  # デフォルトがFalse
        )

        self.in_observer = MinMaxObserver()
        self.in_fake_quant = FakeQuantize(self.in_observer, bit_width=config.bit_width)

        self.out_observer = MinMaxObserver()
        self.out_fake_quant = FakeQuantize(self.out_observer, bit_width=config.bit_width)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden_states: (seq_len, batch, hidden_size)
        q = k = self.in_fake_quant(hidden_states)

        # attention_mask が (batch, seq_len) のboolマスクの場合、key_padding_mask=attention_mask
        # ただし BERTの 1=利用, 0=padding とは逆かもしれないので実際の環境に合わせて反転など必要
        attn_output, _ = self.attn(q, k, hidden_states, key_padding_mask=attention_mask)
        attn_output = self.out_fake_quant(attn_output)
        return attn_output


class BitBertSelfOutput(nn.Module):
    """
    Attention出力に対して dropout + residual + layernorm
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BitBertAttention(nn.Module):
    """
    BERTの SelfAttention + SelfOutput
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.self = BitBertSelfAttention(config)
        self.output = BitBertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BitBertIntermediate(nn.Module):
    """
    FeedForward前半: hidden -> intermediate_size -> 活性化
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.dense = AdvancedBitLinear(config.hidden_size, config.intermediate_size, bit_width=config.bit_width)
        self.intermediate_act_fn = F.gelu  # BERT既定は gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BitBertOutput(nn.Module):
    """
    FeedForward後半: intermediate -> hidden + Residual + LayerNorm
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.dense = AdvancedBitLinear(config.intermediate_size, config.hidden_size, bit_width=config.bit_width)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BitBertLayer(nn.Module):
    """
    1層分: Attention + FeedForward
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.attention = BitBertAttention(config)
        self.intermediate = BitBertIntermediate(config)
        self.output = BitBertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BitBertEncoder(nn.Module):
    """
    BERTエンコーダ: BitBertLayer を複数積む
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.layer = nn.ModuleList([BitBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


#
# 5) MaskedLM Head + 全体モデル
#

class BitBertLMPredictionHead(nn.Module):
    """
    MaskedLM用: hidden -> vocab_size
    BERTではもう少し複雑(LayerNormなど)だが簡略化。
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.dense = AdvancedBitLinear(config.hidden_size, config.vocab_size, bit_width=config.bit_width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.dense(hidden_states)  # (batch, seq_len, vocab_size)
        return logits


class BitNetForMaskedLM(nn.Module):
    """
    BertForMaskedLM に相当するインターフェースを持つ量子化対応モデル
    """
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config

        self.embeddings = BitBertEmbeddings(config)
        self.encoder = BitBertEncoder(config)
        self.lm_head = BitBertLMPredictionHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        BertForMaskedLM と似たシグネチャ。
        """
        # Embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )  # (batch, seq_len, hidden_size)

        # nn.MultiheadAttention はデフォルト (seq_len, batch, hidden_size)
        hidden_states = embedding_output.transpose(0, 1)

        # attention_mask が (batch, seq_len) で 1=有効,0=無効の場合は以下の反転などが必要かも
        # attention_mask = (attention_mask == 0)  # True/Falseでマスク
        encoder_output = self.encoder(hidden_states, attention_mask=attention_mask)

        # 戻す
        sequence_output = encoder_output.transpose(0, 1)  # (batch, seq_len, hidden_size)

        # LM 頭
        logits = self.lm_head(sequence_output)  # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            # CrossEntropyLoss(ignore_index=-100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss, logits)
            else:
                return (logits, )

        return {
            "loss": loss,
            "logits": logits
        }
