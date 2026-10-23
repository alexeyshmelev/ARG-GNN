import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

#torch.autograd.set_detect_anomaly(True)

#the majority of the following code was adapted from the GRPE repo 
#https://github.com/lenscloth/GRPE

#this version is compatible with TorchScript 
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout_rate: float):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def infer_dim_feedforward_mult_from_state_dict(
    state_dict: dict,
    d_model: int,
    *,
    default: int = 4,
) -> int:
    """Infer FFN multiplier from transformer weights when checkpoint metadata is absent."""
    weight = state_dict.get("layers.0.ffn.linear1.weight")
    if weight is None or not hasattr(weight, "shape"):
        return int(default)
    ffn_size = int(weight.shape[0])
    hidden_size = int(d_model)
    if hidden_size > 0 and ffn_size > 0 and ffn_size % hidden_size == 0:
        return ffn_size // hidden_size
    return int(default)


class SpatialBiasAttention(nn.Module):
    """
    Multi-head attention with spatial bias from provided TMRCA distance matrix.
    """
    def __init__(
        self,
        hidden_size: int,
        attn_dropout_rate: float,
        num_heads: int,
    ):
        super(SpatialBiasAttention, self).__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads #size of attention 
        self.scale = self.head_dim ** -0.5 #sqrt d_z 

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim) #
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim) 
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        self.att_dropout = nn.Dropout(attn_dropout_rate)

        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        key_hop_emb,
        value_hop_emb,
        distance,
        mask=None,
    ):
        orig_q_size = q.size() 
        
        d_k = self.head_dim 
        d_v = self.head_dim 

        batch_size = q.size(0)

        q = self.q_proj(q).view(batch_size, -1, self.num_heads, d_k)
        k  = self.k_proj(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2) # [b, h, q_len, d_k]
        v = v.transpose(1, 2) # [b, h, v_len, d_v]
        k = k.transpose(1, 2) # [b, h, d_k, k_len]

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0] #my understanding is that this is size 1 for us

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.num_heads, self.head_dim
        ).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2,3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2,3))
        key_hop = torch.gather(
            key_hop, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        )

        #spatial_bias = query_hop + key_hop 
        #disable spatial bias 
        spatial_bias = key_hop + query_hop
        #spatial_bias = 0

        x = torch.matmul(q, k.transpose(2, 3)) + spatial_bias 

        x = x * self.scale 

        #Need this for CLS token at least, we don't want attention on CLS token  
        '''
        if mask is not None:
            x = x.masked_fill(
                mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf")
            )
        '''
        #inverting this to match rest of argmix, this is different than GRPE implementation 
        if mask is not None:
            # mask: True = valid token; we want to mask out PADs, invert it
            mask_expanded = mask.view(mask.size(0), 1, 1, mask.size(1))
            x = x.masked_fill(
                ~mask_expanded,      # now True at PAD positions
                float("-inf")
            )

        attn = torch.softmax(x, dim=3) #save attention!
        x = self.att_dropout(attn)

        value_hop_emb = value_hop_emb.view(
            1, num_hop_types, self.num_heads, self.head_dim
        ).transpose(1,2)

        value_hop_att = torch.zeros(
            (batch_size, self.num_heads, sequence_length, num_hop_types),
            device=x.device, dtype=x.dtype
        )
        value_hop_att = torch.scatter_add(
            value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1), x
        )

        #Graph-encoded value
        x = (
            torch.matmul(x, v)
            + torch.matmul(value_hop_att, value_hop_emb)
        )

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.out_proj(x)
        assert x.size() == orig_q_size

        return x


#this uses SpatialBiasAttention
class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        dropout_rate: float ,
        attn_dropout_rate: float,
        num_heads: int
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = SpatialBiasAttention(
            hidden_size,
            attn_dropout_rate,
            num_heads
        )
        #self.self_attention = DummyAttn(hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x,
        query_hop_emb,
        key_hop_emb,
        value_hop_emb,
        distance,
        mask=None,
    ):
        # ---- self‑attention block --------------------------------------------
        y = self.self_attention_norm(x)
        y = self.self_attention(
            y, y, y,
            query_hop_emb, key_hop_emb, value_hop_emb,
            distance, mask=mask
        )
        y = self.self_attention_dropout(y)
        x = x + y                          
        # ---- FFN block --------------------------------------------------------
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        # ---- output -----------------------------------------------------------
        return x




#for now we're dropping perturb noise
class TmrcaAttn(nn.Module):
    """
    Transformer classifier using full TMRCA matrix including sample at index 0.
    """
    def __init__(
      self,
      vocab_size: int, #this replaces num_node_type in GRPE implementation 
      num_classes: int, #number of classes we're actually predicting 
      num_layer: int = 4,
      d_model: int = 256,
      num_heads: int = 4,
      dim_feedforward_mult: int = 4,
      dropout: float = 0.1,
      attn_dropout: float = 0.1,
      cls_token: bool = True,
      max_rel_distance: int = 1024, #in generations
      gt_output_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.dim_feedforward_mult = int(dim_feedforward_mult)
        self.gt_output_dim = int(gt_output_dim) if gt_output_dim is not None else self.d_model
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.cls_token_flag = cls_token
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        #cls distance is farther than anything else
        self.cls_distance = max_rel_distance+1

        self.query_hop_emb = nn.Embedding(max_rel_distance + 3, d_model)
        self.key_hop_emb   = nn.Embedding(max_rel_distance + 3, d_model)
        self.value_hop_emb = nn.Embedding(max_rel_distance + 3, d_model)

        
        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=d_model,
                ffn_size=d_model * self.dim_feedforward_mult,
                dropout_rate=dropout,
                attn_dropout_rate=attn_dropout,
                num_heads=num_heads
            )
            for _ in range(num_layer)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        if self.gt_output_dim == self.d_model:
            self.feature_proj = nn.Identity()
        else:
            self.feature_proj = nn.Linear(self.d_model, self.gt_output_dim)

    def encode(
        self,
        input_ids: torch.LongTensor,
        tmrca_mat: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, num_nodes) population token IDs
            tmrca_mat: (B, num_nodes, num_nodes) full TMRCA distances (sample+refs)
            mask: (B, num_nodes) mask over reference tokens
        """

        #token embeddings
        batch_size, seq_len = input_ids.size() # (B, L) 
        embeddings = self.embeddings(input_ids) # (B, L, H)

        #we could add perturb if training here
        # ─── CLAMP + cast for embedding indices ────────────────
        max_idx   = self.query_hop_emb.num_embeddings - 1
        tmrca_mat = tmrca_mat.clamp(0, max_idx)
        if tmrca_mat.dtype != torch.long:
            tmrca_mat = tmrca_mat.long()

        #add our CLS token to beginning of sequence
        assert self.cls_token_flag == True #for now 
        
        cls_tok = self.cls_token.expand(batch_size, -1, -1) # Shape: (B, 1, H)

        embeddings = torch.cat([cls_tok, embeddings], dim=1)

        assert mask is not None
        mask = F.pad(mask, (1, 0), value=True)
            
        seq_len = seq_len + 1 

        # Row: distances from CLS to other nodes
        # Create tensors using tensor operations to preserve device in TorchScript
        cls_row = tmrca_mat[:, :1, :].clone().fill_(self.cls_distance)
        cls_row = cls_row.expand(batch_size, 1, tmrca_mat.size(2))
        tmrca_mat = torch.cat([cls_row, tmrca_mat], dim=1)

        # Column: distances from other nodes to CLS
        # Create tensors using tensor operations to preserve device in TorchScript
        cls_col = tmrca_mat[:, :, :1].clone().fill_(self.cls_distance)
        cls_col = cls_col.expand(batch_size, tmrca_mat.size(1), 1)
        tmrca_mat = torch.cat([cls_col, tmrca_mat], dim=2)

        #set self-distance for CLS to zero
        tmrca_mat[:, 0, 0] = 0

        #transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                self.query_hop_emb.weight,
                self.key_hop_emb.weight,
                self.value_hop_emb.weight,
                tmrca_mat,
                mask=mask
            )

        out = self.final_ln(hidden_states[:, 0])
        out = self.dropout(out)
        return out

    def forward_features(
        self,
        input_ids: torch.LongTensor,
        tmrca_mat: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        out = self.encode(input_ids, tmrca_mat, mask)
        return self.feature_proj(out)

    def forward_with_features(
        self,
        input_ids: torch.LongTensor,
        tmrca_mat: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        out = self.encode(input_ids, tmrca_mat, mask)
        logits = self.classifier(out)
        features = self.feature_proj(out)
        return logits, features

    def forward(
        self,
        input_ids: torch.LongTensor,
        tmrca_mat: torch.LongTensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        out = self.encode(input_ids, tmrca_mat, mask)
        logits = self.classifier(out)
        return logits 
        

