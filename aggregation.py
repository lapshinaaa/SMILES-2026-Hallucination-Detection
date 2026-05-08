"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)``.

    Student task:
        This version keeps the representation focused on the last real token,
        but aggregates it across the top 3 transformer layers using a weighted
        average. The intuition is that the final layer may contain the most
        task-relevant signal, while the previous layers provide a stabilizing
        context.

    """

    # ------------------------------------------------------------------
    # STUDENT: Weighted top-3 last-token aggregation.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero(as_tuple=False)   
    last_pos = int(real_positions[-1].item())                 

    top3_last_tokens = hidden_states[-3:, last_pos, :]       

    weights = torch.tensor(
        [0.3, 0.3, 0.4], # empirically chosen weights
        dtype=top3_last_tokens.dtype,
        device=top3_last_tokens.device,
    )                                                         

    feature = (top3_last_tokens * weights.unsqueeze(1)).sum(dim=0)  

    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.py``. The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(9,)``.
    """
    # ------------------------------------------------------------------
    # STUDENT: Compact geometric features.
    # ------------------------------------------------------------------

    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    real_positions = attention_mask.nonzero(as_tuple=False)   
    last_pos = int(real_positions[-1].item())                 

    
    seq_len_feature = attention_mask.sum().float()            

    
    top3_last_tokens = hidden_states[-3:, last_pos, :]        

    # norm statistics
    top3_norms = top3_last_tokens.norm(dim=1)                 
    final_norm = top3_last_tokens[-1].norm()                  
    mean_norm = top3_norms.mean()                             
    std_norm = top3_norms.std(unbiased=False)                 

    # consine similarity
    cos_last2 = torch.nn.functional.cosine_similarity(
        top3_last_tokens[-1].unsqueeze(0),
        top3_last_tokens[-2].unsqueeze(0),
        dim=1,
    ).squeeze(0)                                              

    # some extra features for richness 
    cos_last13 = torch.nn.functional.cosine_similarity(
        top3_last_tokens[-1].unsqueeze(0),
        top3_last_tokens[-3].unsqueeze(0),
        dim=1,
    ).squeeze(0)                                              

    l2_last12 = torch.norm(top3_last_tokens[-1] - top3_last_tokens[-2], p=2)
    l2_last13 = torch.norm(top3_last_tokens[-1] - top3_last_tokens[-3], p=2)

    norm_ratio = final_norm / (mean_norm + 1e-6)

    return torch.stack(
        [
            seq_len_feature,
            final_norm,
            mean_norm,
            std_norm,
            cos_last2,
            cos_last13,
            l2_last12,
            l2_last13,
            norm_ratio,
        ],
        dim=0,
    ).float().to(device)
    # ------------------------------------------------------------------


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.py`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
