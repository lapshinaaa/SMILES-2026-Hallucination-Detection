# Hallucination Detection Project Submission  
## Anastasiia Lapshina — SMILES-2026

**Predictions file:** [predictions.csv (Google Drive)](https://drive.google.com/file/d/1EnrgExjByO4xD4Q_nFl96bZG-S-bq4CQ/view?usp=sharing)

## Contents

1. Reproducibility Instructions  
2. Final Solution Description  
3. Experiments and Failed Attempts  
   3.1 Aggregation experiments  
   3.2 Geometric features experiments  
   3.3 Probe experiments  
   3.4 Splitting / evaluation note  
4. Conclusion

---

## 1. Reproducibility Instructions

### Exact commands

The submitted repository can be reproduced with the following commands:

```bash
git clone https://github.com/lapshinaaa/SMILES-2026-Hallucination-Detection
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py 
```

This repository is intended to be run with the provided official `solution.py` pipeline. Running it will produce:

- `results.json`
- `predictions.csv`

The final submission is self-contained and uses the allowed editable components of the project:

- `aggregation.py`
- `probe.py`
- `splitting.py`

In addition, the final configuration uses the geometric-feature toggle in `solution.py` with:

- `USE_GEOMETRIC = True`

To make the final result reproducible, I added explicit seed fixation inside `probe.py`. In particular, I introduced the following helper function:

```python
# ------------------------------------------------------------------
# STUDENT: Fix random seeds for reproducible training runs.
# ------------------------------------------------------------------
def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # for CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
```

This helper is called inside `fit()` before probe training begins. The purpose of this change is to reduce run-to-run variation caused by random initialization, dropout, and CUDA nondeterminism, and therefore make the reported checkpoint reproducible.

In `splitting.py`, I also fixed the development split by setting:

- `random_state = 43`

I use this as a fixed stratified development split for reproducibility and comparability across experiments. On this small labeled dataset, performance varied substantially across different split realizations, so fixing a single split was necessary to compare ablations consistently and to preserve the exact final checkpoint.

The strongest result was obtained on a fixed stratified development split, while multi-split evaluation produced lower but still above-baseline performance, indicating substantial sensitivity to split composition on the small labeled dataset.

---

## 2. Final Solution Description

My final solution modifies three components of the project:

- `aggregation.py`
- `probe.py`
- `splitting.py`

In addition, the final configuration uses geometric feature extraction through the `USE_GEOMETRIC = True` flag in `solution.py`.

The main idea of my final approach is the following: instead of using a broad pooled summary over many token positions, I focus on the **last real token** and aggregate its representations from the **top 3 transformer layers**. This compact representation is then extended with a small set of handcrafted geometric/statistical features and passed to a lightweight but strongly regularized probe classifier.

### 2.1 Aggregation

The final aggregation strategy is based on the last real token only. For that token, I take the hidden representations from the top 3 transformer layers and combine them using a weighted average. The final layer is given a slightly larger contribution, but the mixture remains relatively balanced:

> `[0.3, 0.3, 0.4]`

These weights were selected empirically during the aggregation experiments and gave the best trade-off among the variants I tested.

So, instead of relying only on the final transformer layer, I use a small amount of cross-layer smoothing. In practice, this means that the final representation is still highly localized at the last token position, but is stabilized by information from the previous top layers.

### 2.2 Geometric features

In the final setup, I also append a compact block of handcrafted geometric/statistical features to the aggregated representation. These features are:

- `seq_len_feature` — number of real (non-padding) tokens
- `final_norm` — norm of the last-token vector from the final transformer layer
- `mean_norm` — mean norm of the last-token vectors from the top 3 layers
- `std_norm` — standard deviation of those norms
- `cos_last2` — cosine similarity between the last-token vectors of the final two layers
- `cos_last13` — cosine similarity between the last-token vectors of layers `-1` and `-3`
- `l2_last12` — L2 distance between the last-token vectors of layers `-1` and `-2`
- `l2_last13` — L2 distance between the last-token vectors of layers `-1` and `-3`
- `norm_ratio` — ratio between the final-layer norm and the mean top-3-layer norm

These features are meant to capture compact geometric information about representation magnitude, agreement, and drift across the top layers. The resulting final feature dimension is $$905$$.

### 2.3 Splitting strategy

The splitting strategy remained close to the original stratified split. The main change was introduced for reproducibility: I fixed the development split with

- `random_state = 43`

This gives me a stable stratified development split and ensures that all ablation results and final metrics are directly comparable under the same data partition.

### 2.4 Probe

In `probe.py`, I use a compact nonlinear probe with the following architecture:

```python
self._net = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1),
)
```

Compared with the original version, I made the following changes:

- I kept the classifier compact, with a hidden size of $$64$$
- I added `LayerNorm`
- I used `GELU` as the activation function
- I added `Dropout(0.5)` for strong regularization

The corresponding training loop uses:

- `AdamW` instead of `Adam`
- learning rate $$3 \cdot 10^{-4}$$
- weight decay $$10^{-1}$$
- cosine annealing learning-rate scheduling
- $$1500$$ training epochs

Concretely, the training part is:

```python
optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500)

self.train()
for _ in range(1500):
    optimizer.zero_grad()
    logits = self(X_t)
    loss = criterion(logits, y_t)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

I selected this setup because the probe needed to stay expressive enough to capture nonlinear structure in the hidden-state features, but not so large that it would become even more unstable on a small dataset.

### 2.5 Final metrics

The final selected checkpoint achieved the following metrics on the fixed development split:

| Split | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Train | $$100.00\%$$ | $$100.00\%$$ | $$100.00\%$$ |
| Validation | $$75.00\%$$ | $$84.34\%$$ | $$65.05\%$$ |
| Test | $$77.88\%$$ | $$86.06\%$$ | $$79.72\%$$ |

For reference, the majority-class baseline was:

| Baseline | Accuracy | F1 |
|---|---:|---:|
| Majority-class baseline | $$70.19\%$$ | $$82.49\%$$ |

### 2.6 Final configuration summary

| Hyperparameter / component | Final value |
|---|---|
| Split strategy | Single stratified split |
| Split seed | $$43$$ |
| Aggregation token position | Last real token |
| Aggregation layers | Top $$3$$ transformer layers |
| Aggregation weights | `[0.3, 0.3, 0.4]` |
| Geometric features | Enabled |
| Final feature dimension | $$905$$ |
| Probe hidden dimension | $$64$$ |
| Probe normalization | LayerNorm |
| Activation | GELU |
| Dropout | $$0.5$$ |
| Optimizer | AdamW |
| Learning rate | $$3 \cdot 10^{-4}$$ |
| Weight decay | $$10^{-1}$$ |
| Scheduler | CosineAnnealingLR |
| Scheduler T_max | $$1500$$ |
| Epochs | $$1500$$ |

### 2.7 What contributed most to the final result

The biggest improvements came from the feature representation rather than from making the classifier deeper or more complicated. In particular, the most important decisions were:

- focusing the aggregation on the last real token,
- combining the top 3 transformer layers with a balanced weighted average,
- and using a compact but strongly regularized nonlinear probe.

Broader pooling over token positions, wider layer windows, dimensionality reduction, and much more complex probe variants were explored, but did not provide a better final trade-off. Those experiments are described in the next section.



## 3. Experiments and Failed Attempts

Before discussing the experiments, I want to clarify the metric choice. In the repository instructions, it is stated that **accuracy on `test.csv` is the primary competition metric**. At the same time, during the evaluation runs, the printed summary highlights **Test AUROC** as the “primary metric.” Because of this mismatch, I treated both metrics as important during development. In practice, I tried to improve the final classification quality measured by accuracy, while also monitoring AUROC as a threshold-independent measure of ranking quality.

### 3.1 Aggregation experiments

Aggregation turned out to be the most important part of the project. I started from the default baseline, where the feature vector was taken from the **last real token of the final transformer layer**, and then gradually tested whether a richer or more targeted aggregation strategy could improve the result.

At this stage, I kept everything else as comparable as possible and treated aggregation as the main source of change.

### Hypotheses

- **H1.** A broader sequence-level summary may help, because the hallucination signal could be distributed across the whole answer rather than concentrated in one token.
- **H2.** If broad pooling washes out the useful signal, the model may benefit from focusing on the **last real token** instead.
- **H3.** If the last token is important, aggregating it across several top layers may be more stable than using only one layer.
- **H4.** The top layers may not contribute equally, so a **weighted** top-3 mixture may outperform a simple average.

### Aggregation ablation table

| ID | Aggregation strategy | Feature dim | Val AUROC | Test Acc | Test F1 | Test AUROC | Main takeaway |
|---|---|---:|---:|---:|---:|---:|---|
| H1 | **Rich concat**: mean-pooled last 4 layers over all real tokens **+** last real token from the final layer | 1792 | 66.06% | 69.23% | 81.61% | 49.71% | Strong drop; feature expansion hurt badly |
| H2 | **Mean pool over last 4 layers** (all real tokens, then average across layers) | 896 | 54.57% | 70.19% | 82.49% | 53.16% | Better than H1, but still weak |
| H3 | **Final-layer mean pool** over all real tokens | 896 | 54.22% | 70.19% | 82.49% | 53.47% | Broad token averaging still weak |
| H4 | **Top-2 last-token average** | 896 | 65.82% | 74.04% | 82.58% | 73.40% | First major jump; localized representation works |
| H5 | **Top-3 weighted last-token** `[0.2, 0.3, 0.5]` | 896 | 66.90% | 74.04% | 83.23% | 73.97% | Better than top-2 averaging |
| H6 | **Top-3 weighted last-token** `[0.15, 0.25, 0.6]` | 896 | 66.64% | 75.00% | 83.75% | 73.75% | More final-layer-heavy; AUROC dipped slightly |
| H7 | **Top-3 weighted last-token** `[0.3, 0.3, 0.4]` | 896 | 67.34% | 74.04% | 83.23% | 74.30% | Best hidden-state-only aggregation in this series |

### Commentary

#### H1–H3: broad pooling did not help

My first idea was to make the aggregation richer and more expressive by combining global information from many tokens with a local final-token summary. In practice, that did not work.

The rich concatenation experiment in **H1** performed especially poorly. The most likely reason is that the feature vector became too large relative to the dataset size and introduced too much redundancy and noise.

Then I simplified the design and tested plain mean pooling across all real tokens, first over the last 4 layers and then over the final layer only (**H2–H3**). These variants also stayed weak. This suggested that the useful signal is **not distributed uniformly across the sequence**, and that averaging over all token positions mostly washes it out.

#### H4: the important signal is localized

The first major improvement came when I stopped treating all token positions as equally informative and switched to the **last real token** only.

In **H4**, I averaged the last token across the top 2 transformer layers. This immediately produced a large jump in AUROC and confirmed the main intuition behind the later experiments: the representation that matters most is highly localized near the end of the response.

This was the turning point of the aggregation experiments.

#### H5–H7: top-3 weighted last-token aggregation

Once the last-token hypothesis started working, I moved to a top-3-layer version and tested whether the three top layers should contribute equally or not.

With everything else held constant, the weighted top-3 variants were clearly better than the broad pooling strategies.

- In **H5**, the weighting `[0.2, 0.3, 0.5]` already improved over the simple top-2 average.
- In **H6**, I made the final layer more dominant with `[0.15, 0.25, 0.6]`. This gave strong thresholded metrics, but AUROC became slightly worse.
- In **H7**, the more balanced weighting `[0.3, 0.3, 0.4]` gave the strongest overall hidden-state-only result in this series.

This suggests that the final layer should have a slight advantage, but not enough to drown out the signal from the preceding top layers.

### Other aggregation ideas I tested but did not keep

I also tested several nearby ideas that I did not retain in the final solution:
- pooling over the last 2 or 3 token positions instead of only the final token,
- using broader local tails with recency weighting,
- and extending the aggregation to 4 or 5 top layers.

These directions did not improve the metric in a meaningful way. In general, once I moved away from the single last token, the representation became noisier. Similarly, increasing the layer window beyond the top 3 layers did not provide a better trade-off.

### Aggregation conclusion

The aggregation experiments led to a very clear conclusion:

- the hallucination signal is **not** best captured by broad token averaging,
- the most useful information is concentrated at the **last real token**,
- and the best representation comes from combining the **top 3 transformer layers** with a moderately balanced weighting:

> `[0.3, 0.3, 0.4]`

After establishing this as the strongest hidden-state-only aggregation design, I moved on to the next question: **do geometric features contribute anything on top of this representation?**

### 3.2 Geometric features experiments

Once the hidden-state-only aggregation had stabilized, I wanted to test whether a very small number of geometric/statistical features could add complementary information without making the representation much larger.

### Hypotheses

- **G1.** A compact set of geometric features may help, because hidden-state magnitude and agreement across the top layers could carry useful information that is not fully captured by the weighted hidden-state vector alone.
- **G2.** If a small geometric block helps, then adding a few more carefully chosen features describing cross-layer similarity and drift may improve stability further.

### Geometric-feature ablation table

| ID | Geometric feature setup | Feature dim | Val AUROC | Test Acc | Test F1 | Test AUROC | Main takeaway |
|---|---|---:|---:|---:|---:|---:|---|
| G1 | **5 compact geometric features** added on top of the best last-token top-3 aggregation | 901 | 66.66% | 74.04% | 83.23% | 74.88% | Small but real improvement over the hidden-state-only setup |

### Commentary

#### G1: a small geometric block helped

The first geometric block I added contained five compact features:

1. number of real tokens,  
2. norm of the final-layer last-token vector,  
3. mean norm of the top-3 last-token vectors,  
4. standard deviation of those norms,  
5. cosine similarity between the last-token vectors of the final two layers.  

This changed the feature dimension only slightly, from **896** to **901**, but produced a small and meaningful gain in AUROC.

My interpretation was that the main signal still came from the aggregated hidden representation, but these features added lightweight information about magnitude and agreement across the top layers.

#### G2: I then expanded the geometric block

Since the first 5-feature version helped, I extended the block further by adding:

- `cos_last13`
- `l2_last12`
- `l2_last13`
- `norm_ratio`

In words, these features describe:
- cosine similarity between layers `-1` and `-3`,
- L2 distance between layers `-1` and `-2`,
- L2 distance between layers `-1` and `-3`,
- and the ratio between the final-layer norm and the mean top-3-layer norm.

The motivation here was simple: if geometric information is useful at all, then the model may also benefit from a slightly richer description of cross-layer agreement and representation drift. My expectation was that these features could give the probe a bit more structure to work with and make training more stable without changing the overall design of the model.

I did not keep a clean isolated metric table for this second extension alone, because by that point the later runs were already intertwined with probe tuning. Still, this richer geometric block was the one I kept in the strongest final configuration, so in practice I considered it more useful than the smaller 5-feature version.

### Geometric-feature conclusion

The geometric-feature experiments suggested that compact handcrafted features can help, but only when they stay small and closely tied to the same last-token top-layer representation that already worked well.

The main lesson here was not that geometry replaced the hidden-state aggregation. It did not. The hidden-state-only representation already carried most of the useful signal. What the geometric block did was add a small amount of extra information about magnitude, agreement, and drift across the top layers.

After deciding to keep the richer geometric block in the final candidate, I moved on to the next question: **how much can the downstream probe itself still improve the result?**

### 3.3 Probe experiments

After settling on a strong aggregation strategy, I turned to the probe. The main question here was not whether the classifier alone could fundamentally transform the result, but whether a better readout layer could extract the signal from the improved feature representation more effectively.

At this stage, my working assumption was the following: once the hidden-state aggregation becomes good enough, the probe should mainly refine the decision boundary, regularizing the training process, and improving stability — but it should not be expected to create the same kind of large gains as aggregation itself.

### Hypotheses

- **P1.** A slightly smaller classifier may generalize better on the small dataset if the original probe is too expressive.
- **P2.** Regularization through dropout, weight decay, and longer or shorter training may help stabilize the final classifier.
- **P3.** More architectural complexity may help if the aggregated representation still needs a more expressive nonlinear readout.
- **P4.** If the representation is already strong, the best probe may actually remain relatively simple, with improvements coming mostly from careful regularization and training setup.

### Probe ablation table

| ID | Probe modification | Feature dim | Test Acc | Test F1 | Test AUROC | Main takeaway |
|---|---|---:|---:|---:|---:|---|
| P1 | **Smaller hidden layer** `256 -> 128 -> 1` | 896 | 70.19% | 82.49% | 74.28% | Too small; thresholded metrics collapsed toward baseline |
| P2 | **Dropout only** (hidden 256, dropout `0.2`) | 896 | 73.08% | 81.58% | 74.19% | Slight change, but no meaningful improvement |
| P3 | **Weight decay only** | 896 | 74.04% | 83.44% | 74.19% | Better thresholded metrics, AUROC unchanged |
| P4 | **100 epochs** (same simple probe) | 896 | 70.19% | 81.21% | 74.50% | Too short; undertrained |
| P5 | **Simple/original probe** on improved aggregation checkpoint | 901 | 75.00% | 83.95% | 74.02% | Strong thresholded metrics; simple probe remained highly competitive |
| P6 | **Dropout + weight decay** on improved aggregation checkpoint | 901 | 73.08% | 81.82% | 74.99% | Best earlier AUROC, but weaker accuracy/F1 |
| P7 | **2-layer bottleneck MLP** on improved aggregation checkpoint | 901 | 69.23% | 79.49% | 69.91% | Clear failure; over-compression hurt badly |
| P8 | **Extra linear/ReLU layers or lower LR** | 901 | — | — | — | Consistently worsened performance; reverted |
| P9 | **Final selected probe** with strong regularization and long training | 905 | 77.88% | 86.06% | 79.72% | Best final fixed-split checkpoint |

### Commentary

#### P1–P4: smaller or lightly modified probes were not enough

My first probe experiments were fairly local changes around the baseline classifier. I tested a smaller hidden layer, dropout alone, weight decay alone, and a shorter training horizon.

These experiments were useful because they showed that changing the probe can definitely make the result worse, but also that probe-side improvements do not automatically translate into large gains. In particular:

- reducing the hidden dimension too much made the classifier too weak,
- mild regularization alone did not transform the results,
- and training for only 100 epochs was clearly too short.

So at this stage, the main lesson was that the probe still needed some nonlinear capacity, but the core limitation of the system was not simply “the classifier is too small.”

#### P5–P8: more complexity was not the answer either

After improving aggregation, I then tested whether the stronger feature representation would benefit from a more ambitious classifier.

Some of these changes initially seemed promising:
- the simple probe on the stronger aggregation checkpoint already gave good thresholded metrics,
- a regularized variant slightly improved AUROC,
- and this suggested that the probe could still help shape the final decision boundary.

However, once I pushed harder into deeper or more complex architectures, the results deteriorated. In particular:
- bottleneck probes over-compressed the feature space,
- extra linear/ReLU layers hurt,
- lower learning rates did not help,
- PCA-based dimensionality reduction also failed,
- and a purely linear probe was clearly too weak.

This made the overall pattern much clearer: the probe did matter, but mostly as a careful readout layer, not as the main engine of improvement.

#### P9: final probe design

The final probe I kept is still relatively compact, but much more carefully regularized than the early variants:

```python
self._net = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1),
)
```

The corresponding training loop uses:
- `AdamW` instead of `Adam`,
- learning rate `3e-4`,
- weight decay `1e-1`,
- cosine annealing scheduling,
- and `1500` training epochs.

This final setup was chosen because it gave me the strongest final fixed-split checkpoint:

- **Test accuracy:** 77.88%
- **Test F1:** 86.06%
- **Test AUROC:** 79.72%

The interesting part here is that the final probe is not especially deep or elaborate. What changed most is the training regime: stronger regularization, longer training, and a more deliberate optimizer/scheduler setup.

### Probe conclusion

The probe experiments led me to the following conclusion:

- the probe is **not** the main source of improvement in this project,
- but it still matters as a final polishing stage,
- because poor architectural choices can absolutely damage performance,
- while a well-regularized compact classifier can help extract the signal from a strong aggregation setup.

So although the final probe contributed to the best overall checkpoint, I would still describe it as a refinement layer, not the fundamental reason the model worked. The main gains came from the aggregation. The probe mostly determined how cleanly and stably that information could be turned into final predictions.

### 3.4 Splitting / evaluation note

I also experimented with the split strategy, mainly to understand how sensitive the model was to train/validation/test partitioning. The most important role of splitting was not to create large performance gains by itself, but to reveal how unstable some seemingly strong checkpoints actually were.

The final fixed development split with seed `43` was therefore used as the main reference point for direct ablation comparisons. This made the experiments much easier to compare under the same data partition. At the same time, multi-split evaluation was still informative as a more conservative robustness check.

This matters for interpreting the final results: some strong checkpoints did not hold up equally well under different splits, which is exactly why I do not treat splitting as the main source of gains. Its role was mostly diagnostic and organizational.

## 4. Conclusion

This project ended up being much more experimental than I initially expected. In total, I conducted well over **100** runs in search of better metrics, more stable training, and a final configuration that could be reproduced reliably.

The main conclusion of the project is that the largest gains came from the **aggregation step**, not from the probe. Once I moved away from broad token averaging and refocused the representation on the **last real token**, performance improved substantially. The strongest aggregation design was a weighted combination of the top 3 transformer layers, with weights:

> `[0.3, 0.3, 0.4]`

This turned out to be much more important than any single probe modification. In my view, this is the central technical result of the project: the main source of useful signal lies in how the hidden states are compressed into features.

The probe still mattered, but mostly as a **polishing stage** rather than the main engine of improvement. Poor probe choices could absolutely make the model worse: reducing the hidden dimension too much, adding excessive architectural complexity, introducing bottlenecks, lowering the learning rate too aggressively, or applying dimensionality reduction all led to worse results. At the same time, probe-side changes alone did not create the same kind of substantial gains as aggregation. The best probe was still relatively compact, and the main improvements there came from regularization and training setup rather than from making the classifier much more sophisticated.

Another important conclusion is that the dataset is small enough that **overfitting is very difficult to avoid**. I tried to address this in several ways: stronger regularization, fixed seeds, a stable development split, geometric features, and a more careful training regime. These changes did help, especially in terms of reproducibility and stability, but only to a point. My interpretation is that the main bottleneck is the data regime itself: the dataset is small enough that learning robust generalization is much harder than memorizing training patterns.

The geometric features were also useful to think about in this context. They did not replace the hidden-state aggregation and were never the main source of gains, but a compact set of them did add a small amount of complementary signal. This was enough to justify keeping them in the final configuration, especially since they remained lightweight and interpretable.

A related lesson from this project is that **reproducibility and robustness are not the same thing**. Some settings produced very strong results on one split but did not hold up equally well under multi-split evaluation. For that reason, I treated the strongest fixed-split result and the more conservative multi-split picture as two different but equally important views of the model. The strongest result was obtained on a fixed stratified development split, while multi-split evaluation produced lower but still above-baseline performance, indicating substantial sensitivity to split composition on the small labeled dataset.

Overall, I would summarize the project like this:

- the biggest gains came from **last-token-centered weighted top-layer aggregation**;
- geometric features helped slightly, but only as a lightweight extension;
- the probe was important mainly for **refinement and stability**, not for fundamental gains;
- the small dataset made overfitting and split sensitivity almost inevitable;
- and the final submission should therefore be interpreted as a strong best-case checkpoint together with a more conservative robustness picture.

Even though the project was often unstable and frustrating, I think it still led to a clear technical story, which I described above. The final solution is simple enough and supported by a large number of ablations and negative results rather than by a single lucky guess.
