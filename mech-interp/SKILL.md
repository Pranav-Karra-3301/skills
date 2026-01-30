---
name: mech-interp
description: |
  Mechanistic interpretability skill for understanding neural network internals. Use when:
  (1) Working with TransformerLens, nnsight, or SAELens
  (2) Analyzing attention patterns, circuits, or features
  (3) Training or analyzing sparse autoencoders (SAEs)
  (4) Activation patching, logit lens, or direct logit attribution
  (5) Understanding model internals ("what is this head doing?")
  (6) Setting up a mech interp research project
  Triggers: "mech interp", "mechanistic interpretability", "TransformerLens", "SAE", "sparse autoencoder", "activation patching", "logit lens", "circuits", "features", "attention heads", "residual stream", "nnsight", "nnterp", "superposition", "polysemantic", "monosemantic", "ablation", "probing", "induction head"
---

# Mechanistic Interpretability

## Overview

Mechanistic interpretability (mech interp) is the science of reverse-engineering neural networks to understand the algorithms they learn. The core question: **"What computation is this model performing, and how?"**

**Key concepts:**
- **Residual stream**: The main highway through the model; each layer reads from and writes to it
- **Features**: Directions in activation space representing interpretable concepts
- **Circuits**: Subgraphs implementing specific behaviors
- **Superposition**: Models represent more features than dimensions using non-orthogonal directions

**Why it matters:**
- Understand model capabilities and limitations
- Debug unexpected behaviors
- Verify safety properties
- Build interpretable AI systems

## Core Workflow

### Phase 1: Environment Setup

1. **Detect compute resources**
   ```python
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Device: {device}")
   if device == "cuda":
       print(f"GPU: {torch.cuda.get_device_name(0)}")
       print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
   ```

2. **Install libraries based on model size**

   | Model Size | Primary Tool | Install |
   |------------|--------------|---------|
   | ≤2B params | TransformerLens | `pip install transformer-lens` |
   | 2B-13B params | nnsight | `pip install nnsight` |
   | SAE training | SAELens | `pip install sae-lens[train]` |
   | Both ecosystems | nnterp | `pip install nnterp` |

3. **Load model**
   ```python
   from transformer_lens import HookedTransformer

   model = HookedTransformer.from_pretrained(
       "gpt2-small",
       device=device,
   )
   ```

See [references/tools.md](references/tools.md) for detailed tool setup.

### Phase 2: Experiment Design

Before writing code, clarify:

1. **Research question**: What specifically are you trying to understand?
   - "What does attention head L5H3 do?"
   - "How does the model represent 'is_capital_of' relationships?"
   - "Which components contribute to this prediction?"

2. **Hypothesis**: What do you expect to find?
   - Phrase as testable predictions
   - Include what would falsify the hypothesis

3. **Technique selection** (see table below)

4. **Validation plan**: How will you verify findings?
   - Causal interventions
   - Held-out examples
   - Alternative explanations to rule out

### Phase 3: Implementation

Select technique based on your goal:

| Goal | Technique | Tool/Method | Reference |
|------|-----------|-------------|-----------|
| What tokens does model predict at each layer? | Logit Lens | `resid @ W_U` | [techniques.md](references/techniques.md) |
| Which component affects this output? | Activation Patching | `run_with_hooks` | [techniques.md](references/techniques.md) |
| How much does each head contribute to logit? | Direct Logit Attribution | Decompose residual | [techniques.md](references/techniques.md) |
| What information does this head move? | OV Circuit Analysis | `W_V @ W_O` | [techniques.md](references/techniques.md) |
| What attends to what? | QK Circuit Analysis | `W_Q @ W_K.T` | [techniques.md](references/techniques.md) |
| Is information X represented here? | Probing | Train classifier | [techniques.md](references/techniques.md) |
| Find interpretable features | SAE | Train/load SAE | [sae-guide.md](references/sae-guide.md) |
| Which feature represents concept Y? | Feature Search | Max activating examples | [sae-guide.md](references/sae-guide.md) |

### Phase 4: Analysis

1. **Run experiments**
   - Cache activations: `logits, cache = model.run_with_cache(tokens)`
   - Always use `torch.no_grad()` for inference
   - Save intermediate results

2. **Visualize results**
   - Attention heatmaps
   - Patching effect matrices
   - Feature activation distributions

   See [references/visualization.md](references/visualization.md)

3. **Iterate**
   - Refine hypothesis based on findings
   - Test edge cases
   - Look for counterexamples

### Phase 5: Validation

Before claiming findings, verify:

- [ ] **Causal evidence**: Ablating/patching changes behavior as predicted
- [ ] **Held-out data**: Results replicate on unseen examples
- [ ] **Multiple seeds**: Not an artifact of specific randomness
- [ ] **Alternative explanations**: Ruled out simpler stories
- [ ] **Effect size**: Practically meaningful, not just statistically significant

See [references/pitfalls.md](references/pitfalls.md) for common mistakes.

## Technique Quick Reference

### Logit Lens

Project intermediate representations through unembedding to see evolving predictions.

```python
for layer in range(model.cfg.n_layers):
    resid = cache["resid_post", layer]
    resid_normed = model.ln_final(resid)
    logits = resid_normed @ model.W_U
    top_token = logits[0, -1].argmax()
    print(f"Layer {layer}: {model.to_str_tokens(top_token)}")
```

### Activation Patching

Measure causal effect by swapping activations between runs.

```python
def patch_hook(activation, hook):
    activation[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return activation

patched_logits = model.run_with_hooks(
    corrupted_tokens,
    fwd_hooks=[(hook_point, patch_hook)]
)
```

### Direct Logit Attribution

Decompose final logits into per-component contributions.

```python
target_dir = model.W_U[:, target_token_idx]
for layer in range(model.cfg.n_layers):
    attn_contribution = cache["attn_out", layer][0, -1] @ target_dir
    mlp_contribution = cache["mlp_out", layer][0, -1] @ target_dir
    print(f"L{layer} attn: {attn_contribution:.3f}, mlp: {mlp_contribution:.3f}")
```

### SAE Feature Analysis

Find interpretable features in activations.

```python
from sae_lens import SAE

sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
feature_acts = sae.encode(cache["resid_pre", 8])
top_features = feature_acts[0, -1].topk(10)
```

## Model Size Guidance

| Model | Library | Memory (FP16) | Notes |
|-------|---------|---------------|-------|
| GPT-2-small | TransformerLens | ~0.25GB | Best for learning |
| GPT-2-medium/large | TransformerLens | ~0.7-1.5GB | Good balance |
| GPT-2-xl | TransformerLens | ~3GB | Needs decent GPU |
| Pythia-70M to 410M | TransformerLens | ~0.15-0.8GB | Checkpoints available |
| Pythia-1B to 2.8B | TransformerLens | ~2-5.5GB | Pushes memory |
| Pythia-6.9B+ | nnsight | ~14GB+ | Use nnsight for efficiency |
| Llama-2-7B, Mistral-7B | nnsight | ~14GB | Needs 24GB+ GPU |
| Llama-2-13B+ | nnsight | ~26GB+ | Need A100/multi-GPU |

See [references/compute-awareness.md](references/compute-awareness.md) for memory estimation.

## When to Ask the User

**Ask before proceeding when:**

1. **Research question unclear**
   > "What specific behavior or component are you trying to understand?"

2. **Compute constraints unknown**
   > "What GPU do you have available? This model needs ~XGB VRAM."

3. **Multiple valid approaches**
   > "We could use activation patching (causal) or probing (correlational). Which do you prefer?"

4. **Unexpected results**
   > "The results don't match expectations. Should we investigate further or try a different approach?"

5. **Scaling decisions**
   > "Initial results look promising on GPT-2-small. Want to scale up to a larger model?"

## Common Tasks

### "Set up a mech interp project"

1. Create project structure (see [repo-maintenance.md](references/repo-maintenance.md))
2. Install dependencies based on target model
3. Set up CLAUDE.md with project-specific instructions
4. Configure experiment tracking (wandb or simple JSON logging)

### "What does this attention head do?"

1. Visualize attention patterns across diverse inputs
2. Analyze QK circuit (what attends to what)
3. Analyze OV circuit (what information moves)
4. Test with activation patching (is it necessary?)
5. Check for known patterns (induction, copying, etc.)

### "Find the circuit for behavior X"

1. Design clean/corrupted input pairs
2. Patch residual stream: layer × position heatmap
3. Narrow to specific layers
4. Patch individual heads
5. Validate with ablation
6. Analyze winning components

### "Train an SAE"

1. Choose layer and hook point
2. Estimate memory requirements
3. Set hyperparameters (expansion factor, L1 coefficient)
4. Run training with monitoring (L0, reconstruction loss, dead features)
5. Evaluate quality before analysis

See [references/sae-guide.md](references/sae-guide.md) for detailed guidance.

### "Interpret SAE features"

1. Load pretrained SAE or train your own
2. Find max activating examples for features of interest
3. Look for patterns in activating contexts
4. Test hypothesis with feature steering/ablation
5. Validate causal role

## Quality Checklist

Before concluding analysis:

- [ ] Research question clearly stated
- [ ] Appropriate technique selected
- [ ] Code runs without errors
- [ ] Results visualized
- [ ] Causal validation performed
- [ ] Edge cases tested
- [ ] Alternative explanations considered
- [ ] Results documented with reproducibility info

## Reference Files

| File | Contents |
|------|----------|
| [tools.md](references/tools.md) | TransformerLens, nnsight, SAELens setup |
| [techniques.md](references/techniques.md) | Patching, logit lens, circuits, probing |
| [sae-guide.md](references/sae-guide.md) | SAE training and analysis |
| [visualization.md](references/visualization.md) | Plotting patterns and dashboards |
| [pitfalls.md](references/pitfalls.md) | Common mistakes and validation |
| [repo-maintenance.md](references/repo-maintenance.md) | Project structure templates |
| [vocabulary.md](references/vocabulary.md) | Glossary of terms |
| [compute-awareness.md](references/compute-awareness.md) | GPU/memory guidance |
