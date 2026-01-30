# Common Pitfalls in Mechanistic Interpretability

This reference covers common mistakes and how to avoid them.

## Epistemic Pitfalls

### 1. Conflating Hypotheses with Conclusions

**The mistake**: Presenting an interpretation as established fact before proper validation.

**Example**: "This attention head implements copying" after only looking at a few examples.

**Fix**:
- Phrase findings as hypotheses until validated
- Use language like "appears to", "may be", "we hypothesize"
- List evidence strength: "based on N examples" or "validated causally"

### 2. Beautiful Bullshit Hypotheses

**The mistake**: Elegant, compelling explanations that don't survive rigorous testing.

**Signs**:
- Explanation sounds too clean
- Only tested on cherry-picked examples
- No adversarial testing
- Ignores edge cases that don't fit

**Fix**:
- Actively seek disconfirming evidence
- Test on held-out examples before finalizing interpretation
- Ask "what would falsify this hypothesis?"
- Have someone else try to break your interpretation

### 3. Streetlight Interpretability

**The mistake**: Only studying things that are easy to interpret (small models, simple tasks) and assuming they generalize.

**Example**: Drawing conclusions about GPT-4's mechanisms from GPT-2-small experiments.

**Fix**:
- Acknowledge limitations of model/task choice
- Don't overclaim generalization
- Scale up experiments when possible
- Be explicit about what transfers vs what might not

### 4. Cherry-Picking Examples

**The mistake**: Showing only examples that support your interpretation.

**Fix**:
- Report statistics over many examples
- Show failure cases
- Use random sampling, not manual selection
- Include confidence intervals

### 5. Ignoring Alternative Explanations

**The mistake**: Not considering simpler or different explanations for observations.

**Fix**:
- List alternative hypotheses
- Design experiments that distinguish between them
- Consider null hypotheses (e.g., "this component does nothing important")

### 6. Overfitting to Specific Inputs

**The mistake**: Developing interpretations that only work for the exact prompts used in experiments.

**Fix**:
- Test on diverse inputs
- Use paraphrased versions of the same semantic content
- Check if interpretation holds for different surface forms

## Technical Pitfalls

### 7. Memory Leaks with Hooks

**The mistake**: Accumulating activations in hooks without clearing them.

```python
# BAD: Memory leak
all_activations = []
def hook_fn(activation, hook):
    all_activations.append(activation)  # Never freed!
    return activation

# GOOD: Use torch.no_grad() and process immediately
def hook_fn(activation, hook):
    with torch.no_grad():
        stats = compute_stats(activation.detach())
    return activation
```

**Fix**:
- Always call `model.reset_hooks()` after experiments
- Use context managers: `with model.hooks(...):`
- Detach tensors if storing: `.detach().cpu()`
- Process activations immediately rather than accumulating

### 8. Cache Invalidation Issues

**The mistake**: Reusing stale cached activations after modifying the model or input.

**Fix**:
- Cache with input hash as key
- Clear cache when model/tokenizer changes
- Be explicit about what's cached vs recomputed

### 9. Precision Issues

**The mistake**: Getting different results due to float16/bfloat16 vs float32.

**Signs**:
- Results change between GPU and CPU
- Results change with batch size
- Numerical instability in attention patterns

**Fix**:
- Use consistent dtype throughout
- Be aware that bfloat16 has less precision
- Compare results across dtypes if uncertain
- Use `torch.float32` for final analysis

### 10. Wrong Baseline for Patching

**The mistake**: Comparing patched model to wrong baseline.

**Example**: Measuring patching effect against random baseline instead of corrupted baseline.

**Fix**:
- Be explicit about baselines: clean, corrupted, ablated
- Report what you're comparing to
- Use consistent metrics across experiments

### 11. Position/Batch Dimension Errors

**The mistake**: Operating on wrong dimension of activation tensors.

```python
# Shape: [batch, seq, d_model]
# BAD: Wrong dimension
activations.mean(dim=0)  # Averages over batch AND seq positions

# GOOD: Be explicit
activations.mean(dim=1)  # Average over positions
activations.mean(dim=0)  # Average over batch (if that's what you want)
```

**Fix**:
- Always check tensor shapes
- Name dimensions explicitly in comments
- Use einops for clarity: `rearrange(x, "batch seq dim -> ...")`

### 12. Not Handling BOS Token

**The mistake**: Including or excluding BOS token inconsistently.

**Fix**:
- Know if your model uses BOS
- Be consistent about whether position 0 is BOS
- Document your convention

### 13. Attention Mask Issues

**The mistake**: Not accounting for causal masking or padding masks.

**Fix**:
- Check if attention scores include -inf for masked positions
- Don't interpret attention to padding tokens
- Remember causal models can't attend to future positions

## Validation Failures

### 14. Not Testing Causal Role

**The mistake**: Claiming a component "does X" without showing that ablating it changes X.

**Fix**:
- Always pair observations with interventions
- If head H "copies tokens", show that ablating H reduces copying
- Correlation is not causation

### 15. Not Testing on Held-Out Data

**The mistake**: Developing and testing interpretation on same examples.

**Fix**:
- Split data: develop interpretation on training set, validate on test set
- Use random samples for final metrics
- Report train/test differences

### 16. Single Random Seed

**The mistake**: Results that don't replicate with different random seeds.

**Fix**:
- Run experiments with multiple seeds
- Report variance across seeds
- Flag results that are seed-sensitive

### 17. Not Testing Simpler Explanations

**The mistake**: Proposing complex circuits when simpler mechanisms suffice.

**Fix**:
- Test ablations of individual components
- Ask: "Is this full circuit necessary, or is one component sufficient?"
- Apply Occam's razor

## Mindset Issues

### 18. Overthinking vs Just Doing Stuff

**The mistake**: Spending too long planning perfect experiments instead of running quick tests.

**Fix**:
- Run quick sanity checks first
- Iterate: hypothesis → quick test → refine → rigorous test
- "Just try it and see" is often valuable
- Perfect is the enemy of good

### 19. Not Recording Negative Results

**The mistake**: Only tracking experiments that "worked."

**Fix**:
- Log all experiments, including failures
- Negative results inform what doesn't work
- Share failed approaches to help others

### 20. Tooling vs Research

**The mistake**: Spending all time building infrastructure instead of running experiments.

**Fix**:
- Use existing tools (TransformerLens, SAELens) when possible
- Only build custom tools when necessary
- Balance: some infrastructure is needed, but it's not the goal

## Validation Checklist

Before claiming you've found something:

- [ ] **Causal evidence**: Does ablating/patching change the behavior?
- [ ] **Held-out testing**: Does it replicate on new examples?
- [ ] **Multiple seeds**: Is the result robust to randomness?
- [ ] **Alternative explanations**: Have you ruled out simpler stories?
- [ ] **Edge cases**: What inputs break the interpretation?
- [ ] **Quantitative metrics**: Can you put a number on it?
- [ ] **Adversarial examples**: Can you find counterexamples?
- [ ] **Effect size**: Is the effect practically meaningful?

## Red Flags in Interpretability Claims

Watch for these in your own work and others':

| Red Flag | Better Practice |
|----------|-----------------|
| "Head X does Y" (no evidence) | "Head X appears to do Y based on..." |
| Only cherry-picked examples | Random sample + statistics |
| No ablation/patching | Causal intervention |
| Works only on one prompt | Tested on diverse inputs |
| No comparison to baseline | Explicit baseline + effect size |
| Results on toy model only | Acknowledged limitations |
| Single experiment | Replicated multiple times |
| No code/data shared | Reproducible artifacts |
