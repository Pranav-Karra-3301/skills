# Mechanistic Interpretability Vocabulary

This glossary defines key terms used in mechanistic interpretability research. Understanding this vocabulary is essential for reading papers, communicating with researchers, and implementing experiments.

## Core Concepts

| Term | Definition |
|------|------------|
| **Mechanistic Interpretability** | The study of reverse-engineering neural networks to understand the algorithms they learn, not just their inputs/outputs |
| **Residual Stream** | The main "highway" through a transformer; each layer reads from and writes to this stream additively |
| **Feature** | A direction in activation space that represents a human-interpretable concept |
| **Circuit** | A subgraph of the model that implements a specific behavior or algorithm |
| **Superposition** | When a model represents more features than it has dimensions by using non-orthogonal directions |
| **Polysemanticity** | When a single neuron activates for multiple unrelated concepts |
| **Monosemanticity** | When a neuron activates for a single, interpretable concept |
| **Privileged Basis** | Directions in activation space that are special (e.g., individual neurons in MLPs due to nonlinearities) |
| **Feature Splitting** | When a coarse feature splits into finer-grained features as SAE width increases |

## Architecture Terms

| Term | Definition |
|------|------------|
| **Attention Head** | A single attention mechanism within a multi-head attention layer |
| **QK Circuit** | The Query-Key interaction that determines which positions attend to which |
| **OV Circuit** | The Output-Value interaction that determines what information gets moved |
| **MLP (Multi-Layer Perceptron)** | The feedforward network in each transformer layer; often stores factual knowledge |
| **Layer Norm** | Normalization applied before/after sublayers; affects how information flows |
| **Embedding** | Initial projection from tokens to vectors (W_E) |
| **Unembedding** | Final projection from vectors to logits (W_U) |
| **Positional Encoding** | How the model knows token positions; can be absolute or relative |

## Technique Terms

| Term | Definition |
|------|------------|
| **Hook** | An interception point to read or modify activations during forward pass |
| **Activation Patching** | Replacing activations from one run with those from another to measure causal effects |
| **Ablation** | Removing or zeroing out a component to measure its contribution |
| **Mean Ablation** | Replacing activations with their mean over a dataset |
| **Resample Ablation** | Replacing activations with those from a different input |
| **Logit Lens** | Looking at intermediate layer outputs through the unembedding matrix |
| **Tuned Lens** | Logit lens with learned affine transformations per layer |
| **Direct Logit Attribution (DLA)** | Decomposing final logits into contributions from each component |
| **Path Patching** | Patching along specific paths to isolate circuit components |
| **Probing** | Training a small classifier on intermediate activations to detect features |

## SAE Terms

| Term | Definition |
|------|------------|
| **Sparse Autoencoder (SAE)** | An autoencoder trained with sparsity constraints to find interpretable features |
| **L0** | The average number of active features per input (sparsity measure) |
| **Reconstruction Loss** | How well the SAE reconstructs the original activations |
| **Expansion Factor** | Ratio of SAE hidden dimension to input dimension (e.g., 4x, 8x, 32x) |
| **Dead Neuron/Feature** | A feature that never activates; needs resampling |
| **Feature Activation** | How strongly a feature fires for a given input |
| **Max Activating Examples** | Inputs that cause highest activation of a feature |
| **Feature Steering** | Adding feature directions to activations to influence model behavior |
| **Feature Ablation** | Removing specific features to test their causal role |
| **Ghost Gradients** | Technique to prevent dead features during training |

## SAE Architecture Variants

| Variant | Key Difference |
|---------|----------------|
| **Standard/Vanilla** | Basic ReLU activation with L1 penalty |
| **TopK** | Only top-K features active per input (fixed sparsity) |
| **BatchTopK** | TopK applied per batch for better gradient flow |
| **JumpReLU** | Discontinuous activation function for sharper features |
| **Gated** | Separate gating mechanism for feature selection |

## Evaluation Terms

| Term | Definition |
|------|------------|
| **Explained Variance** | How much of activation variance the SAE captures |
| **Loss Recovered** | Model performance when using SAE reconstructions |
| **Faithfulness** | Whether interpretations accurately describe model behavior |
| **Completeness** | Whether we've found all components involved in a behavior |
| **Minimality** | Whether the identified circuit is actually necessary |
| **Specificity** | Whether a feature activates only for the claimed concept |
| **Sensitivity** | Whether a feature activates for all instances of the concept |

## Model Internals

| Term | Definition |
|------|------------|
| **Induction Head** | An attention head pattern that copies tokens from earlier in context |
| **Copying Head** | Attention head that moves information from one position to another |
| **Name Mover Head** | Head that moves subject names to relevant positions |
| **Negative Head** | Head that decreases logit of certain tokens |
| **Backup Head** | Head that activates when primary head is ablated |
| **Virtual Weights** | Effective weights after combining multiple components |
| **Information Flow** | How information moves through the network via residual connections |

## Experimental Concepts

| Term | Definition |
|------|------------|
| **Activation Cache** | Stored activations from a forward pass for later analysis |
| **Clean Run** | Forward pass on the original input |
| **Corrupted Run** | Forward pass on a modified/counterfactual input |
| **Counterfactual** | A modified input used to measure causal effects |
| **Patching Metric** | What we measure to assess the effect of patching |
| **Hook Point** | Named location in model where we can intervene |
| **Batch Position** | Index in the batch dimension (0 for single samples) |
| **Sequence Position** | Index in the sequence/token dimension |

## Common Patterns

| Pattern | What It Indicates |
|---------|-------------------|
| **High L0, Low Recon Loss** | Features too active, increase sparsity penalty |
| **Many Dead Features** | Training unstable, reduce learning rate or use resampling |
| **Feature Activates on Unrelated Concepts** | Polysemantic feature, need more SAE width |
| **Patching Has No Effect** | Component not causally involved, or redundancy exists |
| **Different Seeds Give Different Results** | Finding may not be robust, test multiple seeds |
