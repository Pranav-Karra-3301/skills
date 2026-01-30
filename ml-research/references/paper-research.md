# Paper Research Guide

## Finding Papers

### Primary Sources

| Source | Best For | URL |
|--------|----------|-----|
| **arXiv** | Latest preprints, ML/AI | arxiv.org |
| **Semantic Scholar** | Citation graphs, related work | semanticscholar.org |
| **Google Scholar** | Broad search, citations | scholar.google.com |
| **Papers With Code** | Implementations, benchmarks | paperswithcode.com |
| **Connected Papers** | Visual exploration | connectedpapers.com |

### Search Strategies

#### Finding Foundational Papers
```
Start with:
1. Survey papers: "[topic] survey" or "[topic] review"
2. Check "Related Work" sections
3. Find highly-cited papers in citations

Example searches:
- "transformer architecture survey"
- "attention mechanism deep learning review"
```

#### Finding Recent Advances
```
1. Check arXiv daily/weekly
2. Search "[topic] 2024" on Google Scholar
3. Follow key authors on Twitter/arXiv
4. Check conference proceedings (NeurIPS, ICML, ICLR, ACL, CVPR)
```

#### Finding Implementations
```
1. Papers With Code - browse by task/dataset
2. GitHub search: "[paper name]" or "[method name]"
3. Check paper's supplementary materials
4. Look for "Official" implementations
```

### Useful Search Queries

```
# arXiv categories
cat:cs.LG - Machine Learning
cat:cs.CL - Computational Linguistics (NLP)
cat:cs.CV - Computer Vision
cat:stat.ML - Statistics/ML

# Example arXiv search
site:arxiv.org "transformer" "attention" 2024

# Google Scholar - recent, highly cited
[topic] after:2023
[method] "state-of-the-art"
```

## Reading Papers Efficiently

### The Three-Pass Method

**Pass 1: Survey (5-10 minutes)**
- Read title, abstract, introduction (first few paragraphs)
- Read section headings
- Look at figures and tables (especially results)
- Read conclusion
- Answer: What problem? What approach? What results?

**Pass 2: Understand (30-60 minutes)**
- Read the full paper, skip dense math
- Understand the method at a high level
- Note key equations (don't derive yet)
- Identify the core contribution
- Answer: How does it work? What's novel?

**Pass 3: Master (1-4 hours, if implementing)**
- Work through all mathematical details
- Mentally re-implement the method
- Identify assumptions and limitations
- Think about how to apply/extend

### Key Sections to Focus On

```
For Understanding:
├── Abstract (the 30-second summary)
├── Introduction (motivation, contributions)
├── Method/Approach (core contribution)
├── Experiments
│   ├── Datasets (what they test on)
│   ├── Baselines (what they compare to)
│   └── Results tables (quantitative claims)
└── Conclusion (takeaways, limitations)

For Implementation:
├── Method section (detailed)
├── Appendix (often has crucial details)
├── Hyperparameters (often in appendix or supplementary)
└── Code (if available)
```

### Questions to Answer While Reading

1. **Problem**: What problem does this solve? Why does it matter?
2. **Approach**: What's the key insight/method?
3. **Novelty**: What's new vs. prior work?
4. **Results**: How much better? On what benchmarks?
5. **Limitations**: What doesn't work? What's missing?
6. **Applicability**: Can I use this? For what?

## Implementing from Papers

### Before You Start

```
Checklist:
[ ] Read paper thoroughly (at least Pass 2)
[ ] Check for official code release
[ ] Search Papers With Code for implementations
[ ] Check GitHub for unofficial implementations
[ ] Look for blog posts explaining the paper
[ ] Note all hyperparameters mentioned
```

### Implementation Strategy

**Step 1: Reproduce a Baseline**
```python
# Start with the simplest version
# Don't add bells and whistles yet

# Example: If implementing new attention mechanism
# First: Get standard attention working
# Then: Add the new modification
```

**Step 2: Match Paper's Setup Exactly**
```python
# Use same:
# - Dataset (or as close as possible)
# - Preprocessing
# - Model architecture details
# - Optimizer and LR schedule
# - Batch size and training duration
# - Evaluation metrics
```

**Step 3: Verify Against Reported Numbers**
```python
# Your results should be within ~1-2% of paper
# If not, something is likely wrong

# Common causes of discrepancy:
# - Different preprocessing
# - Missing data augmentation
# - Wrong hyperparameters
# - Different random seeds
# - Bug in implementation
```

### Common Implementation Pitfalls

#### 1. Missing Normalization Details
```python
# Paper says "we normalize inputs"
# Could mean:
# - Instance normalization
# - Batch normalization
# - Layer normalization
# - Standard scaling
# - Min-max scaling
# - Per-image normalization

# Check: appendix, code, or related papers by same authors
```

#### 2. Unclear Data Augmentation
```python
# Paper says "standard augmentations"
# For ImageNet, this typically means:
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# But verify! Different papers use different "standards"
```

#### 3. Unreported Hyperparameters
```python
# Common missing details:
# - Weight initialization
# - Dropout rates
# - Learning rate warmup steps
# - Gradient clipping threshold
# - Label smoothing
# - Weight decay

# Solutions:
# 1. Check appendix
# 2. Check official code
# 3. Email authors
# 4. Try common defaults
```

#### 4. Different Evaluation Protocol
```python
# "Accuracy" could mean:
# - Top-1 accuracy
# - Top-5 accuracy
# - Balanced accuracy
# - Per-class accuracy averaged

# "F1" could mean:
# - Micro F1
# - Macro F1
# - Weighted F1
# - F1 for positive class only
```

### Debugging Implementation Issues

```python
# 1. Unit test each component
def test_attention_mechanism():
    # Test with known inputs and outputs
    pass

# 2. Compare intermediate outputs with reference
def debug_forward_pass(model, reference_model, inputs):
    for name, module in model.named_modules():
        # Hook to capture intermediate outputs
        pass

# 3. Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")

# 4. Overfit on tiny dataset first
tiny_dataset = dataset[:10]
# Should achieve ~100% accuracy quickly
```

## Citing Papers

### BibTeX Format

```bibtex
# Conference paper
@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
          Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
          Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

# Journal paper
@article{lecun2015deep,
  title={Deep learning},
  author={LeCun, Yann and Bengio, Yoshua and Hinton, Geoffrey},
  journal={Nature},
  volume={521},
  number={7553},
  pages={436--444},
  year={2015},
  publisher={Nature Publishing Group}
}

# arXiv preprint
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers
         for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and
          Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

### Getting Citations

```
1. Google Scholar: Click "Cite" under paper
2. Semantic Scholar: "Cite" button
3. arXiv: "Export citation" on paper page
4. BibTeX from venue (most accurate)
```

### Citation Best Practices

```
DO:
- Cite the published version if available (not arXiv)
- Include all authors (don't use "et al." in BibTeX)
- Use consistent format throughout
- Cite foundational work, not just recent

DON'T:
- Cite arXiv if published version exists
- Cite blog posts as primary sources
- Cite without reading (at least Pass 1)
- Over-cite your own work
```

## Organizing Literature

### Recommended Tools

| Tool | Best For | Cost |
|------|----------|------|
| **Zotero** | Free, browser integration | Free |
| **Mendeley** | PDF organization | Free (basic) |
| **Paperpile** | Google Docs integration | $36/year |
| **Notion** | Custom organization | Free (basic) |
| **ReadCube Papers** | Cross-platform | $36/year |

### Organization System

```
papers/
├── by-topic/
│   ├── attention-mechanisms/
│   ├── efficient-transformers/
│   └── vision-transformers/
├── by-project/
│   ├── current-project/
│   └── archived-projects/
└── to-read/
    ├── high-priority/
    └── low-priority/
```

### Reading Notes Template

```markdown
# Paper: [Title]

**Authors**: [Authors]
**Venue**: [Conference/Journal, Year]
**Link**: [URL]

## Summary (1-2 sentences)
[What does this paper do?]

## Key Contributions
1. [Contribution 1]
2. [Contribution 2]

## Method
[Brief description of approach]

## Results
- [Key result 1]
- [Key result 2]

## Strengths
- [Strength 1]
- [Strength 2]

## Weaknesses/Limitations
- [Limitation 1]
- [Limitation 2]

## Relevance to My Work
[How can I use this?]

## Key Equations/Figures
[Note any important formulas or figures]

## Questions
- [Question 1]
- [Question 2]

## Related Papers
- [Related paper 1]
- [Related paper 2]
```

## Staying Current

### Weekly Routine

```
Monday:
- Check arXiv cs.LG, cs.CL, cs.CV (last week)
- Papers With Code trending

Thursday:
- Semantic Scholar recommendations
- Twitter ML community

Monthly:
- Conference proceedings (when released)
- Survey papers in your area
```

### Key Venues by Area

```
General ML:
- NeurIPS (December)
- ICML (July)
- ICLR (May)

NLP:
- ACL (July)
- EMNLP (November)
- NAACL (June)

Computer Vision:
- CVPR (June)
- ICCV (October, odd years)
- ECCV (October, even years)

ML Systems:
- MLSys (March)
- OSDI/SOSP (systems, biannual)
```

### Newsletters and Aggregators

- **Import AI** - Weekly AI news
- **The Batch** (DeepLearning.AI) - Weekly digest
- **Papers With Code Newsletter** - New papers/code
- **Hugging Face Daily Papers** - Community picks
