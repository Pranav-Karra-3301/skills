# ML Visualization Guide

## Publication-Quality Matplotlib

### Global Settings

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication settings
plt.rcParams.update({
    # Font
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],

    # Axes
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Ticks
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,

    # Legend
    'legend.fontsize': 11,
    'legend.frameon': False,

    # Figure
    'figure.figsize': (8, 6),
    'figure.dpi': 150,

    # Saving
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # Lines
    'lines.linewidth': 2,
    'lines.markersize': 8,
})
```

### Figure Sizing for Papers

```python
# Common column widths (in inches)
SINGLE_COLUMN = 3.5   # Single column figure
DOUBLE_COLUMN = 7.0   # Full width figure
GOLDEN_RATIO = 1.618

def get_figure_size(width_type='single', height_ratio=None):
    """Get figure size for publication."""
    if width_type == 'single':
        width = SINGLE_COLUMN
    elif width_type == 'double':
        width = DOUBLE_COLUMN
    else:
        width = width_type

    if height_ratio is None:
        height = width / GOLDEN_RATIO
    else:
        height = width * height_ratio

    return (width, height)

# Usage
fig, ax = plt.subplots(figsize=get_figure_size('single'))
```

## Colorblind-Friendly Palettes

### Recommended Palettes

```python
# Tol's bright (excellent for categorical)
TOL_BRIGHT = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

# Okabe-Ito (most widely recommended)
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']

# IBM Design
IBM_DESIGN = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']

# Paul Tol's muted
TOL_MUTED = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

def get_colorblind_palette(n_colors=None, palette='okabe'):
    """Get colorblind-friendly color palette."""
    palettes = {
        'okabe': OKABE_ITO,
        'tol_bright': TOL_BRIGHT,
        'tol_muted': TOL_MUTED,
        'ibm': IBM_DESIGN,
    }

    colors = palettes.get(palette, OKABE_ITO)
    if n_colors:
        return colors[:n_colors]
    return colors
```

### Built-in Options

```python
# Sequential (for continuous values)
cmap = plt.cm.viridis    # Most popular, colorblind-safe
cmap = plt.cm.cividis    # Specifically designed for colorblindness
cmap = plt.cm.plasma     # Good contrast

# Diverging (for values around a center)
cmap = plt.cm.RdBu       # Red-Blue
cmap = plt.cm.coolwarm   # Cool-Warm

# Categorical (use custom palettes above)
plt.style.use('seaborn-v0_8-colorblind')
```

## Standard ML Plots

### Training Curves

```python
def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves.

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train', color='#0072B2')
    ax.plot(history['val_loss'], label='Validation', color='#D55E00')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training Loss')

    # Accuracy (or other metric)
    ax = axes[1]
    ax.plot(history['train_acc'], label='Train', color='#0072B2')
    ax.plot(history['val_acc'], label='Validation', color='#D55E00')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_title('Accuracy')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')

    return fig
```

### Confusion Matrix

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, save_path=None):
    """
    Plot confusion matrix.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        class_names: list of class names
        normalize: whether to normalize values
        save_path: optional path to save
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores, class_names=None, save_path=None):
    """
    Plot ROC curve for binary or multiclass.

    Args:
        y_true: ground truth (binary or one-hot for multiclass)
        y_scores: predicted probabilities
        class_names: list of class names
        save_path: optional path to save
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_colorblind_palette(len(y_scores[0]) if len(y_scores.shape) > 1 else 1)

    if len(y_scores.shape) == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[0], lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multiclass
        n_classes = y_scores.shape[1]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            name = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

### Learning Rate Schedule

```python
def plot_lr_schedule(scheduler, num_steps, save_path=None):
    """
    Visualize learning rate schedule.
    """
    lrs = []
    optimizer = scheduler.optimizer

    for _ in range(num_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lrs, color='#0072B2')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')

    # Log scale if range is large
    if max(lrs) / min(lrs) > 100:
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

### Feature Importance

```python
def plot_feature_importance(importance, feature_names, top_k=20, save_path=None):
    """
    Plot feature importance (horizontal bar chart).
    """
    # Sort by importance
    indices = np.argsort(importance)[-top_k:]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))

    ax.barh(range(len(indices)), importance[indices], color='#0072B2')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_k} Feature Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

### Loss Landscape (2D)

```python
def plot_loss_landscape(model, criterion, loader, device, resolution=20, save_path=None):
    """
    Plot 2D loss landscape around current parameters.
    """
    import copy

    # Get current parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}

    # Random directions
    direction1 = {name: torch.randn_like(param) for name, param in model.named_parameters()}
    direction2 = {name: torch.randn_like(param) for name, param in model.named_parameters()}

    # Normalize directions
    for d in [direction1, direction2]:
        norm = sum(p.norm() ** 2 for p in d.values()).sqrt()
        for name in d:
            d[name] /= norm

    # Compute loss at grid points
    alphas = np.linspace(-1, 1, resolution)
    betas = np.linspace(-1, 1, resolution)
    losses = np.zeros((resolution, resolution))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Set parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(
                        original_params[name] +
                        alpha * direction1[name] +
                        beta * direction2[name]
                    )

            # Compute loss
            total_loss = 0
            for batch in loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    output = model(x)
                    loss = criterion(output, y)
                total_loss += loss.item()
                break  # Just one batch for speed

            losses[i, j] = total_loss

    # Restore original parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(original_params[name])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(alphas, betas, losses.T, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Loss')
    ax.plot(0, 0, 'r*', markersize=15, label='Current')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_title('Loss Landscape')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

## Interactive Plots with Plotly

### Training Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_training_dashboard(history):
    """
    Create interactive training dashboard.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss', 'Accuracy', 'Learning Rate', 'Gradient Norm')
    )

    epochs = list(range(1, len(history['train_loss']) + 1))

    # Loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', mode='lines'),
        row=1, col=1
    )

    # Accuracy
    if 'train_acc' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', mode='lines'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc', mode='lines'),
            row=1, col=2
        )

    # Learning rate
    if 'lr' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['lr'], name='LR', mode='lines'),
            row=2, col=1
        )

    # Gradient norm
    if 'grad_norm' in history:
        fig.add_trace(
            go.Scatter(x=epochs, y=history['grad_norm'], name='Grad Norm', mode='lines'),
            row=2, col=2
        )

    fig.update_layout(height=600, showlegend=True)
    return fig

# Save as HTML
# fig.write_html('training_dashboard.html')
```

## Saving Figures

### For Papers

```python
def save_figure_for_paper(fig, name, formats=['pdf', 'png']):
    """
    Save figure in multiple formats for publication.
    """
    for fmt in formats:
        path = f'figures/{name}.{fmt}'
        if fmt == 'pdf':
            fig.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.02)
        else:
            fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight', pad_inches=0.02)

    print(f"Saved: {', '.join([f'figures/{name}.{f}' for f in formats])}")
```

### For Presentations

```python
def save_figure_for_slides(fig, name):
    """
    Save figure optimized for slides (larger fonts, higher contrast).
    """
    # Temporarily increase font sizes
    with plt.rc_context({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    }):
        fig.savefig(f'figures/{name}_slides.png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
```
