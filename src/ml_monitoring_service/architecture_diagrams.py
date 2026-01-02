from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelDiagramConfig:
    hidden_dim: int
    num_services: int
    num_features: int
    window_size: int
    cross_service_heads: int
    temporal_heads: int
    temporal_layers: int

    @property
    def cross_service_head_dim(self) -> int:
        if self.cross_service_heads <= 0:
            return 0
        return self.hidden_dim // self.cross_service_heads

    @property
    def temporal_d_model(self) -> int:
        return self.hidden_dim * self.num_services

    @property
    def temporal_head_dim(self) -> int:
        if self.temporal_heads <= 0:
            return 0
        return self.temporal_d_model // self.temporal_heads


# Color groups (light fills so text remains readable).
COLOR_DATA = (0.93, 0.96, 1.00, 1.00)  # light blue
COLOR_CROSS = (1.00, 0.95, 0.85, 1.00)  # light orange
COLOR_TEMPORAL = (0.89, 0.97, 0.90, 1.00)  # light green
COLOR_DECODE = (0.98, 0.90, 0.93, 1.00)  # light pink
COLOR_DEFAULT = (0.94, 0.94, 0.94, 1.00)  # neutral gray


def _draw_box(
    ax,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    url: str | None = None,
    rotation: float = 0.0,
    fontsize: int = 10,
    facecolor: tuple[float, float, float, float] | None = None,
    show_expandable_hint: bool = True,
):
    from matplotlib.patches import Rectangle

    # Use a filled rectangle so the whole box area is clickable in SVG viewers.
    # Always set edgecolor so linked boxes keep their visible border.
    box_face = facecolor or COLOR_DEFAULT
    box_edge = (0.0, 0.0, 0.0, 1.0)
    rect = Rectangle(
        (x, y),
        w,
        h,
        fill=True,
        facecolor=box_face,
        edgecolor=box_edge,
        linewidth=1.6,
    )
    if url:
        # When saved as SVG, this becomes a clickable hyperlink.
        rect.set_url(url)
    ax.add_patch(rect)
    txt = ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        rotation=rotation,
    )
    if url:
        # Make the label itself clickable too (helps in some viewers).
        txt.set_url(url)

    if url and show_expandable_hint:
        hint = ax.text(
            x + w - 0.02 * w,
            y + h - 0.08 * h,
            "Expandable",
            ha="right",
            va="top",
            fontsize=max(7, int(fontsize) - 3),
            color="#0b5fff",
        )
        # Underline isn't supported in all backends equally, but works in SVG.
        try:
            hint.set_underline(True)
        except Exception as exc:
            logger.debug("Matplotlib backend does not support underline: %s", exc)
        hint.set_url(url)
    return rect


def _draw_arrow(ax, *, x0: float, y0: float, x1: float, y1: float):
    from matplotlib.patches import FancyArrowPatch

    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
    )
    ax.add_patch(arr)
    return arr


def _setup_figure(*, figsize: tuple[float, float]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    return fig, ax, plt


def _extract_config(
    model, *, num_services: int, num_features: int, window_size: int
) -> ModelDiagramConfig:
    hidden_dim = int(getattr(model, "hidden_dim", 0) or 0)

    cross_heads = 0
    if hasattr(model, "cross_service_attention"):
        cross_heads = int(getattr(model.cross_service_attention, "num_heads", 0) or 0)

    temporal_heads = 0
    temporal_layers = 0
    if hasattr(model, "transformer_encoder"):
        layers = getattr(model.transformer_encoder, "layers", None)
        if layers is not None:
            try:
                temporal_layers = int(len(layers))
            except Exception:
                temporal_layers = 0
            try:
                temporal_heads = int(getattr(layers[0].self_attn, "num_heads", 0) or 0)
            except Exception:
                temporal_heads = 0

    return ModelDiagramConfig(
        hidden_dim=hidden_dim,
        num_services=int(num_services),
        num_features=int(num_features),
        window_size=int(window_size),
        cross_service_heads=cross_heads,
        temporal_heads=temporal_heads,
        temporal_layers=temporal_layers,
    )


def write_model_architecture_diagrams(
    *, model, num_services: int, num_features: int, window_size: int, out_dir: Path
) -> None:
    """Create a small bundle of manual diagrams.

    Produces:
      - overview.svg (clickable links to deeper diagrams)
      - cross_service_attention.svg
      - temporal_transformer_layer.svg
      - index.html (convenience page for local viewing)

    Note: hyperlinks work best when opening the SVG locally in a browser.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _extract_config(
        model,
        num_services=num_services,
        num_features=num_features,
        window_size=window_size,
    )

    _write_overview_svg(cfg, out_dir)
    _write_cross_service_attention_svg(cfg, out_dir)
    _write_temporal_transformer_layer_svg(cfg, out_dir)
    _write_attention_mechanism_deep_dive_svg(cfg, out_dir)
    _write_index_html(cfg, out_dir)


def _write_attention_mechanism_deep_dive_svg(
    cfg: ModelDiagramConfig, out_dir: Path
) -> None:
    """Deep dive into (scaled dot-product) attention used inside MHA.

    This is a generic attention mechanism diagram (applies to cross-service and temporal attention).
    """

    fig, ax, plt = _setup_figure(figsize=(14, 9))

    ax.text(
        0.5,
        0.98,
        "Attention mechanism — deep dive (scaled dot-product attention + multi-head)\n"
        "Applies to both: cross-service attention and temporal self-attention",
        ha="center",
        va="top",
        fontsize=12,
    )

    # Top row: projections into Q/K/V
    _draw_box(
        ax,
        x=0.05,
        y=0.78,
        w=0.24,
        h=0.12,
        text="Input tokens\nX ∈ ℝ[n_tokens, d_model]\n(services at t OR timesteps in window)",
        facecolor=COLOR_DATA,
        fontsize=9,
    )

    _draw_arrow(ax, x0=0.29, y0=0.84, x1=0.33, y1=0.84)

    _draw_box(ax, x=0.33, y=0.84, w=0.17, h=0.08, text="Q = XWq", facecolor=COLOR_CROSS)
    _draw_box(ax, x=0.33, y=0.74, w=0.17, h=0.08, text="K = XWk", facecolor=COLOR_CROSS)
    _draw_box(ax, x=0.33, y=0.64, w=0.17, h=0.08, text="V = XWv", facecolor=COLOR_CROSS)

    ax.text(
        0.52,
        0.84,
        "(Per head: d_k = d_model / n_heads)",
        ha="left",
        va="center",
        fontsize=9,
    )

    # Middle: attention weights
    _draw_arrow(ax, x0=0.50, y0=0.79, x1=0.58, y1=0.79)
    _draw_box(
        ax,
        x=0.58,
        y=0.72,
        w=0.30,
        h=0.14,
        text="Scores = (QKᵀ) / √d_k\nSoftmax(scores) → weights\nA ∈ ℝ[n_tokens, n_tokens]",
        facecolor=COLOR_TEMPORAL,
        fontsize=9,
    )

    # Weighted sum
    _draw_arrow(ax, x0=0.73, y0=0.72, x1=0.73, y1=0.62)
    _draw_box(
        ax,
        x=0.58,
        y=0.52,
        w=0.30,
        h=0.10,
        text="Output = A · V\nY ∈ ℝ[n_tokens, d_k] (per head)",
        facecolor=COLOR_TEMPORAL,
        fontsize=9,
    )

    # Multi-head combine
    _draw_arrow(ax, x0=0.73, y0=0.52, x1=0.73, y1=0.44)
    _draw_box(
        ax,
        x=0.58,
        y=0.34,
        w=0.30,
        h=0.10,
        text="Concat heads → ℝ[n_tokens, d_model]\nWo projection → ℝ[n_tokens, d_model]",
        facecolor=COLOR_CROSS,
        fontsize=9,
    )

    # Side callout: intuition
    _draw_box(
        ax,
        x=0.05,
        y=0.40,
        w=0.45,
        h=0.22,
        text=(
            "Intuition (at a glance)\n"
            "• Each token asks a question (Q): what should I look for?\n"
            "• Each token advertises what it has (K) and what it can contribute (V).\n"
            "• Similarity(Q, K) decides how much of each V to mix into the output.\n"
            "• Multiple heads let the model learn several mixing patterns at once."
        ),
        facecolor=COLOR_DEFAULT,
        fontsize=9,
    )

    # Token domains
    _draw_box(
        ax,
        x=0.05,
        y=0.16,
        w=0.45,
        h=0.18,
        text=(
            "What are the tokens here?\n"
            "• Cross-service attention: tokens = services (S tokens) at fixed time t.\n"
            "• Temporal attention: tokens = timesteps (T tokens) in the window.\n"
            "In both cases, attention builds a weighted mixture of information across tokens."
        ),
        facecolor=COLOR_DEFAULT,
        fontsize=9,
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    out_path = out_dir / "attention_mechanism_deep_dive.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_overview_svg(cfg: ModelDiagramConfig, out_dir: Path) -> None:
    fig, ax, plt = _setup_figure(figsize=(12, 9))

    title = (
        "HybridAutoencoderTransformerModel — manual overview (click boxes for details)\n"
        f"H={cfg.hidden_dim}, services={cfg.num_services}, features={cfg.num_features}, window={cfg.window_size}"
    )
    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=12)

    x0 = 0.08
    box_w = 0.84
    # Keep everything on-canvas (Decoder/Output were drifting below 0.0).
    box_h = 0.06
    vgap = 0.02

    blocks: list[tuple[str, str | None, tuple[float, float, float, float]]] = [
        ("Input\n[B,T,S,F]", None, COLOR_DATA),
        ("Concat time features\n[B,T,S,F+4]", None, COLOR_DATA),
        ("Feature encoder\nLinear(F+4→H)", None, COLOR_DATA),
        ("Add service embeddings\n[S,H]", None, COLOR_DATA),
        (
            f"Cross-service attention\nMHA(embed=H={cfg.hidden_dim}, heads={cfg.cross_service_heads}, head_dim≈{cfg.cross_service_head_dim})",
            "cross_service_attention.svg",
            COLOR_CROSS,
        ),
        ("Flatten services\n[B,T,S·H]", None, COLOR_TEMPORAL),
        ("Positional encoding", None, COLOR_TEMPORAL),
        (
            f"Temporal transformer\nEncoder ×{cfg.temporal_layers} (d_model=S·H={cfg.temporal_d_model}, heads={cfg.temporal_heads}, head_dim≈{cfg.temporal_head_dim})",
            "temporal_transformer_layer.svg",
            COLOR_TEMPORAL,
        ),
        ("Reshape\n[B,T,S,H]", None, COLOR_TEMPORAL),
        ("Decoder\nLinear(H→F)", None, COLOR_DECODE),
        ("Output\n[B,T,S,F]", None, COLOR_DECODE),
    ]

    y = 0.90
    for i, (label, url, color) in enumerate(blocks):
        _draw_box(
            ax,
            x=x0,
            y=y - box_h,
            w=box_w,
            h=box_h,
            text=label,
            url=url,
            facecolor=color,
        )
        if i < len(blocks) - 1:
            _draw_arrow(
                ax,
                x0=x0 + box_w / 2,
                y0=y - box_h,
                x1=x0 + box_w / 2,
                y1=y - box_h - vgap,
            )
        y -= box_h + vgap

    out_path = out_dir / "overview.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_cross_service_attention_svg(cfg: ModelDiagramConfig, out_dir: Path) -> None:
    fig, ax, plt = _setup_figure(figsize=(14, 7))

    ax.text(
        0.5,
        0.98,
        "Cross-service attention (per timestep): multi-head attention over services\n"
        f"embed_dim=H={cfg.hidden_dim}, heads={cfg.cross_service_heads}, head_dim≈{cfg.cross_service_head_dim}",
        ha="center",
        va="top",
        fontsize=12,
    )

    # Left: input tokens = services at one timestep
    _draw_box(
        ax,
        x=0.05,
        y=0.40,
        w=0.18,
        h=0.18,
        text="Input at t\n[S,H]",
        facecolor=COLOR_DATA,
    )
    _draw_arrow(ax, x0=0.23, y0=0.49, x1=0.30, y1=0.49)

    # Middle: show Q/K/V projections then split heads.
    _draw_box(
        ax, x=0.30, y=0.64, w=0.18, h=0.12, text="Wq: Linear(H→H)", facecolor=COLOR_DATA
    )
    _draw_box(
        ax, x=0.30, y=0.48, w=0.18, h=0.12, text="Wk: Linear(H→H)", facecolor=COLOR_DATA
    )
    _draw_box(
        ax, x=0.30, y=0.32, w=0.18, h=0.12, text="Wv: Linear(H→H)", facecolor=COLOR_DATA
    )

    for yy in (0.70, 0.54, 0.38):
        _draw_arrow(ax, x0=0.49, y0=yy, x1=0.54, y1=yy)

    # Right: heads
    heads = max(1, cfg.cross_service_heads)
    head_w = 0.36 / heads
    base_x = 0.54
    y_head = 0.30
    h_head = 0.46

    ax.text(0.72, 0.80, "Split into heads", ha="center", va="center", fontsize=10)

    for i in range(heads):
        x = base_x + i * head_w
        _draw_box(
            ax,
            x=x,
            y=y_head,
            w=head_w - 0.01,
            h=h_head,
            text=f"Head {i + 1}\nAttn(Q,K,V)\n[S,head_dim]",
            rotation=90,
            fontsize=9,
            facecolor=COLOR_CROSS,
        )

    # Concat + output projection
    _draw_arrow(ax, x0=0.90, y0=0.53, x1=0.93, y1=0.53)
    _draw_box(
        ax,
        x=0.93,
        y=0.46,
        w=0.20,
        h=0.14,
        text="Concat heads\n[S,H]",
        facecolor=COLOR_CROSS,
    )
    _draw_arrow(ax, x0=1.13, y0=0.53, x1=1.18, y1=0.53)
    _draw_box(
        ax,
        x=1.18,
        y=0.46,
        w=0.20,
        h=0.14,
        text="Wo: Linear(H→H)\nOutput [S,H]",
        facecolor=COLOR_CROSS,
    )

    # Keep everything inside bounds for tight bbox.
    ax.set_xlim(0, 1.40)
    ax.set_ylim(0, 1.0)

    out_path = out_dir / "cross_service_attention.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_temporal_transformer_layer_svg(
    cfg: ModelDiagramConfig, out_dir: Path
) -> None:
    fig, ax, plt = _setup_figure(figsize=(13, 8))

    ax.text(
        0.5,
        0.98,
        "Temporal transformer encoder layer (one layer)\n"
        f"d_model=S·H={cfg.temporal_d_model}, heads={cfg.temporal_heads}, head_dim≈{cfg.temporal_head_dim}   (stacked ×{cfg.temporal_layers} layers)",
        ha="center",
        va="top",
        fontsize=12,
    )

    # One encoder layer: MHA over time + FFN + residuals.
    _draw_box(
        ax,
        x=0.06,
        y=0.76,
        w=0.25,
        h=0.12,
        text="Input\n[T, d_model]",
        facecolor=COLOR_DATA,
    )
    _draw_arrow(ax, x0=0.31, y0=0.82, x1=0.36, y1=0.82)

    _draw_box(
        ax,
        x=0.36,
        y=0.74,
        w=0.30,
        h=0.16,
        text=f"Self-attention over time\nMHA(d_model, heads={max(1, cfg.temporal_heads)})\n(T tokens)",
        facecolor=COLOR_TEMPORAL,
    )

    # Per-head blocks
    heads = max(1, cfg.temporal_heads)
    head_cols = min(heads, 8)
    head_w = 0.52 / head_cols
    y0 = 0.46
    h0 = 0.18
    ax.text(0.62, 0.69, "Per-head (conceptual)", ha="center", va="center", fontsize=10)

    for i in range(head_cols):
        _draw_box(
            ax,
            x=0.36 + i * head_w,
            y=y0,
            w=head_w - 0.01,
            h=h0,
            text=f"Head {i + 1}\nAttn(Q,K,V)",
            rotation=90,
            fontsize=9,
            facecolor=COLOR_TEMPORAL,
        )

    if heads > head_cols:
        ax.text(
            0.90,
            0.56,
            f"… +{heads - head_cols} more heads",
            ha="left",
            va="center",
            fontsize=10,
        )

    _draw_arrow(ax, x0=0.66, y0=0.74, x1=0.66, y1=0.64)

    # After per-head attention, outputs are concatenated back into [T, d_model]
    # before the residual + normalization step.
    concat_y = 0.41
    concat_h = 0.04
    _draw_arrow(ax, x0=0.66, y0=0.46, x1=0.66, y1=concat_y + concat_h)
    _draw_box(
        ax,
        x=0.36,
        y=concat_y,
        w=0.30,
        h=concat_h,
        text="Concat heads\n[T, d_model]",
        facecolor=COLOR_TEMPORAL,
        fontsize=9,
    )
    _draw_arrow(ax, x0=0.66, y0=concat_y, x1=0.66, y1=0.40)

    _draw_box(
        ax, x=0.36, y=0.30, w=0.30, h=0.10, text="Add & Norm", facecolor=COLOR_TEMPORAL
    )

    _draw_arrow(ax, x0=0.66, y0=0.30, x1=0.66, y1=0.26)

    _draw_box(
        ax,
        x=0.36,
        y=0.12,
        w=0.30,
        h=0.12,
        text="FFN\nLinear(d→d_ff) → GELU → Linear(d_ff→d)",
        facecolor=COLOR_TEMPORAL,
    )

    _draw_arrow(ax, x0=0.66, y0=0.12, x1=0.66, y1=0.08)
    _draw_box(
        ax, x=0.36, y=0.00, w=0.30, h=0.08, text="Add & Norm", facecolor=COLOR_TEMPORAL
    )

    _draw_arrow(ax, x0=0.66, y0=0.00, x1=0.74, y1=0.00)
    _draw_box(
        ax,
        x=0.74,
        y=-0.02,
        w=0.25,
        h=0.12,
        text="Output\n[T,d_model]",
        facecolor=COLOR_DATA,
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.08, 1.0)

    out_path = out_dir / "temporal_transformer_layer.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _write_index_html(cfg: ModelDiagramConfig, out_dir: Path) -> None:
    # MLflow's artifact HTML preview can break relative <img src="..."> URLs.
    # Inline the SVG contents so the diagrams render reliably in the UI.
    overview_svg = (out_dir / "overview.svg").read_text(encoding="utf-8")
    cross_svg = (out_dir / "cross_service_attention.svg").read_text(encoding="utf-8")
    temporal_svg = (out_dir / "temporal_transformer_layer.svg").read_text(
        encoding="utf-8"
    )
    attention_svg = (out_dir / "attention_mechanism_deep_dive.svg").read_text(
        encoding="utf-8"
    )

    overview_nontech = "A simple map of the model: data goes in at the top and comes out at the bottom, step by step."
    overview_expl = (
        "This is a high-level, end-to-end view of how tensors flow through the model. "
        "Each box is a transformation applied to the batch/window. "
        "The key idea is: encode metrics into embeddings, model dependencies across services at a fixed time, "
        "then model dependencies across time inside the window, and finally decode back to metric space."
    )

    cross_nontech = (
        "A zoom-in on how services influence each other at the same moment in time."
    )
    cross_expl = (
        "At a fixed timestamp t, each service embedding is treated like a token. "
        "Multi-head attention allows every service to pay attention to (and thereby be influenced by) every other service (capturing cross-service correlations). "
        "Different heads can learn different interaction patterns, then outputs are concatenated and projected back to [S,H]."
    )

    temporal_nontech = (
        "A zoom-in on how the model learns patterns over time within a window."
    )
    temporal_expl = (
        "For each window, we flatten all services into one vector per timestep (d_model = S·H). "
        "Self-attention runs over the T timesteps to capture temporal dependencies. "
        "Residual connections + layer norm stabilize learning, and the feed-forward network adds nonlinearity."
    )

    attention_nontech = "A close-up of how the model decides what to ‘pay attention’ to when mixing information between tokens."
    attention_expl_nontech = (
        "Think of each token as reading a room: it looks at all other tokens and decides who is most relevant. "
        "It then builds a new representation by blending the most relevant pieces together. "
        "Multiple heads are like multiple ‘perspectives’ doing this blending in parallel."
    )
    attention_expl_tech = (
        "Technical view (scaled dot-product attention): we compute Q = XWq, K = XWk, V = XWv. "
        "Then attention weights are A = softmax((QKᵀ)/√d_k), and the per-head output is Y = A·V. "
        "Multi-head attention runs this in parallel across heads (smaller d_k per head), concatenates the results, "
        "and applies an output projection Wo. The only difference between cross-service and temporal attention is what counts as a token (S vs T)."
    )

    legend = """<div class=\"legend\">
    <div><span class=\"swatch\" style=\"background:#EDF5FF\"></span> <strong>Data/Encoding</strong> (inputs, reshaping, embedding/encoding)</div>
    <div><span class=\"swatch\" style=\"background:#FFF2D9\"></span> <strong>Cross-service</strong> (services influence each other at the same time)</div>
    <div><span class=\"swatch\" style=\"background:#E3F7E6\"></span> <strong>Temporal</strong> (time dependencies within the window)</div>
    <div><span class=\"swatch\" style=\"background:#FBE6ED\"></span> <strong>Decode/Output</strong> (reconstruct metrics back in feature space)</div>
</div>"""

    notation = """<div class=\"notation\">
    <h2>Notation & intuition</h2>
    <ul>
        <li><strong>B</strong> = batch size: how many windows are processed at once.</li>
        <li><strong>T</strong> = timesteps per window: the length of the time window.</li>
        <li><strong>S</strong> = services: how many services/tokens exist at each time (one token per service).</li>
        <li><strong>F</strong> = features: how many raw metrics per service at each time.</li>
        <li><strong>H</strong> = hidden/embedding size: how many numbers represent each service after encoding.</li>
        <li><strong>d_model</strong> = model width for the temporal transformer. Here it is <strong>S·H</strong> because services are flattened into one vector per timestep.</li>
        <li><strong>Q, K, V</strong> (Query/Key/Value): think of <strong>Q</strong> as “what am I looking for?”, <strong>K</strong> as “what do I contain?”, and <strong>V</strong> as “what information do I pass along?”. Attention compares Q to K to decide how much of each V to mix in.</li>
    </ul>
</div>"""

    html = f"""<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Model architecture diagrams</title>
        <style>
            body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
            .row {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
            .card {{ padding: 12px; border: 1px solid #eee; border-radius: 8px; }}
            .desc {{ margin: 0 0 10px 0; color: #333; line-height: 1.35; }}
            .legend {{ display: grid; gap: 6px; margin: 8px 0 0 0; }}
            .swatch {{ display: inline-block; width: 14px; height: 14px; border: 1px solid #000; vertical-align: -2px; margin-right: 6px; }}
            .notation ul {{ margin: 8px 0 0 18px; }}
            a {{ color: #0b5fff; }}
            .svg-wrap {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px; overflow: auto; }}
            .svg-wrap svg {{ width: 100%; height: auto; }}
            details > summary {{ cursor: pointer; font-weight: 600; }}
        </style>
    </head>
    <body>
        <h1>Model architecture diagrams</h1>
        <p>
            H={cfg.hidden_dim}, services={cfg.num_services}, features={cfg.num_features}, window={cfg.window_size}
        </p>

        <div class=\"card\">{notation}{legend}</div>

        <div class=\"row\">
            <div class=\"card\">
                <h2>Overview</h2>
                <p>
                    Open <a href=\"overview.svg\">overview.svg</a> (clickable boxes).
                </p>
                <p class=\"desc\"><strong>Non-technical:</strong> {overview_nontech}</p>
                <p class=\"desc\">{overview_expl}</p>
                <div class=\"svg-wrap\">{overview_svg}</div>
            </div>

            <div class=\"card\">
                <h2>Cross-service attention</h2>
                <p>
                    <a href=\"cross_service_attention.svg\">cross_service_attention.svg</a>
                </p>
                <p class=\"desc\"><strong>Non-technical:</strong> {cross_nontech}</p>
                <p class=\"desc\">{cross_expl}</p>
                <div class=\"svg-wrap\">{cross_svg}</div>
            </div>

            <div class=\"card\">
                <h2>Temporal transformer layer</h2>
                <p>
                    <a href=\"temporal_transformer_layer.svg\">temporal_transformer_layer.svg</a>
                </p>
                <p class=\"desc\"><strong>Non-technical:</strong> {temporal_nontech}</p>
                <p class=\"desc\">{temporal_expl}</p>
                <div class=\"svg-wrap\">{temporal_svg}</div>
            </div>

            <div class=\"card\">
                <h2>Attention mechanism (deep dive)</h2>
                <p>
                    <a href=\"attention_mechanism_deep_dive.svg\">attention_mechanism_deep_dive.svg</a>
                </p>
                <p class=\"desc\"><strong>Non-technical:</strong> {attention_nontech}</p>
                <p class=\"desc\">{attention_expl_nontech}</p>
                <p class=\"desc\"><strong>Technical:</strong> {attention_expl_tech}</p>
                <div class=\"svg-wrap\">{attention_svg}</div>
            </div>

            <div class=\"card\">
                <details>
                    <summary>Local viewing tips</summary>
                    <p>
                        If the links don’t open correctly inside MLflow’s preview, download this folder and open
                        <code>index.html</code> locally in your browser. The SVG boxes in <code>overview.svg</code> are clickable.
                    </p>
                </details>
            </div>
        </div>
    </body>
</html>
"""
    (out_dir / "index.html").write_text(html)
