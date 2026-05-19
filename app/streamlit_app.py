"""AstroClassify by Agentra — Streamlit demo for Vietnam AI Open Hackathon 2026.

Run locally:
    streamlit run app/streamlit_app.py

Run on Agentra VPS:
    streamlit run app/streamlit_app.py --server.port 8103 --server.address 0.0.0.0
    # then expose via Cloudflare named tunnel:
    cloudflared tunnel run astroclassify
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `csnet` importable when running directly with `streamlit run`
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from app.inference import ModelBundle
from app.samples import all_samples
from csnet.utils import CLASSES, NUM_CLASSES
from csnet.xai import integrated_gradients


# --------------------------------------------------------------------------- #
# Page setup                                                                   #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="AstroClassify by Agentra",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .agentra-title { font-size: 2.4rem; font-weight: 800; color: #0B3D91;
                       margin-bottom: 0.2rem; }
      .agentra-sub   { font-size: 1.05rem; color: #555; margin-top: 0; }
      .metric-card   { background: #F0F2F5; padding: 0.7rem 1rem;
                       border-radius: 8px; border-left: 4px solid #0B3D91; }
      .demo-banner   { background: #FFF3CD; padding: 0.7rem 1rem;
                       border-radius: 6px; border-left: 4px solid #FC3D21;
                       font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="agentra-title">🪐 AstroClassify <span style="color:#FC3D21">by Agentra</span></div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="agentra-sub">Multi-survey stellar spectral classification powered by '
    'Conv-KAN, trained on NVIDIA H200 · Vietnam AI Open Hackathon 2026</p>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Resource caches                                                              #
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading 5 model architectures...")
def get_bundle() -> ModelBundle:
    return ModelBundle()


@st.cache_data(show_spinner=False)
def get_samples(seq_len: int = 4096) -> list[dict]:
    return all_samples(seq_len=seq_len)


bundle = get_bundle()
samples = get_samples(seq_len=4096)

if bundle.is_random_weights():
    st.markdown(
        '<div class="demo-banner"><b>⚠ Demo mode — random weights.</b> '
        'Real weights from H200 training will be dropped into '
        '<code>weights/&lt;model&gt;.pt</code> after the hackathon. '
        'The UI, inference pipeline, and XAI flow are fully functional.</div>',
        unsafe_allow_html=True,
    )

st.write("")


# --------------------------------------------------------------------------- #
# Sidebar — sample picker + upload                                             #
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Choose spectrum")
    sample_names = [s["name"] for s in samples]
    picked = st.selectbox("Pre-loaded sample", sample_names, index=4)  # G-type default
    sample = next(s for s in samples if s["name"] == picked)

    st.markdown("---")
    st.subheader("Or upload your own")
    uploaded = st.file_uploader(
        "FITS file (4096-pixel resampled spectrum)", type=["fits", "npy"],
        help="Use the same preprocess.py output format: 1D flux array, length 4096."
    )
    if uploaded is not None:
        if uploaded.name.endswith(".npy"):
            arr = np.load(uploaded)
            sample = {
                "name": uploaded.name,
                "survey": "user upload",
                "true_class": None, "true_class_name": "?",
                "spectrum": arr.astype(np.float32),
                "wavelength": np.linspace(3800, 8000, len(arr)),
            }
        else:
            st.warning("FITS upload requires the production preprocessing step. "
                       "For the demo, try a pre-loaded sample.")

    st.markdown("---")
    st.subheader("XAI options")
    do_xai = st.checkbox("Compute Integrated Gradients", value=True)
    xai_steps = st.slider("IG steps", 10, 100, 30, help="More steps = smoother attribution but slower")
    xai_model_for = st.selectbox("XAI for which model", list(bundle.models.keys()), index=0)


# --------------------------------------------------------------------------- #
# Top row — spectrum + sample metadata                                          #
# --------------------------------------------------------------------------- #

col_spec, col_meta = st.columns([3, 1])

with col_spec:
    st.subheader("Input spectrum")
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(
        x=sample["wavelength"], y=sample["spectrum"],
        mode="lines", line=dict(color="#0B3D91", width=1),
        name="flux"
    ))
    fig_spec.update_layout(
        height=260, margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title="Wavelength (Å)", yaxis_title="Flux (normalised)",
        showlegend=False,
    )
    st.plotly_chart(fig_spec, use_container_width=True)

with col_meta:
    st.subheader("Sample")
    st.markdown(f'<div class="metric-card"><b>Name</b><br>{sample["name"]}</div>',
                unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="metric-card"><b>Survey</b><br>{sample["survey"]}</div>',
                unsafe_allow_html=True)
    st.write("")
    truth = sample["true_class_name"]
    st.markdown(f'<div class="metric-card"><b>Ground truth</b><br>{truth}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Run inference                                                                 #
# --------------------------------------------------------------------------- #

with st.spinner("Running inference on all models..."):
    results = bundle.predict_all(sample["spectrum"])


# --------------------------------------------------------------------------- #
# Predictions panel                                                             #
# --------------------------------------------------------------------------- #

st.subheader("Model predictions")

cols = st.columns(len(results))
for col, (mname, r) in zip(cols, results.items()):
    with col:
        correct = (sample["true_class"] is not None and r["pred_class"] == sample["true_class"])
        emoji = "✅" if correct else "❌" if sample["true_class"] is not None else "•"
        st.markdown(f"**{mname}** {emoji}")
        st.metric("Predicted", r["pred_class_name"], f"{r['probs'][r['pred_class']]*100:.1f}%")
        st.caption(f"Inference: {r['latency_ms']:.2f} ms")

st.write("")

# Confidence bar charts grouped
st.markdown("**Per-class confidence (all models)**")
fig_conf = go.Figure()
for mname, r in results.items():
    fig_conf.add_trace(go.Bar(
        x=CLASSES, y=r["probs"], name=mname,
        text=[f"{p*100:.0f}%" for p in r["probs"]],
        textposition="outside",
    ))
fig_conf.update_layout(
    barmode="group", height=320,
    margin=dict(l=40, r=10, t=20, b=40),
    yaxis_title="Probability", yaxis_range=[0, 1.05],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_conf, use_container_width=True)


# --------------------------------------------------------------------------- #
# XAI panel                                                                     #
# --------------------------------------------------------------------------- #

if do_xai:
    st.subheader(f"Integrated Gradients — what the {xai_model_for} model looked at")
    xai_model = bundle.models[xai_model_for]
    target_cls = results[xai_model_for]["pred_class"]

    with st.spinner(f"Computing IG with {xai_steps} steps..."):
        attribution = integrated_gradients(
            xai_model,
            torch.from_numpy(sample["spectrum"]).float(),
            target_class=target_cls,
            n_steps=xai_steps,
        )

    fig_xai = go.Figure()
    fig_xai.add_trace(go.Scatter(
        x=sample["wavelength"], y=sample["spectrum"],
        mode="lines", line=dict(color="#888", width=1), name="flux",
    ))
    # Heatmap-style overlay via marker colour
    norm_attr = attribution / (np.abs(attribution).max() + 1e-8)
    fig_xai.add_trace(go.Scatter(
        x=sample["wavelength"], y=sample["spectrum"],
        mode="markers",
        marker=dict(
            color=norm_attr, colorscale="RdBu", cmin=-1, cmax=1,
            size=4, showscale=True, colorbar=dict(title="IG"),
        ),
        name="IG attribution",
    ))
    fig_xai.update_layout(
        height=320, margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title="Wavelength (Å)", yaxis_title="Flux",
        showlegend=False,
    )
    st.plotly_chart(fig_xai, use_container_width=True)
    st.caption(
        f"Red regions push the model toward class **{CLASSES[target_cls]}**; "
        f"blue regions push away. Physically meaningful absorption/emission lines "
        f"(Balmer series, H&K, Mg b, He I 4471/6678 Å) should dominate when the "
        f"model is well-trained."
    )


# --------------------------------------------------------------------------- #
# Footer — about + speed widget                                                 #
# --------------------------------------------------------------------------- #

st.markdown("---")
foot_l, foot_r = st.columns([2, 1])

with foot_l:
    st.markdown(
        """
        **About this demo.** AstroClassify is the public face of *Agentra-Astro*'s
        submission to the **Vietnam AI Open Hackathon 2026** (NVIDIA Open Hackathons,
        DSAC Đà Nẵng). The model is a Convolutional Kolmogorov-Arnold Network
        ( **Conv-KAN** ) that scales our prior work (under review at IEEE TAI) to
        a multi-survey corpus of ≈12 M spectra from SDSS DR19, LAMOST DR8, and
        SDSS APOGEE DR17.

        Trained on **NVIDIA H200** (FP8 Transformer Engine), served on
        **Agentra's own RTX 4090 infrastructure** behind a Cloudflare tunnel.
        Code is open-source (Apache 2.0) at
        [github.com/haodpsut/conv-kan-stellar](https://github.com/haodpsut/conv-kan-stellar).
        """
    )

with foot_r:
    st.markdown("**Speed comparison**")
    device = "GPU (RTX 4090)" if torch.cuda.is_available() else "CPU"
    fastest = min(r["latency_ms"] for r in results.values())
    st.metric(f"Fastest model latency on this {device}", f"{fastest:.2f} ms")
    st.caption("After H200 training with FP8, expect a further 5–10× speed-up.")
