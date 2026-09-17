"""
METASPEQ Haridra NIR — interactive analysis app.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chemometrics import (  # noqa: E402
    DEFAULT_BAND, PREPROCESSING_CHOICES, band_mask, component_scan,
    compute_metrics, interpret_rpd, leave_one_sample_out, load_starch_dataset,
    preprocess, train_starch_model,
)

NAVY = "#1F4E79"
BLUE = "#2E75B6"
GREEN = "#2E7D32"
AMBER = "#E67E22"
RED = "#B02418"

st.set_page_config(page_title="METASPEQ Haridra NIR",
                   page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1280px;}
      h1, h2, h3 {color: #1F4E79;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .verdict {padding: .65rem .9rem; border-radius: 6px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data/model

@st.cache_data(show_spinner="Loading spectra…")
def get_data():
    return load_starch_dataset()


@st.cache_resource(show_spinner="Building calibration model…")
def get_model(n_components: int, method: str):
    return train_starch_model(n_components, method, DEFAULT_BAND)


@st.cache_data(show_spinner="Cross-validating…")
def get_cv_predictions(n_components: int, method: str):
    wl, X, y, groups = get_data()
    Xp = preprocess(X[:, band_mask(wl, DEFAULT_BAND)], method)
    return y, leave_one_sample_out(Xp, y, groups, n_components), groups


@st.cache_data(show_spinner="Scanning model complexity…")
def get_component_scan(method: str):
    wl, X, y, groups = get_data()
    Xp = preprocess(X[:, band_mask(wl, DEFAULT_BAND)], method)
    return component_scan(Xp, y, groups, 14)


@st.cache_data
def get_curcuminoid():
    return pd.read_csv(ROOT / "data" / "curcuminoid_linearity.csv")


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🌿 METASPEQ Haridra NIR")
    st.caption("Curcuminoid assay and starch adulteration screening on the "
               "MQNIR-DR26A, with SPECTRAFIND chemometrics.")
    st.divider()

    st.subheader("Model configuration")
    n_components = st.slider("PLS latent variables", 1, 14, 8,
                             help="8 is the validated default. Below 6 the "
                                  "model underfits; above 8 there is no gain.")
    method = st.selectbox("Preprocessing", PREPROCESSING_CHOICES, index=0)

    if n_components < 6:
        st.warning(f"{n_components} components underfits this data. "
                   "See the Model Diagnostics tab.")

    st.divider()
    st.caption(
        "**Validation** — all predictions shown are leave-one-sample-out. "
        "The 16 replicate scans of each physical sample are withheld together, "
        "so the model is always scored on a sample it has not seen."
    )

model = get_model(n_components, method)
cv = model.cv_metrics
wl, X, y, groups = get_data()


# ---------------------------------------------------------------- header

st.title("Haridra Dry Extract — NIR Quality Analysis")
st.caption("Ayudyog Private Limited · METASPEQ  |  in association with "
           "Mendine Pharmaceuticals Pvt. Ltd.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("R² (cross-validated)", f"{cv['r2']:.3f}")
c2.metric("RMSECV", f"{cv['rmse']:.2f} % w/w")
c3.metric("Bias", f"{cv['bias']:+.2f} % w/w")
c4.metric("RPD", f"{cv['rpd']:.2f}")
c5.metric("Calibration range",
          f"{cv['ref_min']:.1f}–{cv['ref_max']:.1f} %")

rpd = cv["rpd"]
colour = GREEN if rpd >= 3 else (AMBER if rpd >= 2 else RED)
st.markdown(
    f"<div class='verdict' style='background:{colour}18;color:{colour};"
    f"border-left:5px solid {colour}'>RPD {rpd:.2f} — {interpret_rpd(rpd)}</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Predict", "Spectra", "Model Diagnostics",
                "Curcuminoid Assay", "Model Card"])


# ---------------------------------------------------------------- 1 predict

with tabs[0]:
    st.subheader("Predict starch content from a spectrum")
    st.write("Pick a measured spectrum from the study, or upload your own "
             "as a CSV with 256 absorbance values on the METASPEQ wavelength grid.")

    source = st.radio("Spectrum source", ["Use a study sample", "Upload a CSV"],
                      horizontal=True, label_visibility="collapsed")

    spectrum = None
    truth = None

    if source == "Use a study sample":
        ids = sorted(np.unique(groups), key=lambda s: int(s[1:]))
        left, right = st.columns([2, 3])
        with left:
            chosen = st.selectbox("Sample", ids,
                                  format_func=lambda s: f"{s} — "
                                  f"{y[groups == s][0]:.2f} % starch")
            rows = np.where(groups == chosen)[0]
            rep = st.selectbox("Replicate scan", range(1, len(rows) + 1))
        spectrum = X[rows[rep - 1]]
        truth = float(y[rows[0]])
    else:
        up = st.file_uploader("Absorbance CSV", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up, header=None)
            vals = raw.to_numpy(dtype=float).ravel()
            if vals.size != 256:
                st.error(f"Expected 256 absorbance values, received {vals.size}.")
            else:
                spectrum = vals
        else:
            st.info("Upload a single-row or single-column CSV of 256 values.")

    if spectrum is not None:
        pred = float(model.predict(spectrum)[0])
        band = model.cv_metrics
        margin = cv["rmse"]

        if pred < band["ref_min"] - margin or pred > band["ref_max"] + margin:
            zone, zcol, msg = ("RED", RED,
                               "Outside the calibration range. Do not report "
                               "this value — the model is extrapolating.")
        elif (pred < band["ref_min"] + margin) or (pred > band["ref_max"] - margin):
            zone, zcol, msg = ("AMBER", AMBER,
                               "Near the edge of the calibration range. "
                               "Confirm by the reference method before acting.")
        else:
            zone, zcol, msg = ("GREEN", GREEN,
                               "Within the calibration range.")

        pc1, pc2 = st.columns([1, 2])
        with pc1:
            st.metric("Predicted starch", f"{pred:.2f} % w/w")
            if truth is not None:
                st.metric("Gravimetric reference", f"{truth:.2f} % w/w",
                          delta=f"{pred - truth:+.2f}")
            st.markdown(
                f"<div class='verdict' style='background:{zcol}18;color:{zcol};"
                f"border-left:5px solid {zcol}'>Confidence zone: {zone}</div>",
                unsafe_allow_html=True)
            st.caption(msg)
            st.caption(f"Expected uncertainty ≈ ±{margin:.2f} % w/w (1 RMSECV).")

        with pc2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wl, y=spectrum, mode="lines",
                                     line=dict(color=BLUE, width=2),
                                     name="Selected spectrum"))
            lo, hi = DEFAULT_BAND
            fig.add_vrect(x0=lo, x1=hi, fillcolor=NAVY, opacity=0.07,
                          line_width=0,
                          annotation_text="Model band", annotation_position="top left")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="Wavelength (nm)",
                              yaxis_title="Absorbance",
                              showlegend=False)
            st.plotly_chart(fig, width="stretch")

        st.caption("This model screens for gross starch adulteration. It is not "
                   "validated for release testing or for reporting against a "
                   "specification limit.")


# ---------------------------------------------------------------- 2 spectra

with tabs[1]:
    st.subheader("Spectral explorer")
    st.write("Raw absorbance rises with starch content across the whole range. "
             "SNV removes the scattering and packing component so the model can "
             "work on the chemistry.")

    show = st.radio("View", ["Raw absorbance", "After preprocessing"],
                    horizontal=True)
    n_show = st.slider("Concentration levels to display", 3, 43, 9)

    levels = np.unique(y)
    picked = levels[np.linspace(0, len(levels) - 1, n_show).astype(int)]

    if show == "Raw absorbance":
        data, ylab = X, "Absorbance"
        xaxis = wl
    else:
        mask = band_mask(wl, DEFAULT_BAND)
        data = preprocess(X[:, mask], method)
        ylab = f"Preprocessed ({method})"
        xaxis = wl[mask]

    fig = go.Figure()
    for i, lv in enumerate(picked):
        idx = np.where(y == lv)[0][0]
        shade = i / max(len(picked) - 1, 1)
        fig.add_trace(go.Scatter(
            x=xaxis, y=data[idx], mode="lines", name=f"{lv:.1f} %",
            line=dict(width=1.6,
                      color=f"rgb({int(31+200*shade)},{int(78-40*shade)},{int(121-60*shade)})")))
    fig.update_layout(height=470, xaxis_title="Wavelength (nm)",
                      yaxis_title=ylab, legend_title="Starch",
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.info("The dominant band near 1450 nm is the water O–H first overtone. "
            "It is the largest single source of variation in this data, which "
            "is why sample moisture must be controlled for reliable results.")


# ---------------------------------------------------------------- 3 diag

with tabs[2]:
    st.subheader("Model diagnostics")

    y_t, y_p, _ = get_cv_predictions(n_components, method)
    m = compute_metrics(y_t, y_p)

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**Predicted vs reference** (leave-one-sample-out)")
        fig = go.Figure()
        lim = [-2, float(y_t.max()) + 3]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Identity",
                                 line=dict(color="#999", dash="dash", width=1.4)))
        fig.add_trace(go.Scatter(x=y_t, y=y_p, mode="markers",
                                 name="Spectra", marker=dict(
                                     size=5, color=BLUE, opacity=0.45)))
        xs = np.linspace(lim[0], lim[1], 10)
        fig.add_trace(go.Scatter(x=xs, y=m.slope * xs + m.intercept, mode="lines",
                                 name=f"Fit (slope {m.slope:.3f})",
                                 line=dict(color=NAVY, width=2)))
        fig.update_layout(height=430, xaxis_title="Reference starch (% w/w)",
                          yaxis_title="Predicted starch (% w/w)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with d2:
        st.markdown("**Choosing the number of latent variables**")
        scan = get_component_scan(method)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=scan.n_components, y=scan.rmsecv,
                                 mode="lines+markers", name="RMSECV",
                                 line=dict(color=NAVY, width=2.4),
                                 marker=dict(size=7)))
        fig.add_vline(x=n_components, line_dash="dash", line_color=GREEN,
                      annotation_text=f"selected: {n_components}")
        fig.update_layout(height=430, xaxis_title="PLS latent variables",
                          yaxis_title="RMSECV (% w/w)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Error falls steeply to 8 components and then flattens. "
                   "Pick the elbow, not the global minimum.")

    st.divider()
    st.markdown("**Residuals by concentration**")
    resid = y_p - y_t
    fig = go.Figure()
    fig.add_hline(y=0, line_color="#555")
    fig.add_hrect(y0=-m.sep, y1=m.sep, fillcolor=NAVY, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=y_t, y=resid, mode="markers",
                             marker=dict(size=5, color=BLUE, opacity=0.5)))
    fig.update_layout(height=300, xaxis_title="Reference starch (% w/w)",
                      yaxis_title="Residual (% w/w)", showlegend=False,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    st.dataframe(pd.DataFrame([{
        "Metric": k,
        "Value": (f"{v:.4f}" if isinstance(v, float) else str(v)),
    } for k, v in m.as_dict().items()]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- 4 curcuminoid

with tabs[3]:
    st.subheader("Curcuminoid assay (Study 1)")
    st.write("Thirty levels of Haridra dry extract blended with sucralose, "
             "0.5 to 15 % w/w curcuminoid, measured September 2025.")

    cur = get_curcuminoid()
    ref = cur.reference_pct_ww.to_numpy()
    prd = cur.nir_predicted_pct_ww.to_numpy()
    cm = compute_metrics(ref, prd)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R²", f"{cm.r2:.4f}")
    k2.metric("Slope", f"{cm.slope:.3f}")
    k3.metric("SEP", f"{cm.sep:.2f} % w/w")
    k4.metric("RPD", f"{cm.rpd:.2f}")

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure()
        xs = np.linspace(0, 16, 10)
        fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="Identity",
                                 line=dict(color="#999", dash="dash")))
        fig.add_trace(go.Scatter(x=xs, y=cm.slope * xs + cm.intercept,
                                 mode="lines", name="Fit", line=dict(color=NAVY, width=2)))
        fig.add_trace(go.Scatter(x=ref, y=prd, mode="markers", name="Levels",
                                 marker=dict(size=9, color=BLUE)))
        fig.update_layout(height=400, xaxis_title="Reference curcuminoid (% w/w)",
                          yaxis_title="NIR predicted (% w/w)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with g2:
        err = np.abs(prd - ref)
        cols = [RED if e > 1.0 else (AMBER if e > 0.6 else GREEN) for e in err]
        fig = go.Figure(go.Bar(x=[f"{v:g}" for v in ref], y=err,
                               marker_color=cols))
        fig.add_hline(y=cm.sep, line_dash="dash", line_color=NAVY,
                      annotation_text=f"SEP {cm.sep:.2f}")
        fig.update_layout(height=400, xaxis_title="Reference level (% w/w)",
                          yaxis_title="Absolute error (% w/w)",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    st.warning(
        f"**Working range.** The absolute error floor is about "
        f"±{cm.sep:.2f} % w/w. Below roughly 3 % w/w that error is a large "
        "fraction of the analyte level, so quantitative reporting should be "
        "restricted to 3–15 % w/w. Below 3 % the method can rank material as "
        "low but should not report a number against a specification."
    )


# ---------------------------------------------------------------- 5 card

with tabs[4]:
    st.subheader("Model card")
    card_path = ROOT / "models" / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
        st.json(card, expanded=True)
        st.download_button("Download model card (JSON)",
                           card_path.read_text(),
                           file_name="model_card.json", mime="application/json")
    else:
        st.info("Run `python src/train.py --components 8` to generate the model card.")

    st.divider()
    st.markdown("### Scope and limitations")
    st.markdown(
        """
Both methods here are **method-development and feasibility studies**, not GMP
analytical method validations. They do not qualify either method for batch
release.

- Reference values are **gravimetric** — mixtures made by weighing — rather
  than anchored to an independent validated assay such as HPLC.
- Both use **binary model blends**, not commercial formulations.
- All work used a **single extract batch** from one supplier. Botanical raw
  material varies substantially between batches, cultivars and seasons.
- **Sample moisture was not controlled**, and the strongest band in these
  spectra belongs to water.
- Accuracy, repeatability, intermediate precision and robustness were **not
  formally assessed**.

Anchoring the calibration to HPLC values on real product, across several
suppliers and batches, is the single largest available improvement.
        """
    )
