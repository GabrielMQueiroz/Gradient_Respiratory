import streamlit as st
import os
import glob

st.title("Spatio-Temporal Structure Tensor Research Platform")

st.markdown("""
Welcome to the interactive research workbench for **Adaptive Spatio-Temporal Gradient Difference Tensors** (`sttensor`).

This platform provides interactive evaluation of the methods demonstrated in the research videos, alongside advanced hypothesis testing for computer vision, structural health monitoring, and physiological motion deconvolution.
""")

st.subheader("Core Mathematical Foundations")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("### Gradient Difference Operator ($D$)")
        st.latex(r"\mathbf{D}(x, y, t) = \nabla I_t - \nabla I_{t-1} = \nabla\left(\frac{\partial I}{\partial t}\right)")
        st.markdown("""
        By Clairaut-Schwarz theorem on mixed partial derivatives, differentiating the spatial gradient over time isolates moving boundary contours while neutralizing static background texture gradients.
        """)

with col2:
    with st.container(border=True):
        st.markdown("### Adaptive Scale Smoothing ($\sigma$)")
        st.latex(r"\sigma(x, y) = \frac{\sigma_{\max}}{1 + k \cdot \text{Tr}(\mathbf{T}(x, y))}")
        st.markdown("""
        - $\text{Tr}(\mathbf{T}) \to 0$ (Background): $\sigma \to \sigma_{\max}$ (strongly eliminates sensor noise).
        - $\text{Tr}(\mathbf{T}) \gg 0$ (Moving Edge): $\sigma \to 0$ (retains sharp edge localization, preventing smearing).
        """)

st.subheader("Repository Demonstration Videos")

videos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "videos"))
video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

if video_files:
    selected_vid = st.selectbox(
        "Select Video to Preview",
        options=[os.path.basename(v) for v in video_files]
    )
    st.video(os.path.join(videos_dir, selected_vid))
else:
    st.info("No videos found in videos/ directory.")
