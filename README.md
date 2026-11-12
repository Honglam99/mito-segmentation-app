# Label-free mitochondria segmentation in holotomography

Streamlit app: upload refractive-index (RI) images from holotomography and get binary mitochondria
masks from a 2D nnU-Net trained on fluorescence-guided labels of HUVECs.

- **Accurate mode** (default): ensemble of the 3 leave-one-dish-out fold models with mirror test-time
  augmentation, identical to the paper. About 5-10 s per 512x512 image on CPU.
- **Fast mode**: one fold model without mirroring, under 1 s per image.

Live app: [APP_LINK]. Model weights: [MODEL_REPO_LINK]. Code and data: [CODE_LINK].

**Scope.** Trained on HUVEC RI images from one holotomographic microscope (held-out test Dice
0.850 ± 0.064). On other microscopes or cell types accuracy is lower; check results visually. Not for
fluorescence, brightfield or phase-contrast images. Uploaded images are processed in memory and not
stored.

## Run locally

```bash
pip install -r requirements.txt   # on Windows/macOS replace the torch line with: pip install torch==2.5.1
streamlit run app.py
```

The app downloads the weights from the Hugging Face repo named in the `MODEL_REPO` environment
variable (or a local folder path). For a private repo also set `HF_TOKEN`.

## Citation

> **Label-Free Segmentation of Endothelial Mitochondria in Live-Cell Holotomography via
> Fluorescence-Guided Annotation and nnU-Net.** [Authors]. *Cell Reports Methods* (2026). [DOI]
