---
title: A Walk through Jezero Crater
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.13"
app_file: app/app.py
pinned: false
short_description: Hazard-aware walking routes on Mars, with or without a DTM
---

# A Walk through Jezero Crater

NASA Space Apps 2026, "Interplanetary Survival Guide: Martian Map".

Plan a walking route in Jezero Crater that avoids steep slopes. DTM mode uses true slope from the HiRISE elevation
model. DTM-free mode uses a U-Net that predicts slope hazard from a single HiRISE image, for places with no
elevation model. Every route is scored on the true terrain. The "How it works" tab has the method and the metrics.

The app runs on CPU and only does A* routing on precomputed layers (`app/data/layers.npz`).
