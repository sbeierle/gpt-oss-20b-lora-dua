import gradio as gr
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

CSV_BEFORE = "prompts_before.csv"
CSV_AFTER = "prompts_after.csv"
CSV_COMPARE = "results_compare.csv"

def safe_load_csv(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return f"✅ {path} geladen – {len(df)} Zeilen", df
        except Exception as e:
            return f"❌ Fehler beim Laden von {path}: {e}", None
    else:
        return f"⚠️ {path} nicht gefunden.", None

status_before, df_before = safe_load_csv(CSV_BEFORE)
status_after, df_after = safe_load_csv(CSV_AFTER)
status_compare, df_compare = safe_load_csv(CSV_COMPARE)

def plot_length_differences(df):
    if df is None or len(df) == 0:
        return None
    df["len_base"] = df["base"].astype(str).apply(len)
    df["len_lora"] = df["lora"].astype(str).apply(len)
    fig, ax = plt.subplots(figsize=(8, 4))
    df[["len_base", "len_lora"]].plot(kind="bar", ax=ax)
    plt.title("Antwortlänge Base vs LoRA")
    plt.xlabel("Prompt Index")
    plt.ylabel("Zeichenanzahl")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_heatmap(df):
    if df is None or len(df) == 0:
        return None
    df["len_base"] = df["base"].astype(str).apply(len)
    df["len_lora"] = df["lora"].astype(str).apply(len)
    df["diff"] = df["len_lora"] - df["len_base"]
    heatmap_data = pd.DataFrame({
        "Base": df["len_base"],
        "LoRA": df["len_lora"],
        "Diff": df["diff"]
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(heatmap_data.T, annot=True, fmt="d", cmap="RdYlGn", cbar=True, ax=ax)
    plt.title("Heatmap – Base vs LoRA Unterschiede")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

with gr.Blocks() as demo:
    gr.Markdown("# 📊 LoRA Vergleich – GPT-OSS Before/After")

    with gr.Tab("Before (Base)"):
        gr.Textbox(value=status_before, label="Status", interactive=False)
        gr.Dataframe(value=df_before, interactive=False)

    with gr.Tab("After (LoRA)"):
        gr.Textbox(value=status_after, label="Status", interactive=False)
        gr.Dataframe(value=df_after, interactive=False)

    with gr.Tab("Compare (Base vs LoRA)"):
        gr.Textbox(value=status_compare, label="Status", interactive=False)
        gr.Dataframe(value=df_compare, interactive=False)

    with gr.Tab("Visualisierung"):
        gr.Markdown("### 📈 Antwortlängen (Bar Chart)")
        gr.Image(value=plot_length_differences(df_compare), type="pil")

        gr.Markdown("### 🔥 Unterschiede Heatmap")
        gr.Image(value=plot_heatmap(df_compare), type="pil")

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
