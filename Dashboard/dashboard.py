import os
import datetime
from typing import List, Dict, Any

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from huggingface_hub import InferenceClient
from matplotlib.patches import Wedge

from models import GlobalModel
# Make sure this import matches where you keep the big prompt dict
from evaluation import STRATEGY4_PROMPTS   # <-- adjust this filename if needed

########################################
# 1. DLCAF MODEL + PROMPT ENCODING
########################################

EVAL_CONFIG = {
    "model_path": "./best_model.pt",
    "model_name": "openai/clip-vit-base-patch16",
    "unfreeze_last_n_blocks": 1,
}

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dlcaf_model_and_text_features() -> Dict[str, Any]:
    """
    Load your DLCAF / GlobalModel and pre-encode text features
    for all classes using STRATEGY4_PROMPTS.
    """
    model_path = EVAL_CONFIG["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = GlobalModel(
        model_name=EVAL_CONFIG["model_name"],
        dropout_rate=0.1,
        unfreeze_last_n_blocks=EVAL_CONFIG["unfreeze_last_n_blocks"]
    )

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Class names from your corrected prompts
    class_names = sorted(STRATEGY4_PROMPTS.keys())

    # Find healthy class (case-insensitive)
    healthy_idx = None
    for i, name in enumerate(class_names):
        if "healthy" in name.lower():
            healthy_idx = i
            break
    if healthy_idx is None:
        raise ValueError("Could not find a 'Healthy' class in STRATEGY4_PROMPTS keys.")
    healthy_label = class_names[healthy_idx]

    # Pre-encode text prompts
    text_features = []
    for cls in class_names:
        prompts = STRATEGY4_PROMPTS[cls]
        text_inputs = model.tokenizer(
            prompts, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            cls_text_feats = model.clip_model.get_text_features(**text_inputs)
            cls_text_feats = F.normalize(
                cls_text_feats.mean(dim=0, keepdim=True), p=2, dim=-1
            )
        text_features.append(cls_text_feats)

    text_features_tensor = torch.cat(text_features, dim=0)  # [num_classes, dim]

    return {
        "model": model,
        "class_names": class_names,
        "text_features": text_features_tensor,
        "healthy_idx": healthy_idx,
        "healthy_label": healthy_label,
    }


def dlcaf_predict_images(state: Dict[str, Any], image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    DLCAF-style inference for a list of leaf images (each image = 1 leaf).
    Returns per-image top-1 & top-3.
    
    Extra rule:
      - If 'Healthy' appears in the top-3 classes for a leaf,
        we force the final label to be 'Healthy' (bias toward healthy).
    """
    model = state["model"]
    class_names = state["class_names"]
    text_features = state["text_features"]  # [C, D]
    healthy_label = state["healthy_label"]

    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(image_transform(img))

    images_tensor = torch.stack(images).to(DEVICE)

    with torch.no_grad():
        img_feats = model.clip_model.get_image_features(images_tensor)
        img_feats = F.normalize(img_feats, p=2, dim=-1)
        sims = img_feats @ text_features.T  # [B, C]

        top1_idx = sims.argmax(dim=1)              # [B]
        top1_vals = sims.max(dim=1).values         # [B]
        top3_vals, top3_idx = sims.topk(3, dim=1)  # [B, 3]

    results = []
    for i in range(len(image_paths)):
        t1 = int(top1_idx[i].item())
        raw_top1_label = class_names[t1]
        raw_top1_prob = float(top1_vals[i].item())

        top3_ids = top3_idx[i].tolist()
        top3_labels = [class_names[j] for j in top3_ids]
        top3_probs = [float(v) for v in top3_vals[i].tolist()]

        # ---- Healthy-bias rule here ----
        final_label = raw_top1_label
        final_prob = raw_top1_prob

        if healthy_label in top3_labels:
            # Option 1 (simple): always choose Healthy if it appears in top-3
            healthy_pos = top3_labels.index(healthy_label)
            final_label = healthy_label
            final_prob = top3_probs[healthy_pos]

            # If you want a margin-based rule instead, you could do:
            # if top3_probs[healthy_pos] >= raw_top1_prob - 0.05:
            #     final_label = healthy_label
            #     final_prob = top3_probs[healthy_pos]

        results.append({
            "top1_label": final_label,      # now biased toward Healthy
            "top1_prob": final_prob,
            "top3_labels": top3_labels,
            "top3_probs": top3_probs,
            "raw_top1_label": raw_top1_label,   # optional: keep original
            "raw_top1_prob": raw_top1_prob,     # optional: keep original
        })
    return results


########################################
# 2. METRICS & PROMPT CREATION
########################################

def compute_spatial_metrics_from_leaves(
    leaf_results: List[Dict[str, Any]],
    num_rows: int,
    plants_per_row: int,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Metrics are leaf-based (each image is one leaf).
    Plant health matrix is plant-based:
      plant is healthy only if *all three* leaves for that plant are healthy.
    """
    total_plants = num_rows * plants_per_row
    total_leaves = len(leaf_results)
    expected_leaves = total_plants * 3

    assert total_leaves == expected_leaves, "leaf_results length mismatch."

    class_names = state["class_names"]
    healthy_label = state["healthy_label"]

    # Per-row leaf counts
    N_r = np.zeros(num_rows, dtype=int)
    N_r_inf = np.zeros(num_rows, dtype=int)

    # Per-disease leaf counts (infected leaves only)
    disease_to_idx = {name: i for i, name in enumerate(class_names)}
    diseased_counts = np.zeros(len(class_names), dtype=int)

    # Plant health: True = healthy, False = infected
    plant_health = np.ones((num_rows, plants_per_row), dtype=bool)

    idx = 0
    for r in range(num_rows):
        for c in range(plants_per_row):
            plant_leaf_labels = []
            for _ in range(3):
                res = leaf_results[idx]
                lbl = res["top1_label"]
                plant_leaf_labels.append(lbl)

                N_r[r] += 1
                if lbl != healthy_label:
                    N_r_inf[r] += 1
                    diseased_counts[disease_to_idx[lbl]] += 1
                idx += 1

            # Plant infected if ANY leaf is infected
            if any(lbl != healthy_label for lbl in plant_leaf_labels):
                plant_health[r, c] = False

    # Row-level infection percentage P_r (leaf-based)
    P_r = (N_r_inf / np.maximum(N_r, 1)) * 100.0

    # Greenhouse infection percentage P_GH (leaf-based)
    P_GH = (N_r_inf.sum() / max(N_r.sum(), 1)) * 100.0

    # Greenhouse disease composition (leaf-based)
    P_Dk_GH = (diseased_counts / max(N_r.sum(), 1)) * 100.0

    P_rows_dict = {f"Row {r+1}": float(P_r[r]) for r in range(num_rows)}
    P_diseases_GH = {}
    for i, name in enumerate(class_names):
        if name != healthy_label:
            P_diseases_GH[name] = float(P_Dk_GH[i])

    S_t = [float(P_GH)] + [float(v) for v in P_Dk_GH] + [float(v) for v in P_r]

    return {
        "N_r": N_r,
        "N_r_inf": N_r_inf,
        "P_GH": float(P_GH),
        "P_rows": P_rows_dict,
        "P_diseases_GH": P_diseases_GH,
        "plant_health": plant_health,
        "S_t": S_t,
    }


def build_observation_prompt(date_str: str, metrics: Dict[str, Any]) -> str:
    """
    Build a structured text snippet that we’ll feed into Llama later.
    Only show diseases with non-zero percentages.
    """
    lines = []
    lines.append(f"Observation date: {date_str}")
    lines.append(f"Greenhouse infection percentage (leaf-based): {metrics['P_GH']:.2f}%")
    lines.append("Row-wise infection percentages (leaf-based):")
    for row_name, val in metrics["P_rows"].items():
        lines.append(f"  - {row_name}: {val:.2f}%")

    # Only non-zero diseases
    nonzero_diseases = {
        d: v for d, v in metrics["P_diseases_GH"].items() if v > 0.0
    }

    if nonzero_diseases:
        lines.append("Greenhouse disease composition (leaf-based):")
        for disease, val in nonzero_diseases.items():
            lines.append(f"  - {disease}: {val:.2f}%")
    else:
        lines.append("Greenhouse disease composition (leaf-based):")
        lines.append("  - No diseases detected (all percentages are 0%).")

    return "\n".join(lines)


########################################
# 3. LLM CLIENT (HF InferenceClient, chat.completions)
########################################

HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    print("WARNING: HF_TOKEN environment variable not set. LLM calls will fail until it is set.")

llm_client = InferenceClient(api_key=HF_TOKEN)

LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"



def call_llm_on_observations(obs1: Dict[str, Any], obs2: Dict[str, Any]) -> str:
    """
    Calls the Llama 3.1 model via HF InferenceClient chat.completions.create,
    using a single user message with both observations.
    """
    if HF_TOKEN is None:
        return "HF_TOKEN environment variable is not set. Please set it before running the LLM advisory."

    user_prompt = (
        "You are an expert agronomist specializing in tomato greenhouse disease management.\n\n"
        "Analyze the following spatiotemporal disease summaries for two inspections.\n\n"
        "=== Earlier inspection ===\n"
        f"{obs1['prompt']}\n\n"
        "=== Later inspection ===\n"
        f"{obs2['prompt']}\n\n"
        "Please provide:\n"
        "a) Spatio-temporal disease summary\n"
        "b) Risk assessment\n"
        "c) Action plan\n"
    )

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    # HF chat API: message content is here
    return completion.choices[0].message["content"]


########################################
# 4. GRADIO CALLBACKS
########################################
def process_observation(
    num_rows: float,
    plants_per_row: float,
    observation_date: str,
    images,
    dlcaf_state,
    observation_history,
):
    try:
        # 1) Basic checks
        num_rows = int(num_rows)
        plants_per_row = int(plants_per_row)

        if num_rows <= 0 or plants_per_row <= 0:
            return (
                "Rows and plants per row must be > 0.",
                None,
                None,
                None,
                observation_history,
                "",          # history_text
            )

        total_plants = num_rows * plants_per_row
        expected_images = total_plants * 3

        if images is None or len(images) != expected_images:
            return (
                f"Expected {expected_images} images (3 per plant), received {len(images) if images else 0}.",
                None,
                None,
                None,
                observation_history,
                "",          # history_text
            )

        # 2) Load DLCAF model + text features lazily
        if dlcaf_state is None:
            dlcaf_state = load_dlcaf_model_and_text_features()

        # For type="filepath", `images` is already a list of path strings
        image_paths = images

        # 3) DLCAF predictions for each leaf image
        leaf_results = dlcaf_predict_images(dlcaf_state, image_paths)

        # 4) Compute spatial metrics & plant health matrix
        metrics = compute_spatial_metrics_from_leaves(
            leaf_results, num_rows, plants_per_row, dlcaf_state
        )

        # 5) Use today's date if none provided
        if not observation_date:
            observation_date = datetime.date.today().isoformat()

        # 6) Build and save prompt for this observation
        obs_prompt = build_observation_prompt(observation_date, metrics)
        obs_record = {
            "date": observation_date,
            "prompt": obs_prompt,
            "metrics": metrics,
        }
        observation_history = observation_history + [obs_record]

        # 7) History text (indices + dates + P_GH)
        history_lines = []
        for idx, obs in enumerate(observation_history):
            m = obs["metrics"]
            history_lines.append(
                f"Index {idx}: date={obs['date']}, P_GH={m['P_GH']:.2f}%"
            )
        history_text = "\n".join(history_lines)

        # 8) Plant-level table
        plant_table = []
        idx_leaf = 0
        for r in range(num_rows):
            for c in range(plants_per_row):
                leaf_labels = []
                for _ in range(3):
                    leaf_labels.append(leaf_results[idx_leaf]["top1_label"])
                    idx_leaf += 1
                infected = any(
                    lbl != dlcaf_state["healthy_label"] for lbl in leaf_labels
                )
                plant_table.append([
                    r + 1,
                    c + 1,
                    ", ".join(leaf_labels),
                    infected,
                ])

        # 9) Text summary
        matrix_str = "\n".join(
            " ".join(
                "H" if metrics["plant_health"][r, c] else "X"
                for c in range(plants_per_row)
            )
            for r in range(num_rows)
        )
        summary_text = (
            f"Observation {observation_date}\n\n"
            f"Plant health matrix (H=healthy plant, X=infected plant):\n"
            f"{matrix_str}\n\n"
            f"P_GH (leaf-based): {metrics['P_GH']:.2f}%\n"
            f"Row-wise infection (leaf-based): {metrics['P_rows']}\n"
            f"Disease composition (leaf-based): {metrics['P_diseases_GH']}\n"
        )

        # 10) Wedge-style plant heatmap
        #     Each plant is a circle split into 3 wedges (one per leaf).
        rows, cols = num_rows, plants_per_row
        fig, ax = plt.subplots(figsize=(cols * 0.6, rows * 0.6))

        for r in range(rows):
            for c in range(cols):
                # index of the first leaf of this plant in leaf_results
                base_idx = (r * cols + c) * 3

                # determine colors for the 3 leaves of this plant
                leaf_colors = []
                for k in range(3):
                    lbl = leaf_results[base_idx + k]["top1_label"]
                    if lbl == dlcaf_state["healthy_label"]:
                        leaf_colors.append("green")  # healthy leaf
                    else:
                        leaf_colors.append("red")    # infected leaf

                # draw 3 wedges of 120° each around the circle
                for k, color in enumerate(leaf_colors):
                    start_angle = 90 + k * 120
                    end_angle = start_angle + 120
                    wedge = Wedge(
                        (c, r),                     # centre at plant position
                        0.4,                        # radius
                        start_angle,
                        end_angle,
                        facecolor=color,
                        edgecolor="black",          # thin black lines between wedges
                        linewidth=0.6,
                    )
                    ax.add_patch(wedge)

        # outer black rectangle around the whole grid
        border = plt.Rectangle(
            (-0.5, -0.5),
            cols,
            rows,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )
        ax.add_patch(border)

        # axes and layout
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)   # flip y so Row 1 is at the top
        ax.set_aspect("equal")

        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xlabel("Plant index in row")
        ax.set_ylabel("Row index")
        ax.set_title("Plant-level Health")

        ax.grid(False)
        plt.tight_layout()

        # ✅ Return 6 values (matches your Gradio outputs)
        return summary_text, plant_table, obs_prompt, fig, observation_history, history_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"Error while processing observation: {e}",
            None,
            None,
            None,
            observation_history,
            "",      # history_text
        )



def run_llm_advisory(observation_history, earlier_idx: float, later_idx: float):
    if observation_history is None or len(observation_history) < 2:
        return "Need at least 2 observations."

    earlier_idx = int(earlier_idx)
    later_idx = int(later_idx)

    if not (0 <= earlier_idx < len(observation_history)) or not (0 <= later_idx < len(observation_history)):
        return "Invalid indices."

    obs1 = observation_history[earlier_idx]
    obs2 = observation_history[later_idx]

    return call_llm_on_observations(obs1, obs2)


########################################
# 5. BUILD GRADIO DASHBOARD
########################################

with gr.Blocks() as demo:
    gr.Markdown("# DLCAF–LLM Greenhouse Disease Dashboard")

    dlcaf_state = gr.State(value=None)
    observation_history = gr.State(value=[])
    history_box = gr.Textbox(
    label="Saved inspections (index → date, P_GH)",
    lines=5,
    interactive=False,
)


    with gr.Tab("1. Upload & Process Inspection"):
        with gr.Row():
            num_rows = gr.Number(label="Number of rows", value=4, precision=0)
            plants_per_row = gr.Number(label="Plants per row", value=4, precision=0)
            observation_date = gr.Textbox(label="Observation date (YYYY-MM-DD, optional)", value="")

        images = gr.File(
            label="Upload leaf images (3 per plant, ordered plant-by-plant, row-by-row)",
            file_count="multiple",
            type="filepath",
        )
        run_btn = gr.Button("Run DLCAF + Compute Metrics")

        summary_box = gr.Textbox(label="Inspection Summary", lines=10)
        plant_df = gr.Dataframe(
            headers=["row", "col", "leaf_labels", "infected"],
            label="Plant-level summary",
            interactive=False,
        )
        obs_prompt_out = gr.Textbox(
            label="Generated Prompt for this Observation (saved for LLM)",
            lines=8
        )
        heatmap_plot = gr.Plot(label="Plant Health Heatmap")

        run_btn.click(
    fn=process_observation,
    inputs=[num_rows, plants_per_row, observation_date, images, dlcaf_state, observation_history],
    outputs=[summary_box, plant_df, obs_prompt_out, heatmap_plot, observation_history, history_box],
)


    with gr.Tab("2. LLM Spatiotemporal Advisory"):
        gr.Markdown(
            "Select two inspection cycles by index (0 = first, 1 = second, etc.) "
            "and let Llama 3.1 analyze spatio-temporal patterns."
        )
        earlier_idx = gr.Number(label="Earlier observation index", value=0, precision=0)
        later_idx = gr.Number(label="Later observation index", value=1, precision=0)
        llm_btn = gr.Button("Run LLM Advisory")
        llm_out = gr.Textbox(label="LLM Output", lines=20)

        llm_btn.click(
            fn=run_llm_advisory,
            inputs=[observation_history, earlier_idx, later_idx],
            outputs=[llm_out],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


