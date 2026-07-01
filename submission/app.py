import gradio as gr
import pandas as pd
import os
import json
from rules import evaluate_hard_filters
from main import check_company_timeline_honeypot, run_rrf_combination

def run_pipeline(sample_input):
    # 1. Parse the pasted JSON data
    if not sample_input or not sample_input.strip():
        return pd.DataFrame({"Error": ["Please paste some JSON data first!"]}), None
    
    try:
        candidates = json.loads(sample_input)
    except Exception as e:
        return pd.DataFrame({"Error": [f"Invalid JSON: {e}"]}), None
        
    if not isinstance(candidates, list):
        candidates = [candidates] # handle single object pastes
        
    # 2. Run Layer 0 (Filtration Gate)
    passing_candidates = []
    for c in candidates:
        if not evaluate_hard_filters(c).passed:
            continue
        if not check_company_timeline_honeypot(c):
            continue
        passing_candidates.append(c)
        
    if not passing_candidates:
        return pd.DataFrame({"Error": ["All pasted candidates were dropped by the Hard Filters!"]}), None
        
    # 3 & 4 & 5. Run Scoring, RRF Matrix, and Narrative Synthesis
    df = run_rrf_combination(passing_candidates, k=60)
    
    # 6. Save live result to a temporary CSV for the download button
    out_path = "live_sandbox_output.csv"
    df.to_csv(out_path, index=False)
    
    return df, out_path

# Define a sleek, premium Lavender & Black theme
custom_theme = gr.themes.Soft(
    primary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    # Dark backgrounds
    body_background_fill="#000000",
    body_background_fill_dark="#000000",
    background_fill_primary="#0a0a0a",
    background_fill_primary_dark="#0a0a0a",
    background_fill_secondary="#111111",
    background_fill_secondary_dark="#111111",
    block_background_fill="#0a0a0a",
    block_background_fill_dark="#0a0a0a",
    input_background_fill="#111111",
    input_background_fill_dark="#111111",
    
    # Text colors (Forced to white/light gray so they don't vanish!)
    body_text_color="#f3f4f6",
    body_text_color_dark="#f3f4f6",
    block_title_text_color="#f3f4f6",
    block_title_text_color_dark="#f3f4f6",
    block_label_text_color="#c4b5fd", # Lavender labels
    block_label_text_color_dark="#c4b5fd",
    input_text_color="#ffffff",
    input_text_color_dark="#ffffff",
    
    # Lavender Accents
    color_accent_soft="#8b5cf6",
    color_accent_soft_dark="#8b5cf6",
    border_color_primary="#2d2d2d",
    border_color_primary_dark="#2d2d2d",
    border_color_accent="#8b5cf6",
    border_color_accent_dark="#8b5cf6",
    
    # Primary Buttons
    button_primary_background_fill="#8b5cf6",
    button_primary_background_fill_dark="#8b5cf6",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_background_fill_hover="#7c3aed",
    button_primary_background_fill_hover_dark="#7c3aed",
    
    # Secondary Buttons
    button_secondary_background_fill="#1f2937",
    button_secondary_background_fill_dark="#1f2937",
    button_secondary_text_color="#ffffff",
    button_secondary_text_color_dark="#ffffff",
)

demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(lines=10, label="Paste Candidate JSON Data Stream", placeholder="[ { 'candidate_id': ... } ]"),
    outputs=[
        gr.Dataframe(label="Top Shortlisted Output (Ranked via RRF Matrix)"),
        gr.File(label="Download Submission CSV")
    ],
    title="Recruitment Information Retrieval Pipeline Sandbox",
    description="Live demonstration of our deterministic layered architecture filtering, ranking, and generating unique justifications.",
    theme=custom_theme
)

demo.launch()