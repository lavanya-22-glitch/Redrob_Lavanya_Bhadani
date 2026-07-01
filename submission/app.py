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
custom_theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    # Pure black backgrounds
    body_background_fill="#000000",
    body_background_fill_dark="#000000",
    background_fill_primary="#121212",
    background_fill_primary_dark="#121212",
    background_fill_secondary="#000000",
    background_fill_secondary_dark="#000000",
    
    # Text colors
    body_text_color="#E0E0E0",
    body_text_color_dark="#E0E0E0",
    
    # Borders
    border_color_primary="#2D2D2D",
    border_color_primary_dark="#2D2D2D",
    border_color_accent="#B392F0",      # Lavender accent
    color_accent_soft="#B392F0",        # Lavender accent
    
    # Primary Buttons (Lavender)
    button_primary_background_fill="#B392F0",
    button_primary_background_fill_dark="#B392F0",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    button_primary_background_fill_hover="#9B72E6",
    button_primary_background_fill_hover_dark="#9B72E6",
    
    # Secondary Buttons
    button_secondary_background_fill="#1E1E1E",
    button_secondary_background_fill_dark="#1E1E1E",
    button_secondary_text_color="#E0E0E0",
    button_secondary_text_color_dark="#E0E0E0",
    
    # Inputs & Blocks
    input_background_fill="#121212",
    input_background_fill_dark="#121212",
    block_background_fill="#121212",
    block_background_fill_dark="#121212",
)

demo = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(lines=5, label="Paste Candidate JSON Data Stream"),
    outputs=[
        gr.Dataframe(label="Top Shortlisted Output (Ranked via RRF Matrix)"),
        gr.File(label="Download Submission CSV")
    ],
    title="Recruitment Information Retrieval Pipeline Sandbox",
    description="Live demonstration of our deterministic layered architecture filtering, ranking, and generating unique justifications.",
    theme=custom_theme
)

demo.launch()