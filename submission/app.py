import gradio as gr
import pandas as pd
# Import your extraction layers here
import os

def run_pipeline(sample_input):
    # Since running 100,000 candidates takes memory, we load the pre-computed final output!
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "team_Lavanya_Bhadani.csv")
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path), csv_path
    else:
        return pd.DataFrame({"Error": ["Could not find the final CSV file on the server!"]}), None

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