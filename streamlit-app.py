import streamlit as st
import base64
import os
import streamlit.components.v1 as components
from analysis import text
from pathlib import Path

# ---



# 1. Get the directory where THIS script is saved
# This works whether you are on Windows, Mac, or Streamlit Cloud
root_dir = Path(__file__).parent.resolve()

# 2. Join it with your data path
base_path = root_dir / "data" / "streamlit"

# # 3. Check if it exists
# if base_path.exists():
#     st.success(f"Path found! Full path is: {base_path}")
#     # List files to be sure
#     st.write("Files found:", os.listdir(base_path))
# else:
#     st.error(f"Path NOT found. Searching at: {base_path}")
#     st.write("Current working directory is:", os.getcwd())

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="CNN Visualization", layout="wide")

# This block handles the centering of the title and the custom scrollbar look
st.markdown("""
    <style>
    /* Center the main title */
    h1 {
        text-align: center;
        margin-bottom: 40px;
    }

    /* Global styling for Pills/Segmented Control active state */
    div[data-selection-mode] button[aria-checked="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("# CNN Visualization Application 🔍")

# --- 2. INPUT SECTION ---
from itertools import product

st.markdown("### Display Matrix")
class_options = ["baseball", "butterfly", "daisy", "cat", "dog"]
action_options = ["noise", "photo"]
heatmap_options = ["photo-backdrop", "black-backdrop"]

left, middle, right = st.columns(3, vertical_alignment="bottom")
selected_classes = left.pills("Class", class_options, selection_mode="multi")
selected_actions = middle.pills("Heatmap", heatmap_options, selection_mode="multi")
selected_heatmaps = right.pills("Maximization", action_options, selection_mode="multi")

def check_pills_state(selected_classes=[], selected_actions=[], selected_heatmaps=[]):
    prefix = selected_classes
    suffix = selected_actions + selected_heatmaps
    # st.write(f"🧐 **Active Classes:** {selected_classes}")
    # st.write(f"🎬 **Active Actions:** {selected_classes}")
    # st.write(f"🔥 **Active Heatmaps:** {selected_classes}")
    if not prefix and not suffix: 
        return []
    else:
        return [ f"{pre}_{suf}" for pre, suf in product(prefix,suffix) ]
        
active_directories = check_pills_state(selected_classes, selected_actions, selected_heatmaps)
st.divider()

# --- 3. HELPER FUNCTIONS (Must be defined BEFORE Section 4) ---
def get_b64_image(file_path):
    """Helper to convert image files to base64 so HTML can display them."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- 4. DATA LOADING & HORIZONTAL SCROLL (Folder-per-Row) ---
# base_path = "data/streamlit/"


# directories_captions = {
#     "butterfly_photo-backdrop": ["Original", "Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Combined"],
#     "butterfly_black-backdrop": ["Original", "Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Combined"],
#     "baseball_photo-backdrop": ["Original", "Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Combined"],
#     "baseball_black-backdrop": ["Original", "Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Combined"],
# }
captions = ["Original", "Block 1", "Block 2", "Block 3", "Block 4", "Block 5", "Combined"]

st.subheader("Neural Network Visualizations")

# CSS for a shared horizontal scroller
css_style = """
<style>
    .shared-scroller {
        display: block;
        overflow-x: auto;
        background-color: #181818;
        border-radius: 15px;
        padding: 15px;
        scrollbar-width: thin;
        scrollbar-color: #ff4b4b #181818;
    }
    .row-container {
        display: flex; 
        gap: 20px;
        margin-bottom: 30px; /* Space between rows */
        width: max-content; /* Prevents rows from wrapping downwards */
    }
    .img-card {
        width: 220px;
        text-align: center;
        flex: 0 0 auto;
    }
    .img-card img {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #444;
        height: 160px;
        object-fit: cover;
    }
    /* Style for the shared scrollbar */
    .shared-scroller::-webkit-scrollbar { height: 10px; }
    .shared-scroller::-webkit-scrollbar-track { background: #181818; border-radius: 10px; }
    .shared-scroller::-webkit-scrollbar-thumb { background: #ff4b4b; border-radius: 10px; }
</style>
"""


def activate_actions():
    # st.write(f"🤝 **Active Combination:** {active_directories}")
    all_rows_html = ""
    found_any = False

    for directory in active_directories:
        path = os.path.join(base_path, directory)
        if not os.path.exists(path):
            continue
        
        file_names = sorted(os.listdir(path))
        row_html = f'<div class="row-container">' 
        
        for img_name, cap in zip(file_names, captions):
            full_path = os.path.join(path, img_name)
            b64 = get_b64_image(full_path) # Function is now defined above!
            
            if b64:
                found_any = True
                row_html += f"""
                <div class="img-card">
                    <img src="data:image/jpeg;base64,{b64}">
                    <p style="margin-top: 5px; font-size: 13px; color: #cfcfcf; font-family: sans-serif; font-weight: bold;">{cap}</p>
                </div>
                """
        
        row_html += '</div>'
        all_rows_html += row_html

    if found_any:
        full_html = f"{css_style}<div class='shared-scroller'>{all_rows_html}</div>"
        num_folders = len(active_directories)
        dynamic_height = max(300, num_folders * 250)
        components.html(full_html, height=dynamic_height)
    else:
        st.info("To display images, select at least one class and either a Maximization or a Heatmap filter.")

activate_actions()

st.markdown("### Analysis")
tabs = zip(st.tabs(list(text.keys())), text.keys(), text.values())


for tab, header, body in tabs:
    with tab:
        st.markdown(body)




