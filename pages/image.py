# Import necessary libraries
import streamlit as st
from pydantic import BaseModel
from typing import List, Dict
from PIL import Image
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import os
import json
from typing import TypedDict
import google.generativeai as genai
from langchain.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY
    # base_url="...",
    # organization="...",
    # other params...
)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Define our state
class GraphState(TypedDict):
    image_path: str  # Path to the image file
    image_description: str
    modality: str  # Medical image modality (X-ray, CT, MRI, etc.)
    selected_scripts: List[str]  # List of scripts to run for the detected modality
    results: Dict

# Gemini image classification
FIND_IMAGE_PROMPT = """
Identify the medical imaging modality (X-ray, CT, MRI, Ultrasound, PET scan, etc.) of the given image.
Only return the modality name.
"""

def classify_image_with_llm(image_path):
    img = Image.open(image_path)  # Convert bytes to PIL image
    response = model.generate_content([FIND_IMAGE_PROMPT, img])  # Pass PIL image
    return response.text.strip() if response.text else "Unknown"

def classify_medical_image(image_path):
    # **Step 1: Filename-based classification**
    filename = os.path.basename(image_path).lower()

    if "xray" in filename:
        return "X-ray"
    elif "ct" in filename:
        return "CT"

    # **Step 2: Fallback to Gemini if filename is not useful**
    print("Filename did not indicate modality. Using AI model...")
    return classify_image_with_llm(image_path)

# Function to set API key for OpenAI
def set_openai_api_key():
    api_key = OPENAI_API_KEY
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        # Interactive input if API key is not set
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

# X-ray agent tools
@tool
def detect_bone_fracture(image_path: str) -> Dict:
    """
    Analyzes an X-ray image to detect bone fractures.

    Args:
        image_path: Path to the X-ray image file

    Returns:
        Dictionary containing fracture information including count, locations, and confidence level
    """
    # This is a dummy implementation - in a real scenario, this would call a specialized ML model
    st.write(f"Running bone fracture detection on {image_path}...")
    # Simulate processing time
    import time
    time.sleep(1)

    # Return dummy results
    return {
        "fractures_detected": 2,
        "locations": ["wrist", "finger"],
        "confidence": 0.89
    }

@tool
def analyze_lung_condition(image_path: str) -> Dict:
    """
    Analyzes an X-ray image to assess lung condition.

    Args:
        image_path: Path to the X-ray image file

    Returns:
        Dictionary containing lung condition assessment, severity, and confidence level
    """
    # This is a dummy implementation
    st.write(f"Running lung condition analysis on {image_path}...")
    # Simulate processing time
    import time
    time.sleep(1)

    # Return dummy results
    return {
        "condition": "pneumonia",
        "severity": "moderate",
        "confidence": 0.92
    }

# CT agent tools
@tool
def detect_tumor(image_path: str) -> Dict:
    """
    Analyzes a CT scan to detect tumors.

    Args:
        image_path: Path to the CT image file

    Returns:
        Dictionary containing tumor detection results, including count, size, and location
    """
    # This is a dummy implementation
    st.write(f"Running tumor detection on {image_path}...")
    # Simulate processing time
    import time
    time.sleep(1)

    # Return dummy results
    return {
        "tumors_detected": 1,
        "size_mm": 8,
        "location": "liver"
    }

@tool
def segment_organs(image_path: str) -> Dict:
    """
    Analyzes a CT scan to segment different organs.

    Args:
        image_path: Path to the CT image file

    Returns:
        Dictionary containing segmented organs and their volumes
    """
    # This is a dummy implementation
    st.write(f"Running organ segmentation on {image_path}...")
    # Simulate processing time
    import time
    time.sleep(1)

    # Return dummy results
    return {
        "organs_segmented": ["liver", "spleen", "kidneys"],
        "volumes_cc": [1500, 200, 300]
    }

# Mapping from modality to script functions
MODALITY_SCRIPT_MAP = {
    "X-ray": {
        "bone_fracture_detection": detect_bone_fracture,
        "lung_condition_analysis": analyze_lung_condition
    },
    "CT": {
        "tumor_detection": detect_tumor,
        "organ_segmentation": segment_organs
    }
}

# Node functions
def identify_modality(state: GraphState) -> GraphState:
    """Identify the medical image modality."""
    image_path = state["image_path"]
    modality = classify_medical_image(image_path)

    st.write(f"Identified modality: {modality}", key="my_custom_text")

    return {**state, "modality": modality}

def select_scripts(state: GraphState) -> GraphState:
    """Select which scripts to run based on the modality."""
    modality = state["modality"]

    # Get the script map for the identified modality
    if modality in MODALITY_SCRIPT_MAP:
        script_map = MODALITY_SCRIPT_MAP[modality]
        scripts_to_run = list(script_map.keys())
    else:
        # Default fallback if modality not recognized
        st.write(f"Modality '{modality}' not recognized. Using X-ray scripts as fallback.")
        script_map = MODALITY_SCRIPT_MAP["X-ray"]
        scripts_to_run = list(script_map.keys())

    return {**state, "selected_scripts": scripts_to_run}

def run_modality_scripts(state: GraphState) -> GraphState:
    """Run all scripts for the identified modality."""
    modality = state["modality"]
    image_path = state["image_path"]
    scripts = state["selected_scripts"]

    # Get the script map for this modality
    if modality in MODALITY_SCRIPT_MAP:
        script_map = MODALITY_SCRIPT_MAP[modality]
    else:
        # Fallback
        script_map = MODALITY_SCRIPT_MAP["X-ray"]

    # Run each script and collect results
    results = {}
    for script_name in scripts:
        if script_name in script_map:
            script_function = script_map[script_name]
            print(f"Running {script_name}...")
            script_result = script_function(image_path)
            results[script_name] = script_result

    return {**state, "results": results}

def summarize_results(state: GraphState) -> GraphState:
    """Generate a summary of all results using GPT-4o Mini."""
    # Only use GPT for summarization if OpenAI API key is available
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Create summary prompt with all the results
        modality = state["modality"]
        results_str = json.dumps(state["results"], indent=2)

        prompt = f"""
        Summarize the following medical imaging results for a {modality} scan:

        {results_str}

        Provide a concise summary highlighting the key findings and their clinical significance.
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content
        st.write(f"Summary: \n{summary}")
        # Add summary to state
        return {**state, "summary": summary}
    except Exception as e:
        print(f"Error generating summary: {e}")
        return {**state, "summary": "Summary generation failed due to an error with the language model."}

def display_results(state: GraphState) -> GraphState:
    """Display the image, modality, and results to the user."""
    # Display original image
    try:
        img = Image.open(state["image_path"])
    except Exception as e:
        st.write(f"Error image not found: {e}")

    # Display modality
    st.write(f"\nIdentified Modality: {state['modality']}")

    # Display scripts run
    st.write(f"\nScripts executed for {state['modality']} analysis:")
    for script in state["selected_scripts"]:
        st.write(f"- {script}")

    # Display detailed results
    st.write("\nDetailed Results:")
    for script_name, result in state["results"].items():
        st.write(f"\n{script_name}:")
        for key, value in result.items():
            st.write(f"  {key}: {value}")

    # Display summary if available
    if "summary" in state and state["summary"]:
        st.write("\nSummary:")
        st.write(state["summary"])

    return state

# Create the graph
def create_workflow():
    # Set up OpenAI API key for summary generation (optional)
    try:
        set_openai_api_key()
    except:
        print("OpenAI API key not set. Summary generation will be skipped.")

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("identify_modality", identify_modality)
    workflow.add_node("select_scripts", select_scripts)
    workflow.add_node("run_modality_scripts", run_modality_scripts)
    workflow.add_node("summarize_results", summarize_results)
    workflow.add_node("display_results", display_results)

    # Add edges
    workflow.add_edge("identify_modality", "select_scripts")
    workflow.add_edge("select_scripts", "run_modality_scripts")
    workflow.add_edge("run_modality_scripts", "summarize_results")
    workflow.add_edge("summarize_results", "display_results")
    workflow.add_edge("display_results", END)

    # Set entry point
    workflow.set_entry_point("identify_modality")

    return workflow.compile()

# Function to process a medical image
def process_medical_image(image_path, description=""):
    # Create workflow
    workflow = create_workflow()

    # Initialize state
    initial_state = {
        "image_path": image_path,
        "image_description": description,
        "modality": "",
        "selected_scripts": [],
        "results": {}
    }

    # Run the workflow
    final_state = workflow.invoke(initial_state)

    return final_state

# Streamlit app logic
def main():
    st.set_page_config(page_title="Medical Image Analysis Agent", page_icon="📈")
    st.sidebar.header("Medical Image Analysis Agent")
    st.title("Medical Image Analysis Agent")

    uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])

    st.markdown("""
    <style>
    .stkey-my_custom_text p {
        color: green;
        font-family: 'Arial', sans-serif;
        background-color: #e6f7ff;
        padding: 15px;
        border-left: 5px solid #1890ff;
    }
    </style>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_image_path = f"temp_{uploaded_file.name}"
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)
        
        # Process the medical image
        result = process_medical_image(temp_image_path)

if __name__ == "__main__":
    main()


