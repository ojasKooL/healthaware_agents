import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import PyPDF2
import google.generativeai as genai
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Page configuration
st.set_page_config(page_title="Medical Report Analyzer", page_icon="📈")
st.sidebar.header("Medical Image Analysis Agent")
st.title("Medical Report Analysis System")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'merged_lab_results' not in st.session_state:
    st.session_state.merged_lab_results = None
if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None

# Main application
tab1, tab2, tab3 = st.tabs(["Upload & Process", "Lab Results", "Diagnosis"])

# Medical report extraction prompt
MEDICAL_PROMPT = """
GIVE PLAINTEXT OUTPUT ONLY
Analyze this medical report and extract structured data following these rules:
1. Output Format (JSON):
{
  "patient_info": {
    "name": "str | null",
    "age": "int | null",
    "gender": "str | null",
    "patient_id": "str | null"
  },
  "clinical_findings": {
    "symptoms": ["list of symptom strings"],
    "diagnosis": ["list of diagnosis strings"],
    "procedures": ["list of procedures performed"]
  },
  "medications": [{
    "name": "str",
    "dose": "str",
    "frequency": "str",
    "duration": "str"
  }],
  "lab_results": [{
    "test_name": "str",
    "value": "float",
    "unit": "str",
    "reference_range": "str",
    "date":"date"
  }],
  "report_metadata": {
    "report_date": "YYYY-MM-DD",
    "issuing_hospital": "str",
    "doctor_name": "str"
  }
}
2. Extraction Rules:
- Use SNOMED CT codes where applicable
- Convert all dates to ISO 8601 format
- Normalize lab values to SI units
- Handle missing fields as null
- Preserve decimal precision for numerical values
- Flag abnormal lab values with '[ABNORMAL]' prefix
3. Include confidence scores (0-1) for each extracted field
"""

# Lab results merging prompt
LAB_MERGE_PROMPT = """
GIVE PLAINTEXT OUTPUT ONLY
Merge test results based on the common key value pair and by grouping values under their respective date. Ensure the output retains the unit and reference_range while restructuring the data to consolidate values by date.
Instructions:
Group test results by test_name: All entries with the same test name should be merged into a single dictionary.
Organize values by date: If multiple entries exist for the same test on a particular date, store all value entries as a list under that date.
Retain metadata: The unit and reference_range should remain unchanged.
Ensure structured output: Convert the list of test results into a single dictionary with the desired format.
OUTPUT SHOULD CONTAIN ONLY FINAL REPORT AND NOTHING ELSE
"""

# Functions
def extract_pdf_text(pdf_path):
    """Extract text from PDF using PyPDF2"""
    try:
        text_content = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text_content += pdf_reader.pages[page_num].extract_text() + "\n\n"
        return text_content
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_medical_report(pdf_path):
    """Process medical PDF reports into structured JSON data"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Extract text from PDF instead of using pdf2image
        text_content = extract_pdf_text(pdf_path)
        
        if not text_content:
            st.error("Could not extract text from PDF")
            return {}
            
        response = model.generate_content([MEDICAL_PROMPT, text_content])
        # Clean up response text
        cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        # Parse JSON safely
        return json.loads(cleaned_text)
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return {}

def load_medical_data(data_string):
    """Parses medical data from various formats."""
    try:
        # If it's already a dictionary, return it
        if isinstance(data_string, dict):
            return data_string

        # If it's a string, try to parse it
        if isinstance(data_string, str):
            # Remove markdown code blocks if present
            cleaned = data_string.replace("``````", "").strip()

            # Check if it looks like a Python dict literal (starts with '{')
            if cleaned.startswith('{') and cleaned.endswith('}'):
                try:
                    # Try parsing as JSON first
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    # If that fails, try evaluating as a Python literal
                    import ast
                    return ast.literal_eval(cleaned)
            else:
                return json.loads(cleaned)
        return None
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        return None

def merge_lab_results(data):
    """Merge lab results using Gemini model"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        formatted_data = {"parts": [{"text": f"{LAB_MERGE_PROMPT}\n\n{str(data)}"}]}
        response = model.generate_content(formatted_data)
        
        cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        return load_medical_data(cleaned_text)
    except Exception as e:
        st.error(f"Error merging lab results: {str(e)}")
        return None

class DiagnosticTool:
    def __init__(self, model):
        self.model = model

    def diagnose(self, patient_data, additional_lab_results=None):
        """Generates a medical diagnosis based on structured data."""
        if not patient_data:
            return "Invalid or missing patient data."

        patient_lab_results = patient_data.get("lab_results", [])

        if additional_lab_results:
            if isinstance(additional_lab_results, dict) and "test_name" in additional_lab_results:
                patient_lab_results.append(additional_lab_results)
            elif isinstance(additional_lab_results, list):
                patient_lab_results.extend(additional_lab_results)

        combined_data = {
            "Patient Information": patient_data.get("patient_info", {}),
            "Clinical Findings": patient_data.get("clinical_findings", {}),
            "Medications": patient_data.get("medications", []),
            "Lab Results": patient_lab_results
        }

        prompt = f"""
        Given the following structured medical data, provide a possible diagnosis:
        {json.dumps(combined_data, indent=2)}

        Ensure the diagnosis is accurate and medically relevant.
        """
        return self.model.invoke(prompt)

# Upload & Process tab
with tab1:
    st.header("Upload Medical Report")
    
    uploaded_files = st.file_uploader("Upload PDF medical reports", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Reports"):
        with st.spinner("Processing reports..."):
            all_reports_data = []
            
            # Create a temporary directory to save the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    # Save the uploaded file to the temporary directory
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the report
                    report_data = extract_medical_report(temp_file_path)
                    if report_data:
                        all_reports_data.append(report_data)
                        st.success(f"Successfully processed: {uploaded_file.name}")
                    else:
                        st.error(f"Failed to process: {uploaded_file.name}")
            
            if all_reports_data:
                # Store the processed data in the session state
                st.session_state.processed_data = all_reports_data
                st.success(f"Processed {len(all_reports_data)} reports successfully!")
                
                # Extract and merge lab results
                all_lab_results = []
                for report in all_reports_data:
                    if "lab_results" in report and report["lab_results"]:
                        all_lab_results.extend(report["lab_results"])
                
                if all_lab_results:
                    merged_results = merge_lab_results(all_lab_results)
                    st.session_state.merged_lab_results = merged_results
                    st.success("Lab results merged successfully!")

# Lab Results tab
with tab2:
    st.header("Lab Results Analysis")
    
    if st.session_state.processed_data:
        st.subheader("Extracted Lab Results")
        
        # Display each report's lab results
        for i, report_data in enumerate(st.session_state.processed_data):
            with st.expander(f"Report {i+1} - Lab Results"):
                if "lab_results" in report_data and report_data["lab_results"]:
                    st.json(report_data["lab_results"])
                else:
                    st.info("No lab results found in this report.")
        
        # Display merged lab results if available
        if st.session_state.merged_lab_results:
            st.subheader("Merged Lab Results")
            st.json(st.session_state.merged_lab_results)
    else:
        st.info("No processed reports yet. Please upload and process reports first.")

# Diagnosis tab
with tab3:
    st.header("Diagnostic Analysis")
    
    if st.session_state.processed_data and st.button("Generate Diagnosis"):
        with st.spinner("Generating diagnostic analysis..."):
            try:
                # Initialize the OpenAI model
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=None,
                    api_key=OPENAI_API_KEY
                )
                
                diagnostic_tool = DiagnosticTool(llm)
                
                # Create an agent
                medical_analysis_tool = Tool(
                    name="Medical Diagnosis Tool",
                    func=lambda x: diagnostic_tool.diagnose(st.session_state.processed_data[0], x),
                    description="Analyzes structured medical data and lab results to provide a diagnosis."
                )
                
                agent = initialize_agent(
                    tools=[medical_analysis_tool],
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False
                )
                
                # Generate diagnosis
                diagnosis = agent.run("Analyze the patient data and provide a detailed diagnosis.")
                st.session_state.diagnosis = diagnosis
                
            except Exception as e:
                st.error(f"Error during diagnosis: {str(e)}")
    
    if st.session_state.diagnosis:
        st.subheader("Diagnostic Analysis")
        st.markdown(st.session_state.diagnosis)
    elif st.session_state.processed_data:
        st.info("Click 'Generate Diagnosis' to analyze the medical reports.")
    else:
        st.info("No processed reports yet. Please upload and process reports first.")

# Footer
st.markdown("---")
st.caption("Medical Report Analysis System - Use for educational purposes only. Not intended for clinical decisions.")
