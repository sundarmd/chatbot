import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json
import logging
import traceback
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []
if 'current_viz' not in st.session_state:
    st.session_state.current_viz = None
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None

def get_api_key() -> Optional[str]:
    """Securely retrieve the API key."""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            st.sidebar.warning("It's recommended to use environment variables or Streamlit secrets for API keys.")
    return api_key

def test_api_key(api_key: str) -> bool:
    """Test if the provided API key is valid."""
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        return False

def preprocess_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and merge the two dataframes for comparison."""
    logger.info("Starting data preprocessing")
    try:
        df1['Source'] = 'CSV file 1'
        df2['Source'] = 'CSV file 2'
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Handle missing values
        merged_df = merged_df.fillna(0)
        
        # Ensure consistent data types
        for col in merged_df.columns:
            if merged_df[col].dtype == 'object':
                try:
                    merged_df[col] = pd.to_numeric(merged_df[col])
                except ValueError:
                    pass  # Keep as string if can't convert to numeric
        
        # Standardize column names
        merged_df.columns = merged_df.columns.str.lower().str.replace(' ', '_')
        
        logger.info("Data preprocessing completed successfully")
        return merged_df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def generate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> str:
    """Generate D3.js code using OpenAI API with emphasis on comparison."""
    logger.info("Starting D3 code generation")
    data_sample = df.head(50).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])
    
    client = OpenAI(api_key=api_key)
    
    base_prompt = f"""
    Create a D3.js visualization that compares data from two CSV files based on the following schema:

    {schema_str}

    Data sample:
    {json.dumps(data_sample[:5], indent=2)}

    Requirements:
    1. Create a chart that clearly shows the comparison between 'CSV file 1' and 'CSV file 2'.
    2. Use different colors or patterns to distinguish between the two data sources.
    3. Include a legend to identify which elements correspond to each CSV file.
    4. Add clear and informative labels for axes and the chart title.
    5. Implement basic interactivity (e.g., tooltips on hover) to show detailed information.
    6. Ensure the chart is responsive and fits within an 800x500 pixel area.
    7. Handle potential null or undefined values gracefully.
    8. Include grid lines and appropriate scales for better readability.
    9. Use D3.js version 7 syntax.
    10. Wrap the entire D3 code in a function named createVisualization(data).
    11. Add accessibility features such as ARIA labels.
    12. Implement data sampling for large datasets.
    13. Use relative positioning for tooltips.

    Return only the D3.js code without any explanations.
    """
    
    prompt = f"{base_prompt}\n\nAdditional user request: {user_input}" if user_input else base_prompt

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Generate D3.js code for comparative visualization based on the given requirements."},
                {"role": "user", "content": prompt}
            ]
        )
        
        d3_code = response.choices[0].message.content
        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")
        
        return postprocess_d3_code(d3_code)
    except Exception as e:
        logger.error(f"Error generating D3 code: {str(e)}")
        return generate_fallback_visualization(df)

def postprocess_d3_code(code: str) -> str:
    """Post-process the generated D3.js code to catch common issues."""
    # Remove the outer function if it exists
    if code.startswith("function createVisualization(data) {"):
        code = code[len("function createVisualization(data) {"):-1].strip()
    
    # Replace unsupported d3.event with d3.pointer
    code = code.replace("d3.event", "d3.pointer(event)")
    
    # Ensure color scale is defined if used
    if "colorScale(" in code and "const colorScale = " not in code:
        code = "const colorScale = d3.scaleOrdinal(d3.schemeCategory10);\n" + code
    
    # Add error checking for data structure
    code = "console.log(data[0]);\n" + code
    
    # Add data sampling for large datasets
    code = """
    const sampleSize = 1000;
    const sampledData = data.length > sampleSize ? data.slice(0, sampleSize) : data;
    """ + code.replace("data", "sampledData")
    
    return code

def display_visualization(d3_code: str) -> str:
    """Generate the HTML content for displaying the D3.js visualization."""
    html_content = f"""
    <style>
        .tooltip {{
            position: absolute;
            background-color: white;
            border: 1px solid #ddd;
            padding: 10px;
            pointer-events: none;
        }}
    </style>
    <div id="visualization"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    (function() {{
        const data = {json.dumps(st.session_state.preprocessed_df)};
        const createVisualization = (data) => {{
            // Error checking
            if (!data || !Array.isArray(data) || data.length === 0) {{
                console.error('Invalid or empty data provided to createVisualization');
                return;
            }}
            
            {d3_code}
        }};
        createVisualization(data);
        
        // Make the chart responsive
        function resizeChart() {{
            const container = d3.select('#visualization');
            const containerWidth = container.node().getBoundingClientRect().width;
            const containerHeight = containerWidth * (500 / 800);
            
            d3.select('#visualization svg')
                .attr('width', containerWidth)
                .attr('height', containerHeight);
        }}
        
        // Call resize function initially and on window resize
        resizeChart();
        window.addEventListener('resize', resizeChart);
    }})();
    </script>
    """
    return html_content

def generate_fallback_visualization(df: pd.DataFrame) -> str:
    """Generate a fallback visualization if the LLM fails."""
    # Implement a simple bar chart or scatter plot here
    pass

def main():
    st.set_page_config(page_title="ChartChat", layout="wide")
    st.title("ChartChat")

    api_key = get_api_key()

    st.header("Upload CSV Files")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload first CSV file", type="csv")
    with col2:
        file2 = st.file_uploader("Upload second CSV file", type="csv")

    if file1 and file2 and api_key:
        try:
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            
            # Preprocess and merge the data
            merged_df = preprocess_data(df1, df2)
            st.session_state.preprocessed_df = merged_df.to_dict(orient='records')
            
            with st.expander("Preview of preprocessed data"):
                st.dataframe(merged_df.head())
            
            if 'current_viz' not in st.session_state or st.session_state.current_viz is None:
                initial_d3_code = generate_d3_code(merged_df, api_key)
                st.session_state.current_viz = initial_d3_code
                st.session_state.workflow_history.append({
                    "request": "Initial comparative visualization",
                    "code": initial_d3_code
                })

            st.subheader("Current Visualization")
            st.components.v1.html(display_visualization(st.session_state.current_viz), height=600)

            st.subheader("Modify Visualization")
            modification_options = [
                "Change chart type",
                "Adjust color scheme",
                "Add annotations",
                "Modify axes",
                "Change data representation",
                "Add trend lines",
                "Modify legend",
                "Other (please specify)"
            ]
            selected_option = st.selectbox("Choose modification type:", modification_options)
            user_input = st.text_input("Enter your modification request:")
            
            if 'update_viz' not in st.session_state:
                st.session_state.update_viz = False

            if st.button("Update Visualization") or st.session_state.update_viz:
                if user_input:
                    with st.spinner("Generating updated visualization..."):
                        modified_d3_code = generate_d3_code(merged_df, f"{selected_option}: {user_input}")
                    st.session_state.current_viz = modified_d3_code
                    st.session_state.workflow_history.append({
                        "request": f"{selected_option}: {user_input}",
                        "code": modified_d3_code
                    })
                    if len(st.session_state.workflow_history) > MAX_WORKFLOW_HISTORY:
                        st.session_state.workflow_history.pop(0)
                    st.session_state.update_viz = False
                else:
                    st.warning("Please enter a modification request.")

            with st.expander("View/Edit Visualization Code"):
                code_editor = st.text_area("D3.js Code", value=st.session_state.current_viz, height=300, key="code_editor")
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    edit_enabled = st.toggle("Edit", key="edit_toggle")
                with col2:
                    if st.button("Execute Code"):
                        if edit_enabled:
                            st.session_state.current_viz = code_editor
                            st.session_state.workflow_history.append({
                                "request": "Manual code edit",
                                "code": code_editor
                            })
                            if len(st.session_state.workflow_history) > MAX_WORKFLOW_HISTORY:
                                st.session_state.workflow_history.pop(0)
                            st.empty()  # Clear the previous visualization
                            st.components.v1.html(display_visualization(st.session_state.current_viz), height=600)
                        else:
                            st.warning("Enable 'Edit' to make changes.")
                with col3:
                    if st.button("Copy Code"):
                        st.write("Code copied to clipboard!")
                        st.write(f'<textarea style="position: absolute; left: -9999px;">{code_editor}</textarea>', unsafe_allow_html=True)
                        st.write('<script>document.querySelector("textarea").select();document.execCommand("copy");</script>', unsafe_allow_html=True)

            with st.expander("Workflow History"):
                for i, step in enumerate(st.session_state.workflow_history):
                    st.subheader(f"Step {i+1}")
                    st.write(f"Request: {step['request']}")
                    if st.button(f"Revert to Step {i+1}"):
                        st.session_state.current_viz = step['code']
                        st.empty()  # Clear the previous visualization
                        st.components.v1.html(display_visualization(st.session_state.current_viz), height=600)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main function: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("An unexpected error occurred. Please try again or contact support if the problem persists.")
    else:
        st.info("Please upload both CSV files and provide a valid API key to visualize your data")

if __name__ == "__main__":
    main()
