import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json
import logging
import traceback
from typing import Dict, List, Optional

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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data to a standard format."""
    logger.info("Starting data preprocessing")
    try:
        # Handle missing values
        df = df.fillna(0)
        
        # Ensure consistent data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass  # Keep as string if can't convert to numeric
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def generate_d3_code(df: pd.DataFrame, api_key: str, user_input: str = "") -> Optional[str]:
    """Generate D3.js code using OpenAI API."""
    logger.info("Starting D3 code generation")
    data_sample = df.head(50).to_dict(orient='records')
    schema = df.dtypes.to_dict()
    schema_str = "\n".join([f"{col}: {dtype}" for col, dtype in schema.items()])
    
    client = OpenAI(api_key=api_key)
    
    base_prompt = f"""
    Create a D3.js visualization based on the following data schema:

    {schema_str}

    Data sample:
    {json.dumps(data_sample[:5], indent=2)}

    Requirements:
    1. Create an appropriate chart type based on the data.
    2. Include grid lines for better readability.
    3. Add clear and informative labels for axes and the chart title.
    4. Include a legend if multiple data series are present.
    5. Use appropriate scales for the data.
    6. Implement basic interactivity (e.g., tooltips on hover).
    7. Ensure the chart is responsive and fits within an 800x500 pixel area.
    8. Use a pleasing color scheme.
    9. Handle potential null or undefined values gracefully.
    10. Include appropriate data transformations if necessary.

    Return only the D3.js code without any explanations.
    """
    
    prompt = f"{base_prompt}\n\nAdditional user request: {user_input}" if user_input else base_prompt

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Generate D3.js code based on the given requirements."},
                {"role": "user", "content": prompt}
            ]
        )
        
        d3_code = response.choices[0].message.content
        if not d3_code.strip():
            raise ValueError("Generated D3 code is empty")
        
        return postprocess_d3_code(d3_code)
    except Exception as e:
        logger.error(f"Error generating D3 code: {str(e)}")
        logger.info("API Response:", response)  # Log the full API response
        return None

def postprocess_d3_code(code: str) -> str:
    """Post-process the generated D3.js code to catch common issues."""
    # Ensure proper function closure
    if "function updateChart(data) {" in code and code.count("}") < code.count("{"):
        code += "\n}"
    
    # Replace unsupported d3.event with d3.pointer
    code = code.replace("d3.event", "d3.pointer(event)")
    
    # Ensure color scale is defined
    if "color(" in code and "const color = " not in code:
        code = "const color = d3.scaleOrdinal(d3.schemeCategory10);\n" + code
    
    # Add error handling
    code = "try {\n" + code + "\n} catch (error) { console.error('Error in D3 code:', error); }"
    
    return code

def display_visualization(d3_code: str) -> str:
    """Generate the HTML content for displaying the D3.js visualization."""
    html_content = f"""
    <div id="visualization"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    function runD3Code() {{
        try {{
            {d3_code}
        }} catch (error) {{
            console.error('Error in D3 code:', error);
            document.getElementById('visualization').innerHTML = '<p style="color: red;">Error generating visualization. Check console for details.</p>';
        }}
    }}
    
    if (document.readyState === 'complete') {{
        runD3Code();
    }} else {{
        document.addEventListener('DOMContentLoaded', runD3Code);
    }}
    </script>
    """
    return html_content

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
            if st.session_state.preprocessed_df is None:
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)

                df1['Source'] = 'CSV file 1'
                df2['Source'] = 'CSV file 2'

                merged_df = pd.concat([df1, df2], ignore_index=True)
                
                # Preprocess the merged data
                st.session_state.preprocessed_df = preprocess_data(merged_df)
            
            with st.expander("Preview of preprocessed data"):
                st.dataframe(st.session_state.preprocessed_df.head())
            
            if 'current_viz' not in st.session_state:
                try:
                    initial_d3_code = generate_d3_code(st.session_state.preprocessed_df, api_key)
                    if initial_d3_code:
                        st.session_state.current_viz = initial_d3_code
                        st.session_state.workflow_history.append({
                            "request": "Initial visualization",
                            "code": initial_d3_code
                        })
                    else:
                        st.error("Failed to generate initial visualization. Please check the error messages above.")
                        st.session_state.current_viz = None
                except Exception as e:
                    logger.error(f"Error generating initial visualization: {str(e)}")
                    st.error(f"Error generating initial visualization: {str(e)}")
                    st.code(traceback.format_exc())
                    st.session_state.current_viz = None

            st.subheader("Current Visualization")
            viz_placeholder = st.empty()
            if st.session_state.current_viz:
                viz_placeholder.write(display_visualization(st.session_state.current_viz))
            else:
                viz_placeholder.error("No visualization available. Check the error messages above.")

            st.subheader("Modify Visualization")
            user_input = st.text_input("Enter your modification request:")
            if st.button("Update Visualization"):
                if user_input:
                    modified_d3_code = generate_d3_code(st.session_state.preprocessed_df, api_key, user_input)
                    if modified_d3_code:
                        st.session_state.current_viz = modified_d3_code
                        st.session_state.workflow_history.append({
                            "request": user_input,
                            "code": modified_d3_code
                        })
                        viz_placeholder.empty()
                        viz_placeholder.write(display_visualization(st.session_state.current_viz))
                        st.success("Visualization updated successfully!")
                    else:
                        st.error("Failed to generate modified visualization. Please check the error messages above.")
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
                            st.empty()  # Clear the previous visualization
                            display_visualization(st.session_state.current_viz)
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
                        display_visualization(st.session_state.current_viz)

        except Exception as e:
            logger.error(f"An error occurred while processing the CSV files: {str(e)}")
            st.error(f"An error occurred while processing the CSV files: {str(e)}")
            st.code(traceback.format_exc())
    else:
        st.info("Please upload both CSV files and provide an API key to visualize your data")

if __name__ == "__main__":
    main()
