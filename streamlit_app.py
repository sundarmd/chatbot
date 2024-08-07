import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json
import re

# Try to import dotenv, but don't fail if it's not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_api_key():
    """Retrieve the API key from various possible sources."""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            st.sidebar.warning("It's recommended to use environment variables or Streamlit secrets for API keys.")
    return api_key

def preprocess_data(df):
    """Preprocess the data to a standard format."""
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
    
    return df

def postprocess_d3_code(code):
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

def generate_d3_code(df, api_key, user_input=""):
    # Prepare data summary
    columns = df.columns.tolist()
    data_types = df.dtypes.to_dict()
    data_sample = df.head(5).to_dict(orient='records')
    data_summary = {
        "columns": columns,
        "data_types": {str(k): str(v) for k, v in data_types.items()},
        "sample_data": data_sample
    }

    scaffold_code = """
    // D3.js Visualization Scaffold
    const margin = {top: 60, right: 120, bottom: 80, left: 80};
    const width = 800 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    d3.select("#visualization").selectAll("*").remove();

    const svg = d3.select("#visualization")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Set up scales (example)
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);

    // Set up axes (example)
    const xAxis = svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .attr("class", "axis");
    const yAxis = svg.append("g")
        .attr("class", "axis");

    // Add title
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", -margin.top / 2)
        .attr("text-anchor", "middle")
        .attr("class", "chart-title")
        .text("Chart Title");

    // Add x-axis label
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", height + margin.bottom / 2)
        .attr("text-anchor", "middle")
        .attr("class", "axis-label")
        .text("X Axis Label");

    // Add y-axis label
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -margin.left / 2)
        .attr("text-anchor", "middle")
        .attr("class", "axis-label")
        .text("Y Axis Label");

    function updateChart(data) {
        // Your visualization code here
    }

    // Load data and call updateChart
    const data = ${json.dumps(data_sample)};
    updateChart(data);

    // Add styles
    const styles = `
        #visualization {
            font-family: Arial, sans-serif;
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .axis path,
        .axis line {
            stroke: #4f4f4f;
        }
        .axis text {
            fill: #ffffff;
            font-size: 12px;
        }
        .chart-title {
            font-size: 20px;
            fill: #ffffff;
        }
        .axis-label {
            font-size: 14px;
            fill: #ffffff;
        }
    `;
    const styleElement = document.createElement('style');
    styleElement.textContent = styles;
    document.head.appendChild(styleElement);
    """

    client = OpenAI(api_key=api_key)
    prompt = f"""
    Given the following data summary, create a D3.js visualization that best represents the data:

    Data Summary:
    {json.dumps(data_summary, indent=2)}

    Scaffold Code:
    {scaffold_code}

    User Request: {user_input}

    Please complete the updateChart function to create an appropriate visualization for this data.
    Ensure the visualization is optimized for a dark background and includes interactive elements like tooltips.
    Use only the variables defined in the scaffold (svg, x, y, xAxis, yAxis).
    If there's a user request, try to incorporate it into the visualization.
    
    Follow these style and design requirements:
    1. Clean, minimalist design with a dark background
    2. Clear, legible labeling with light text
    3. Light gray gridlines
    4. Distinct color scheme for data categories, suitable for dark backgrounds
    5. Responsive design
    6. Large, readable font sizes
    7. Clear title and axis labels
    8. Legend outside the main plot area (if applicable)
    9. Reduced visual clutter
    10. Smooth transitions for interactivity
    11. Tooltips or interactive elements for detailed information
    12. Accessibility with ARIA attributes
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Generate only the code for the updateChart function without explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            n=1,
            temperature=0.7,
        )
        generated_code = response.choices[0].message.content.strip()
        
        # Post-process the generated code
        processed_code = postprocess_d3_code(generated_code)
        
        # Insert the processed code into the scaffold
        complete_code = scaffold_code.replace("// Your visualization code here", processed_code)
        
        return complete_code
    except Exception as e:
        st.error(f"An error occurred while generating the D3.js code: {str(e)}")
        return scaffold_code  # Return the scaffold code as a fallback

def display_visualization(d3_code):
    # Debug: Print the generated D3 code
    st.text("Generated D3 Code:")
    st.code(d3_code, language="javascript")

    html_content = f"""
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <div id="visualization" style="width:100%; height:500px;"></div>
    <script>
    (function() {{
        try {{
            {d3_code}
        }} catch (error) {{
            console.error('Error in D3 code:', error);
            document.getElementById('visualization').innerHTML = '<p style="color: red;">Error generating visualization. Check console for details.</p>';
        }}
    }})();
    </script>
    """
    st.components.v1.html(html_content, height=520, scrolling=False)

def main():
    st.set_page_config(page_title="ChartChat", layout="wide")
    st.title("ChartChat")

    # Initialize session state for workflow history if it doesn't exist
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []

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

            df1['Source'] = 'CSV file 1'
            df2['Source'] = 'CSV file 2'

            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # Preprocess the merged data
            preprocessed_df = preprocess_data(merged_df)
            
            with st.expander("Preview of preprocessed data"):
                st.dataframe(preprocessed_df.head())
            
            if 'current_viz' not in st.session_state:
                initial_d3_code = generate_d3_code(preprocessed_df, api_key)
                st.session_state.current_viz = initial_d3_code
                st.session_state.workflow_history.append({
                    "request": "Initial visualization",
                    "code": initial_d3_code
                })

            st.subheader("Current Visualization")
            display_visualization(st.session_state.current_viz)

            st.subheader("Modify Visualization")
            user_input = st.text_input("Enter your modification request:")
            if st.button("Send Request"):
                modified_d3_code = generate_d3_code(preprocessed_df, api_key, user_input)
                st.session_state.workflow_history.append({
                    "request": user_input,
                    "code": modified_d3_code
                })
                st.session_state.current_viz = modified_d3_code
                st.experimental_rerun()

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
                            st.experimental_rerun()
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
                        st.experimental_rerun()

        except Exception as e:
            st.error(f"An error occurred while processing the CSV files: {str(e)}")
    else:
        st.info("Please upload both CSV files and provide an API key to visualize your data")

if __name__ == "__main__":
    main()
