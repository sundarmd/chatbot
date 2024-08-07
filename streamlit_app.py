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
    columns = df.columns.tolist()[:2]
    data_sample = df.to_dict(orient='records')
    
    if user_input:
        client = OpenAI(api_key=api_key)
        prompt = f"Modify the following D3.js code according to this request: {user_input}\n\nExisting code:\n{st.session_state.current_viz}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Modify the given code according to the user's request. Return only the modified D3.js code."},
                {"role": "user", "content": prompt}
            ]
        )
        modified_code = response.choices[0].message.content
        return postprocess_d3_code(modified_code)
    
    # If no user input, generate initial code
    d3_code = f"""
    // Set the dimensions and margins of the graph
    const margin = {{top: 30, right: 30, bottom: 70, left: 60}},
        width = 800 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    // Append the svg object to the div
    const svg = d3.select("#visualization")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

    // Parse the Data
    const data = {json.dumps(data_sample)};

    // X axis
    const x = d3.scaleBand()
      .range([ 0, width ])
      .domain(data.map(d => d.{columns[0]}))
      .padding(0.2);
    svg.append("g")
      .attr("transform", `translate(0, ${{height}})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    // Add Y axis
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => +d.{columns[1]})])
      .range([ height, 0]);
    svg.append("g")
      .call(d3.axisLeft(y));

    // Bars
    svg.selectAll("mybar")
      .data(data)
      .join("rect")
        .attr("x", d => x(d.{columns[0]}))
        .attr("y", d => y(d.{columns[1]}))
        .attr("width", x.bandwidth())
        .attr("height", d => height - y(d.{columns[1]}))
        .attr("fill", "#69b3a2");
    """
    
    return d3_code

def display_visualization(d3_code):
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
    st.components.v1.html(html_content, height=520, scrolling=False)

    # Display the generated code for debugging
    with st.expander("View Generated D3 Code"):
        st.code(d3_code, language="javascript")

def main():
    st.set_page_config(page_title="ChartChat", layout="wide")
    st.title("ChartChat")

    # Initialize session state
    if 'workflow_history' not in st.session_state:
        st.session_state.workflow_history = []
    if 'current_viz' not in st.session_state:
        st.session_state.current_viz = None
    if 'preprocessed_df' not in st.session_state:
        st.session_state.preprocessed_df = None

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
                initial_d3_code = generate_d3_code(st.session_state.preprocessed_df, api_key)
                st.session_state.current_viz = initial_d3_code
                st.session_state.workflow_history.append({
                    "request": "Initial visualization",
                    "code": initial_d3_code
                })

            st.subheader("Current Visualization")
            viz_placeholder = st.empty()
            viz_placeholder.write(display_visualization(st.session_state.current_viz))

            st.subheader("Modify Visualization")
            user_input = st.text_input("Enter your modification request:")
            if st.button("Update Visualization"):
                if user_input:
                    modified_d3_code = generate_d3_code(st.session_state.preprocessed_df, api_key, user_input)
                    st.session_state.current_viz = modified_d3_code
                    st.session_state.workflow_history.append({
                        "request": user_input,
                        "code": modified_d3_code
                    })
                    viz_placeholder.empty()
                    viz_placeholder.write(display_visualization(st.session_state.current_viz))
                    st.success("Visualization updated successfully!")
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
            st.error(f"An error occurred while processing the CSV files: {str(e)}")
    else:
        st.info("Please upload both CSV files and provide an API key to visualize your data")

if __name__ == "__main__":
    main()
