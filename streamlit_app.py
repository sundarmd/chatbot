import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json

# Try to import dotenv, but don't fail if it's not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_api_key():
    """Retrieve the API key from various possible sources."""
    # Try to get the API key from Streamlit secrets
    api_key = st.secrets.get("OPENAI_API_KEY")
    
    # If not in secrets, try environment variables
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    # If still not found, prompt the user
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        if api_key:
            st.sidebar.warning("It's recommended to use environment variables or Streamlit secrets for API keys.")
    
    return api_key

def main():
    st.title("ChartChat")

    # Add a sidebar
    st.sidebar.header("Settings")
    
    # Get the API key
    api_key = get_api_key()

    # File upload section
    st.header("Upload CSV Files")
    file1 = st.file_uploader("Upload first CSV file", type="csv")
    file2 = st.file_uploader("Upload second CSV file", type="csv")

    if file1 and file2 and api_key:
        try:
            # Read CSV files into pandas dataframes
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)

            # Add a column to specify the source file
            df1['Source'] = 'CSV file 1'
            df2['Source'] = 'CSV file 2'

            # Merge the dataframes
            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # Display preview of merged data
            st.write("Preview of merged data:")
            st.dataframe(merged_df.head())
            
            # Initialize session state for workflow history
            if 'workflow_history' not in st.session_state:
                st.session_state.workflow_history = []

            # Generate initial visualization
            initial_d3_code = generate_d3_code(merged_df, api_key)
            st.session_state.workflow_history.append(initial_d3_code)

            # Display initial visualization
            st.subheader("Initial Visualization")
            display_visualization(initial_d3_code)

            # Display and edit code
            with st.expander("View/Edit Visualization Code"):
                code_editor = st.text_area("D3.js Code", value=initial_d3_code, height=300, disabled=True, key="code_editor")
                col1, col2 = st.columns(2)
                with col1:
                    edit_enabled = st.toggle("Edit", key="edit_toggle")
                with col2:
                    if st.button("Execute Code"):
                        if edit_enabled:
                            st.session_state.workflow_history.append(st.session_state.code_editor)
                            display_visualization(st.session_state.code_editor)
                        else:
                            st.warning("Enable 'Edit' to make changes.")

            # Update text area based on toggle
            if edit_enabled:
                st.session_state.code_editor = st.text_area("D3.js Code", value=st.session_state.code_editor, height=300, key="editable_code")

            # Display workflow history
            st.subheader("Workflow History")
            for i, code in enumerate(st.session_state.workflow_history):
                with st.expander(f"Version {i+1}"):
                    code_editor = st.text_area(f"Code Version {i+1}", value=code, height=300, key=f"history_{i}")
                    if st.button(f"Execute Version {i+1}"):
                        display_visualization(code_editor)
                        st.session_state.workflow_history[i] = code_editor

            # Chat with LLM to modify visualization
            st.subheader("Chat to Modify Visualization")
            user_input = st.text_input("Enter your modification request:")
            if st.button("Send Request"):
                modified_d3_code = modify_visualization(merged_df, api_key, user_input, initial_d3_code)
                st.session_state.workflow_history.append(modified_d3_code)
                display_visualization(modified_d3_code)

        except Exception as e:
            st.error(f"An error occurred while processing the CSV files: {str(e)}")
    else:
        st.info("Please upload both CSV files and provide an API key to visualize your data")

def generate_d3_code(df, api_key):
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
    // D3.js Visualization Scaffold for Dark Background

    // Set up SVG
    const margin = {top: 40, right: 100, bottom: 60, left: 60};
    const width = 800 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const svg = d3.select("#visualization")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Set up scales (example for numerical data)
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);

    // Set up color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Add axes
    const xAxis = svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .attr("class", "axis");

    const yAxis = svg.append("g")
        .attr("class", "axis");

    // Add gridlines
    svg.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x).tickSize(-height).tickFormat(""));

    svg.append("g")
        .attr("class", "grid")
        .call(d3.axisLeft(y).tickSize(-width).tickFormat(""));

    // Add title
    svg.append("text")
        .attr("x", width / 2)
        .attr("y", -margin.top / 2)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .style("fill", "#ffffff")
        .text("Visualization Title");

    // Style for dark background
    svg.selectAll(".axis path, .axis line, .grid line")
        .style("stroke", "#cccccc")
        .style("opacity", 0.2);

    svg.selectAll(".axis text")
        .style("fill", "#ffffff");

    // Add legend
    const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width + 20}, 0)`);

    // Function to update the chart (to be implemented based on data)
    function updateChart(data) {
        // Implementation depends on the specific chart type and data structure
    }

    // Load and process data
    // d3.json("data.json").then(function(data) {
    //     updateChart(data);
    // });
    """

    client = OpenAI(api_key=api_key)
    prompt = f"""
    Given the following data summary and D3.js scaffold, create a visualization that best represents the data:

    Data Summary:
    {json.dumps(data_summary, indent=2)}

    D3.js Scaffold:
    {scaffold_code}

    Please complete the scaffold code to create an appropriate visualization for this data.
    Ensure the visualization is optimized for a dark background and includes interactive elements like tooltips.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Generate only the code without explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating the D3.js code: {str(e)}")
        return ""

def modify_visualization(df, api_key, user_input, current_code):
    # Prepare data summary
    columns = df.columns.tolist()
    data_types = df.dtypes.to_dict()
    data_sample = df.head(5).to_dict(orient='records')
    data_summary = {
        "columns": columns,
        "data_types": {str(k): str(v) for k, v in data_types.items()},
        "sample_data": data_sample
    }

    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Modify the following D3.js code based on the user's request:

    User request: {user_input}

    Current D3.js code:
    {current_code}

    Data Summary:
    {json.dumps(data_summary, indent=2)}

    Please provide the modified D3.js code that incorporates the user's request while maintaining the existing functionality.
    Ensure the code uses D3.js version 7 and creates the chart within the 'visualization' div.
    Maintain the style and design requirements for a dark background:
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
                {"role": "system", "content": "You are a D3.js expert. Modify the given code based on the user's request."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while modifying the D3.js code: {str(e)}")
        return current_code

def display_visualization(d3_code):
    st.components.v1.html(f"""
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <div id="visualization" style="width:100%; height:550px;"></div>
        <script>
        (function() {{
            try {{
                console.log("Executing D3.js code");
                {d3_code}
                console.log("D3.js code executed successfully");
            }} catch (error) {{
                console.error("Error in D3.js code:", error);
                document.getElementById("visualization").innerHTML = "<p style='color: red;'>Error rendering visualization. Check the console for details.</p>";
            }}
        }})();
        </script>
    """, height=600, scrolling=True)
    st.write("If you don't see a visualization above, check the browser console for error messages.")

if __name__ == "__main__":
    main()
