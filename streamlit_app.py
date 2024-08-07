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

            # Generate initial line chart visualization using LLM
            initial_d3_code = generate_d3_code(merged_df, api_key)
            st.session_state.workflow_history.append(initial_d3_code)

            # Display initial visualization
            st.subheader("Initial Line Chart Visualization")
            display_visualization(initial_d3_code)

            # Display and edit code
            with st.expander("View/Edit Visualization Code"):
                code_editor = st.text_area("D3.js Code", value=initial_d3_code, height=300)
                col1, col2 = st.columns(2)
                with col1:
                    editable = st.toggle("Make Code Editable")
                with col2:
                    if st.button("Execute Code"):
                        if editable:
                            st.session_state.workflow_history.append(code_editor)
                            display_visualization(code_editor)
                        else:
                            st.warning("Enable 'Make Code Editable' to execute changes.")

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
    # Convert dataframe to JSON string
    data_json = df.to_json(orient='records')
    
    d3_code = f"""
    // Set the dimensions and margins of the graph
    const margin = {{top: 20, right: 20, bottom: 30, left: 50}},
          width = 960 - margin.left - margin.right,
          height = 500 - margin.top - margin.bottom;

    // Append the svg object to the visualization div
    const svg = d3.select("#visualization")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

    // Parse the Data
    const data = {data_json};

    console.log("Data:", data);  // Log the data for debugging

    // List of columns for X axis
    const columns = Object.keys(data[0]).filter(d => d !== 'Source');

    // Add X axis
    const x = d3.scalePoint()
      .domain(columns)
      .range([0, width]);
    svg.append("g")
      .attr("transform", `translate(0, ${{height}})`)
      .call(d3.axisBottom(x));

    // Add Y axis
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d3.max(columns, c => +d[c]))])
      .range([height, 0]);
    svg.append("g")
      .call(d3.axisLeft(y));

    // Color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw the lines
    const line = d3.line()
      .x(d => x(d.column))
      .y(d => y(+d.value));

    data.forEach((d, i) => {{
      const sourceData = columns.map(column => ({{column: column, value: d[column]}}));
      svg.append("path")
        .datum(sourceData)
        .attr("fill", "none")
        .attr("stroke", color(i))
        .attr("stroke-width", 1.5)
        .attr("d", line);
    }});

    // Add the legend
    const legend = svg.selectAll(".legend")
      .data(data.map(d => d.Source))
      .enter().append("g")
        .attr("class", "legend")
        .attr("transform", (d, i) => `translate(0,${{i * 20}})`);

    legend.append("rect")
      .attr("x", width - 18)
      .attr("width", 18)
      .attr("height", 18)
      .style("fill", (d, i) => color(i));

    legend.append("text")
      .attr("x", width - 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .text(d => d);

    console.log("D3 code executed successfully");
    """
    
    return d3_code

def modify_visualization(df, api_key, user_input, current_code):
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Modify the following D3.js code based on the user's request:

    User request: {user_input}

    Current D3.js code:
    {current_code}

    Please provide the modified D3.js code that incorporates the user's request while maintaining the existing functionality.
    Ensure the code uses D3.js version 7 and creates the chart within the 'visualization' div.
    Maintain the style and design requirements from the original visualization:
    1. Clean, minimalist design with a white background
    2. Clear, legible labeling
    3. Light gray gridlines
    4. Distinct color scheme for data categories
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
