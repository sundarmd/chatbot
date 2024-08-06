import streamlit as st
import pandas as pd
import openai
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
            
            # Display the merged dataframe
            st.subheader("Merged Data")
            st.dataframe(merged_df)
            
            # Download button for merged CSV
            st.download_button(
                label="Download merged CSV",
                data=merged_df.to_csv(index=False),
                file_name="merged_data.csv",
                mime="text/csv",
            )
            
            # Visualization section
            st.header("Data Visualization")

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

        except Exception as e:
            st.error(f"An error occurred while processing the CSV files: {str(e)}")
    else:
        st.info("Please upload both CSV files and provide an API key to visualize your data")

def generate_d3_code(df, api_key):
    openai.api_key = api_key
    
    # Prepare data summary
    columns = df.columns.tolist()
    data_types = df.dtypes.to_dict()
    data_sample = df.head(5).to_dict(orient='records')
    
    prompt = f"""
    Generate a D3.js line chart code for the following dataset:
    
    Columns: {columns}
    Data types: {data_types}
    Sample data: {json.dumps(data_sample)}
    
    Requirements:
    1. Create a line chart comparing data from both CSV files.
    2. Use distinct and meaningful color coding for different lines.
    3. Add appropriate labels for axes and legend.
    4. Make the chart responsive and fit well within a Streamlit app.
    5. Use appropriate scales based on the data types.
    6. Include a legend to distinguish between different data sources or categories.
    
    Please provide the complete D3.js code that can be directly used in a Streamlit component.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Generate only the code without any explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating the D3.js code: {str(e)}")
        return ""

def display_visualization(d3_code):
    st.components.v1.html(f"""
        <script src='https://d3js.org/d3.v7.min.js'></script>
        <div id='visualization'></div>
        <script>{d3_code}</script>
    """, height=600, scrolling=True, sandbox="allow-scripts")

if __name__ == "__main__":
    main()
