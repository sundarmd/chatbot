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
    client = OpenAI(api_key=api_key)
    
    # Prepare data summary
    columns = df.columns.tolist()
    data_types = df.dtypes.to_dict()
    data_sample = df.head(5).to_dict(orient='records')
    
    prompt = f"""
    Generate D3.js code for an interactive visualization based on the following dataset:
    
    Columns: {columns}
    Data types: {data_types}
    Sample data: {json.dumps(data_sample)}
    
    Style and design requirements:
    1. Use a clean, minimalist design with a white background.
    2. Implement clear, legible labeling for all chart elements.
    3. Include light gray gridlines to aid in reading values.
    4. Use a distinct color scheme to differentiate between data categories or series.
    5. Ensure the visualization is responsive and fits well within a Streamlit app.
    6. Use larger font sizes for better readability.
    7. Include a title and axis labels that clearly describe the data being visualized.
    8. If applicable, place a legend outside the main plot area for clarity.
    9. Remove unnecessary chart borders or elements to reduce visual clutter.
    10. Implement smooth transitions for any interactive features or updates.
    11. Add tooltips or interactive elements to display detailed information on user interaction.
    12. Ensure the visualization is accessible with proper ARIA attributes.

    Technical requirements:
    1. Use D3.js version 7 for compatibility.
    2. Create the visualization within a div with the id 'visualization'.
    3. Include all necessary data processing within the D3.js code.
    4. Ensure the code can handle the given dataset structure and types.
    5. Implement appropriate scales and axes based on the data characteristics.
    6. Set the width to 100% of the container and height to 500px.
    7. Add margin to the chart for labels and axes.
    8. Ensure the code is bug-free and handles potential errors gracefully.

    Please provide the complete D3.js code that can be directly used in a Streamlit component.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a D3.js expert. Generate only the code without any explanations."},
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
        <div id="visualization" style="width:100%; height:500px;"></div>
        <script>
        (function() {{
            {d3_code}
        }})();
        </script>
    """, height=550, scrolling=True)

if __name__ == "__main__":
    main()
