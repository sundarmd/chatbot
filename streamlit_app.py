import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def main():
    st.title("ChartChat")

    # Add a sidebar
    st.sidebar.header("Settings")
    
    # Use environment variable for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        st.sidebar.warning("It's recommended to use environment variables for API keys.")

    # Additional settings or options
    advanced_features = st.sidebar.checkbox("Enable advanced features")

    if advanced_features:
        st.sidebar.subheader("Advanced Settings")
        model_options = ["gpt-3.5-turbo", "gpt-4", "Other"]
        selected_model = st.sidebar.selectbox("Choose a model", model_options)
        if selected_model == "Other":
            custom_model = st.sidebar.text_input("Enter custom model name")
        
        # Add more advanced features here

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

            # Display the merged dataframe in an expander
            with st.expander("Your Data"):
                page_size = 100
                page_number = st.number_input("Page", min_value=1, value=1)
                start = (page_number - 1) * page_size
                end = start + page_size
                st.write(merged_df.iloc[start:end])

                # Option to download the merged CSV
                csv = merged_df.to_csv(index=False)
                st.download_button(
                    label="Download merged CSV",
                    data=csv,
                    file_name="merged_data.csv",
                    mime="text/csv",
                )

            # Visualization section
            st.header("Data Visualization")
            openai.api_key = api_key

            # Initialize session state for workflow history
            if 'workflow_history' not in st.session_state:
                st.session_state.workflow_history = []

            # Generate initial visualization
            initial_prompt = f"""Generate a simple comparative visualization using D3.js for the merged dataset. 
            The visualization should compare data from both CSV files. Choose an appropriate chart type for a general comparison.
            Use the following template and modify the specified parts:

            ```
            // Import D3.js
            import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

            // Set the dimensions and margins of the graph
            const margin = {{top: 10, right: 30, bottom: 30, left: 60}},
                width = 460 - margin.left - margin.right,
                height = 400 - margin.top - margin.bottom;

            // Append the svg object to the body of the page
            const svg = d3.select("#visualization")
              .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
              .append("g")
                .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            // Read the data
            const data = {json.dumps(merged_df.to_dict(orient='records'))}

            // Add your comparative visualization code here
            ```

            The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ...
            """

            try:
                model_to_use = custom_model if advanced_features and selected_model == "Other" else (selected_model if advanced_features else "gpt-3.5-turbo")

                response = openai.ChatCompletion.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates D3.js visualizations."},
                        {"role": "user", "content": initial_prompt}
                    ],
                    max_tokens=1000,
                    n=1,
                    temperature=0.7,
                )

                initial_d3_code = response.choices[0].message.content.strip()
                st.session_state.workflow_history.append(initial_d3_code)

                # Display initial visualization
                st.subheader("Initial Comparative Visualization")
                st.components.v1.html(f"""
                    <script src='https://d3js.org/d3.v7.min.js'></script>
                    <div id='visualization'></div>
                    <script>{initial_d3_code}</script>
                """, height=600, scrolling=True, sandbox="allow-scripts")

            except Exception as e:
                st.error(f"An error occurred while generating the initial visualization: {str(e)}")

            # Display workflow history in an expander
            with st.expander("Workflow History"):
                for i, code in enumerate(st.session_state.workflow_history):
                    st.text(f"Version {i+1}")
                    edited_code = st.text_area(f"Edit code for version {i+1}", value=code, height=300, key=f"code_{i}")
                    if st.button(f"Execute Version {i+1}"):
                        st.session_state.workflow_history[i] = edited_code
                        d3_code = edited_code
                        st.subheader(f"D3.js Visualization (Version {i+1})")
                        st.components.v1.html(f"""
                            <script src='https://d3js.org/d3.v7.min.js'></script>
                            <div id='visualization'></div>
                            <script>{d3_code}</script>
                        """, height=600, scrolling=True, sandbox="allow-scripts")

            # User input for new visualizations
            user_input = st.text_area("Describe a new comparative visualization you want")
            if st.button("Generate New Visualization"):
                new_prompt = f"""You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Given some code template, you complete the template to generate a COMPARATIVE visualization given the dataset and the goal described. The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, apply the right transformation, use the right visualization type, use the right data encoding, and use the right aesthetics (e.g., ensure axis are legible). The transformations you apply MUST be correct and the fields you use MUST be correct. The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must consider the field types and use them correctly). You MUST first generate a brief plan for how you would solve the task e.g. what transformations you would apply e.g. if you need to construct a new column, what fields you would use, what visualization type you would use, what aesthetics you would use, etc.

                Please create a COMPARATIVE visualization based on the following description:

                {user_input}

                The visualization MUST be comparative in nature, comparing data from both CSV files. If the user specifies a particular chart type (e.g., bar chart, line chart), use that type. If not specified, choose the most appropriate type for a comparative visualization.

                Always add a legend with various colors where appropriate. The visualization code MUST only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names). Only use variables that have been defined in the code or are in the dataset summary. You MUST return a FULL d3.js PROGRAM ENCLOSED IN BACKTICKS ``` that starts with an import statement. DO NOT add any explanation.

                THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW

                ```
                // Import D3.js
                import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

                // Set the dimensions and margins of the graph
                const margin = {{top: 10, right: 30, bottom: 30, left: 60}},
                    width = 460 - margin.left - margin.right,
                    height = 400 - margin.top - margin.bottom;

                // Append the svg object to the body of the page
                const svg = d3.select("#visualization")
                  .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                  .append("g")
                    .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

                // Read the data
                const data = {json.dumps(merged_df.to_dict(orient='records'))}

                // Add your comparative visualization code here
                ```

                The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ...
                """

                try:
                    response = openai.ChatCompletion.create(
                        model=model_to_use,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates D3.js visualizations."},
                            {"role": "user", "content": new_prompt}
                        ],
                        max_tokens=1000,
                        n=1,
                        temperature=0.7,
                    )

                    new_d3_code = response.choices[0].message.content.strip()
                    st.session_state.workflow_history.append(new_d3_code)

                    # Display new visualization
                    st.subheader("New D3.js Visualization")
                    st.components.v1.html(f"""
                        <script src='https://d3js.org/d3.v7.min.js'></script>
                        <div id='visualization'></div>
                        <script>{new_d3_code}</script>
                    """, height=600, scrolling=True, sandbox="allow-scripts")

                except Exception as e:
                    st.error(f"An error occurred while generating the new visualization: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred while processing the CSV files: {str(e)}")
    else:
        st.info("Please upload both CSV files and provide an API key to visualize your data")

if __name__ == "__main__":
    main()
