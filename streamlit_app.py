import streamlit as st
import pandas as pd
import openai

def main():
    st.title("Generative Data Visualization")

    # Add a sidebar
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    # File upload section
    st.header("Upload CSV Files")
    file1 = st.file_uploader("Upload first CSV file", type="csv")
    file2 = st.file_uploader("Upload second CSV file", type="csv")

    if file1 and file2:
        # Read CSV files into pandas dataframes
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Add a column to specify the source file
        df1['Source'] = 'CSV file 1'
        df2['Source'] = 'CSV file 2'

        # Merge the dataframes
        merged_df = pd.concat([df1, df2], ignore_index=True)

        # Display the merged dataframe
        st.header("Merged Data")
        st.write(merged_df)

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
        if api_key:
            openai.api_key = api_key
            visualization_type = st.selectbox("Select visualization type", 
                ["Bar Chart", "Scatter Plot", "Line Chart", "Comparative Visualization"])
            
            if visualization_type == "Comparative Visualization":
                comparison_type = st.selectbox("Select comparison type", 
                    ["Superposition", "Juxtaposition", "Explicit Encoding"])
            
            user_input = st.text_area("Describe the visualization you want")

            # Initialize session state for workflow history
            if 'workflow_history' not in st.session_state:
                st.session_state.workflow_history = []

            if st.button("Generate Visualization"):
                prompt = f"""You are an expert at data visualization who regularly helps people to make sense of their data by creating beautiful visualizations that answers their questions meaningfully. You are an absolute expert at creating comparative visualizations (superposition, juxtaposition, explicit encoding) using d3.js code.

Please create a {visualization_type} {"with " + comparison_type if visualization_type == "Comparative Visualization" else ""} based on the following description:

{user_input}

The visualization should be created using D3.js. Please provide only the JavaScript code for the visualization."""

                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    max_tokens=1000,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )

                d3_code = response.choices[0].text.strip()
                st.session_state.workflow_history.append(d3_code)

            # Display workflow history
            st.subheader("Workflow History")
            for i, code in enumerate(st.session_state.workflow_history):
                st.text(f"Version {i+1}")
                edited_code = st.text_area(f"Edit code for version {i+1}", value=code, height=300, key=f"code_{i}")
                if st.button(f"Execute Version {i+1}"):
                    st.session_state.workflow_history[i] = edited_code
                    d3_code = edited_code

            if 'workflow_history' in st.session_state and st.session_state.workflow_history:
                st.subheader("Current D3.js Visualization")
                st.code(d3_code, language="javascript")
                st.components.v1.html(f"<script src='https://d3js.org/d3.v7.min.js'></script><div id='visualization'></div><script>{d3_code}</script>", height=600)
        else:
            st.warning("Please enter your OpenAI API key in the sidebar to generate visualizations.")
    else:
        st.info("Please upload both CSV files to merge them and create visualizations.")

    # Additional settings or options
    st.sidebar.checkbox("Enable advanced features")

    model_options = ["GPT-3.5", "GPT-4", "Other"]
    selected_model = st.sidebar.selectbox("Choose a model", model_options)

if __name__ == "__main__":
    main()
