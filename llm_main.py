import pandas as pd
import re
import json
import ast
from pypdf import PdfReader
from PIL import Image
import pytesseract
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_image_text(image_doc):
    image = Image.open(image_doc)
    text = pytesseract.image_to_string(image)
    return text



def extract_data(pages_data):
    template = '''
Extract the following values from the given text: 
- Doctor's Name
- Patient's Name
- Date
- Drugs (with names and dosages)

Text: {pages}

Format the output as JSON with the keys: 
- "Doctor's Name"
- "Patient's Name"
- "Date"
- "Drugs" (a list of dictionaries with "name" and "dosage" keys)

If any value is missing, use "Not Specified".

Example output:
{{
    "Doctor's Name": "Not Specified",
    "Patient's Name": "Jane Smith",
    "Date": "Not Specified",
    "Drugs": [
        {{"name": "Aspirin", "dosage": "100mg"}},
        {{"name": "Metformin", "dosage": "500mg"}}
    ]
}}
'''

    try:
        prompt = template.format(pages=pages_data)
        # print("Formatted Prompt:")
        print(prompt)
    except KeyError as e:
        print(f"KeyError during prompt formatting: {e}")
        raise

    llm = OpenAI(temperature=0.9)
    full_response = llm(prompt)
    return full_response


def create_dataframe():
    df = pd.DataFrame({
        'Doctor\'s Name': pd.Series(dtype='str'),
        'Patient\'s Name': pd.Series(dtype='str'),
        'Date': pd.Series(dtype='str'),
        'Drugs': pd.Series(dtype='object')
    })
    return df

def process_extracted_data_to_df(extracted_data_str):
    # Convert the string to a dictionary
    try:
        extracted_data = json.loads(extracted_data_str)
    except json.JSONDecodeError as e:
        print(f"Error during JSON decoding: {e}")
        return None

    # Fill missing fields with "Not Specified"
    extracted_data.setdefault("Doctor's Name", "Not Specified")
    extracted_data.setdefault("Patient's Name", "Not Specified")
    extracted_data.setdefault("Date", "Not Specified")
    extracted_data.setdefault("Drugs", [])

    drugs_str = "; ".join([f"{drug['name']} ({drug['dosage']})" for drug in extracted_data["Drugs"]])
    # Create a DataFrame with the extracted data
    df = pd.DataFrame({
        'Doctor\'s Name': [extracted_data["Doctor's Name"]],
        'Patient\'s Name': [extracted_data["Patient's Name"]],
        'Date': [extracted_data["Date"]],
        'Drugs': [drugs_str] # Assign the drugs list
    })

    return df

def create_docs(file):
    df = create_dataframe()

    # Determine file type and extract text
    if file.type == "application/pdf":
        raw_data = get_pdf_text(file)
    elif file.type in ["image/png", "image/jpeg"]:
        raw_data = get_image_text(file)
    else:
        raise ValueError("Unsupported file type")

    # Print raw data
    print("Raw data extracted:")
    print(raw_data)

    # Extract data using LLM
    llm_extracted_data = extract_data(raw_data)

    # Print LLM extracted data
    print("LLM extracted data:")
    print(llm_extracted_data)

    # Process the extracted data
    df = process_extracted_data_to_df(llm_extracted_data)

    return df




# Example usage with Streamlit
def main():
    load_dotenv()

    st.set_page_config(page_title="Prescription Extraction Bot")
    st.title("Prescription Extraction Bot...💊 ")
    st.subheader("I can help you in extracting data from prescriptions")

    # Upload the file (pdf, png, jpeg)
    file = st.file_uploader("Upload prescription here (PDF, PNG, JPEG)", type=["pdf", "png", "jpeg"])

    submit = st.button("Extract Data")

    if submit and file is not None:
        with st.spinner('Processing...'):
            df = create_docs(file)
            st.dataframe(df)

            # Check business logic conditions
            if df['Date'].iloc[0] == "Not Specified":
                st.warning("This prescription requires a pharmacist to assist.")

            if (df['Doctor\'s Name'].iloc[0] == "Not Specified" and
                    df['Patient\'s Name'].iloc[0] == "Not Specified" and
                    df['Date'].iloc[0] == "Not Specified" and
                    df['Drugs'].iloc[0] == ""):
                st.error("Blank prescription. Please upload a valid prescription.")

            data_as_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download data as CSV",
                data_as_csv,
                "prescription_data.csv",
                "text/csv",
                key="download-csv",
            )
        st.success("Extraction completed successfully! 🎉")

if __name__ == '__main__':
    main()
