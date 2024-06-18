import pandas as pd
import json
from pypdf import PdfReader
from PIL import Image
import pytesseract
from flask import Flask, request, jsonify
from dotenv import load_dotenv
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
import os
load_dotenv()

app = Flask(__name__)

load_dotenv()
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
    except KeyError as e:
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
    try:
        extracted_data = json.loads(extracted_data_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error during JSON decoding: {e}")

    extracted_data.setdefault("Doctor's Name", "Not Specified")
    extracted_data.setdefault("Patient's Name", "Not Specified")
    extracted_data.setdefault("Date", "Not Specified")
    extracted_data.setdefault("Drugs", [])

    drugs_str = "; ".join([f"{drug['name']} ({drug['dosage']})" for drug in extracted_data["Drugs"]])
    df = pd.DataFrame({
        'Doctor\'s Name': [extracted_data["Doctor's Name"]],
        'Patient\'s Name': [extracted_data["Patient's Name"]],
        'Date': [extracted_data["Date"]],
        'Drugs': [drugs_str]
    })

    return df

def create_docs(file):
    df = create_dataframe()

    print(f"Received file with MIME type: {file.mimetype}")

    if file.mimetype == "application/pdf":
        raw_data = get_pdf_text(file.stream)
    elif file.mimetype in ["image/png", "image/jpeg", "image/jpg"]:
        raw_data = get_image_text(file.stream)
    else:
        raise ValueError("Unsupported file type")

    llm_extracted_data = extract_data(raw_data)

    df = process_extracted_data_to_df(llm_extracted_data)

    return df

@app.route("/extract-prescription", methods=["POST"])
def extract_prescription():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        df = create_docs(file)
        if df is None:
            return jsonify({"error": "Error processing the file"}), 400

        result = df.to_dict(orient="records")[0]

        # Check business logic conditions
        warnings = []
        if result['Date'] == "Not Specified":
            warnings.append("This prescription requires a pharmacist to assist.")
        if (result['Doctor\'s Name'] == "Not Specified" and
                result['Patient\'s Name'] == "Not Specified" and
                result['Date'] == "Not Specified" and
                result['Drugs'] == ""):
            return jsonify({"error": "Blank prescription. Please upload a valid prescription."}), 400

        response = {
            "data": result,
            "warnings": warnings
        }
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
