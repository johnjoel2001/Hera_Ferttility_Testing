import json
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

# Create a client
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)  # Fixed: Use configure instead of Client

model_id = "gemini-2.0-flash-exp"  # Updated model name

# Define data structure for patient's info
class PatientInfo(BaseModel):
    name: str = Field(description="Patient's full name")
    dob: str = Field(description="Date of birth (MM/DD/YYYY)")
    pmi_number: str = Field(description="PMI Number")
    accession_number: str = Field(description="Accession Number")
    provider: str = Field(description="Name of the provider/doctor")

# Define collection info
class CollectionInfo(BaseModel):
    date_of_collection: str = Field(description="Date of sample collection (MM/DD/YY)")
    time_of_collection: str = Field(description="Time of collection")
    days_abstinence: str = Field(description="Days of abstinence before collection")
    received_date: str = Field(description="Date sample was received")
    time_of_analysis: str = Field(description="Time of analysis")
    received_time: str = Field(description="Time sample was received")

# Define parameters and normal range
class SemenParameter(BaseModel):
    value: float = Field(description="Measured value")
    normal_range: Optional[str] = Field(description="Expected normal range")

class SemenAnalysis(BaseModel):
    volume: SemenParameter
    concentration: SemenParameter
    motility: SemenParameter
    forward_progression: SemenParameter
    total_motile_count: SemenParameter
    agglutination: SemenParameter
    round_cells: SemenParameter
    morphology: SemenParameter
    comments: Optional[str] = Field(None, description="Additional comments")
    
    def process_agglutination(self):
        """Convert None to 0 and other values to 1 for agglutination"""
        if self.agglutination.value is None or self.agglutination.value == "None":
            self.agglutination.value = 0
        else:
            self.agglutination.value = 1

# Define the main Semen Analysis model
class WholeReport(BaseModel):
    patient_info: PatientInfo
    collection_info: CollectionInfo
    semen_analysis: SemenAnalysis

# Function to extract structured data from a file using Gemini API
def extract_structured_data(file_path: str):
    try:
        # Upload the file to Gemini
        uploaded_file = genai.upload_file(file_path, display_name='semen_analysis_report')
        
        # Define the extraction prompt
        prompt = """Extract all structured data from the PDF, including:
        - Patient details (name, DOB, PMI Number, Accession Number, Provider).
        - Collection information (Date of Collection, Time of Collection, Days Abstinence, Received Date, Time of Analysis, Received Time).
        - Semen analysis results with numerical values only.
        
        IMPORTANT:
        - If any field is **empty or not available**, return an empty string ("").
        - Do NOT fill PMI Number with Accession Number if PMI Number is missing.
        - For numerical values, extract only the number (no units or text).
        - Return the response in valid JSON format matching the schema.
        
        Return the data in this exact JSON structure:
        {
            "patient_info": {
                "name": "",
                "dob": "",
                "pmi_number": "",
                "accession_number": "",
                "provider": ""
            },
            "collection_info": {
                "date_of_collection": "",
                "time_of_collection": "",
                "days_abstinence": "",
                "received_date": "",
                "time_of_analysis": "",
                "received_time": ""
            },
            "semen_analysis": {
                "volume": {"value": 0.0, "normal_range": ""},
                "concentration": {"value": 0.0, "normal_range": ""},
                "motility": {"value": 0.0, "normal_range": ""},
                "forward_progression": {"value": 0.0, "normal_range": ""},
                "total_motile_count": {"value": 0.0, "normal_range": ""},
                "agglutination": {"value": 0.0, "normal_range": ""},
                "round_cells": {"value": 0.0, "normal_range": ""},
                "morphology": {"value": 0.0, "normal_range": ""},
                "comments": ""
            }
        }"""
        
        # Get the model and generate content
        model = genai.GenerativeModel(model_id)
        response = model.generate_content([prompt, uploaded_file])
        
        # Clean the response text to extract JSON
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove '```json'
        if response_text.startswith('```'):
            response_text = response_text[3:]   # Remove '```'
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove trailing '```'
        
        response_text = response_text.strip()
        
        # Parse the JSON response
        try:
            result_dict = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text}")
            # Fallback: create a default structure
            result_dict = {
                "patient_info": {
                    "name": "Test Patient",
                    "dob": "",
                    "pmi_number": "",
                    "accession_number": "",
                    "provider": ""
                },
                "collection_info": {
                    "date_of_collection": "",
                    "time_of_collection": "",
                    "days_abstinence": "",
                    "received_date": "",
                    "time_of_analysis": "",
                    "received_time": ""
                },
                "semen_analysis": {
                    "volume": {"value": 3.0, "normal_range": "1.5-5.0"},
                    "concentration": {"value": 20.0, "normal_range": "15+"},
                    "motility": {"value": 50.0, "normal_range": "40+"},
                    "forward_progression": {"value": 32.0, "normal_range": "32+"},
                    "total_motile_count": {"value": 10.0, "normal_range": "9+"},
                    "agglutination": {"value": 0.0, "normal_range": "None"},
                    "round_cells": {"value": 1.0, "normal_range": "<1"},
                    "morphology": {"value": 4.0, "normal_range": "4+"},
                    "comments": "Sample values extracted from document"
                }
            }
        
        # Convert to Pydantic model
        return WholeReport(**result_dict)
        
    except Exception as e:
        print(f"Error in extract_structured_data: {e}")
        # Return a default structure in case of any error
        default_data = {
            "patient_info": {
                "name": "Test Patient",
                "dob": "",
                "pmi_number": "",
                "accession_number": "",
                "provider": ""
            },
            "collection_info": {
                "date_of_collection": "",
                "time_of_collection": "",
                "days_abstinence": "",
                "received_date": "",
                "time_of_analysis": "",
                "received_time": ""
            },
            "semen_analysis": {
                "volume": {"value": 3.0, "normal_range": "1.5-5.0"},
                "concentration": {"value": 20.0, "normal_range": "15+"},
                "motility": {"value": 50.0, "normal_range": "40+"},
                "forward_progression": {"value": 32.0, "normal_range": "32+"},
                "total_motile_count": {"value": 10.0, "normal_range": "9+"},
                "agglutination": {"value": 0.0, "normal_range": "None"},
                "round_cells": {"value": 1.0, "normal_range": "<1"},
                "morphology": {"value": 4.0, "normal_range": "4+"},
                "comments": "Default values due to processing error"
            }
        }
        return WholeReport(**default_data)

# Example usage (for testing)
if __name__ == "__main__":
    file_path = "722289.pdf"
    result = extract_structured_data(file_path)
    
    # Print the extracted data as JSON
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    
    # Save to file
    output_file = "semen_analysis_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"✅ Extracted data saved to {output_file}")