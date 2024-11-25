import requests
from PIL import Image
from io import BytesIO
import pytesseract
import re
import pandas as pd

# Function to download image from a URL
def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Ensure we notice bad responses
        img = Image.open(BytesIO(response.content))
        return img
    except requests.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None
    except IOError as e:
        print(f"Error opening image from {image_url}: {e}")
        return None

# Ensure Tesseract's path is correctly set
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from image using Tesseract
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# Function to extract entity value (like weight, volume) from text
def extract_entity_value(text, entity_name):
    patterns = {
        "item_weight": r'(\d+\.?\d*)\s*(kg|kilogram|gram|g|pound|lb|ton|tonne|tonne)',
        "item_volume": r'(\d+\.?\d*)\s*(ml|litre|l|gallon)',
        "width": r'(\d+\.?\d*)\s*(mm|millimetre|cm|centimetre|m|meter|foot|inch|yard)',
        "height": r'(\d+\.?\d*)\s*(mm|millimetre|cm|centimetre|m|meter|foot|inch|yard)',
        "depth": r'(\d+\.?\d*)\s*(mm|millimetre|cm|centimetre|m|meter|foot|inch|yard)',
        "wattage": r'(\d+\.?\d*)\s*(watt|kilowatt|kw|kilo watt|w)',
        "voltage": r'(\d+\.?\d*)\s*(volt|kilovolt|kv|v|kilo volt)',
        "maximum_weight_recommendation": r'(\d+\.?\d*)\s*(kg|kilogram|pound|lb|lbs|ton|tonne)'
    }
    
    # Get the pattern for the specific entity
    pattern = patterns.get(entity_name)
    if pattern:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value, unit = match.groups()
            unit_conversion = {
                'kilogram': 'kg', 'kilogram': 'kg', 'gram': 'g', 'g': 'g',
                'pound': 'pound', 'lb': 'pound', 'ton': 'ton', 'tonne': 'ton',
                'ml': 'ml', 'litre': 'l', 'l': 'l', 'gallon': 'gallon',
                'mm': 'millimetre', 'millimetre': 'millimetre', 'cm': 'centimetre', 
                'centimetre': 'centimetre', 'm': 'meter', 'meter': 'meter', 
                'foot': 'foot', 'inch': 'inch', 'yard': 'yard',
                'watt': 'watt', 'kilowatt': 'kilowatt', 'kw': 'kilowatt', 
                'kilo watt': 'kilowatt', 'v': 'volt', 'kilovolt': 'kilovolt', 
                'kv': 'kilovolt', 'kilo volt': 'kilovolt','voltage':'v'
            }
            # Normalize unit
            unit = unit_conversion.get(unit.lower(), unit.lower())
            return f"{value} {unit}"
    
    return ""

# Main function to process the test dataset and generate predictions
def main():
    # Load the test dataset
    test_file = r'C:\Users\Asus\Downloads\amazon dataset\student_resource 3\dataset\test.csv'
    test_df = pd.read_csv(test_file)

    predictions = []

    # Loop over each test sample
    for i, row in test_df.iterrows():
        index = row['index']
        image_link = row['image_link']
        entity_name = row['entity_name']

        print(f"Processing index {index}: {entity_name}")

        # Download the image
        image = download_image(image_link)
        if image is None:
            predictions.append({'index': index, 'prediction': ""})
            continue

        # Extract text from the image
        text = extract_text_from_image(image)
        # print(f"Extracted text for index {index}: {text}")

        # Extract entity value
        entity_value = extract_entity_value(text, entity_name)
        print(f"Extracted value for index {index}: {entity_value}")

        # Append the prediction
        predictions.append({'index': index, 'prediction': entity_value if entity_value else ""})

    # Save the predictions to CSV in the required format
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

# Correct the main block
if __name__ == "__main__":
    main()