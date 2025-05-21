import os
import numpy as np
import cv2
from pdf2image import convert_from_path
import pandas as pd
import re
from os import path
from PIL import Image
import google.generativeai as genai
import csv
import io
from matplotlib import pyplot as plt
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from google.colab import userdata

# Setting Paths
cwd = os.getcwd()
print(cwd)

# Configure Google Generative AI
# You need to set your API key as an environment variable or load it from a secure file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Get from environment variable
if not GOOGLE_API_KEY:
    try:
        with open('api_key.txt', 'r') as f:  # Or load from a file
            GOOGLE_API_KEY = f.read().strip()
    except FileNotFoundError:
        print("Please set your GOOGLE_API_KEY as an environment variable or in api_key.txt")
        exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Read the prompt with UTF-8 encoding
with open('prompt_CA.txt', 'r', encoding='utf-8') as file:
    prompt = file.read()

# Cropping images function:
def cropping_images(pdf_path, temp_folder_path, cropped_folder_path, city):
    # Create city-specific folder if it doesn't exist
    city_folder = os.path.join(cropped_folder_path, city)
    os.makedirs(city_folder, exist_ok=True)

    # import all pages
    pages = convert_from_path(pdf_path, dpi=400)

    for page_number, page in enumerate(pages, start = 0):
        print(f"Processing page {page_number + 1}...")

        # Convert the PpmImageFile object to a NumPy array
        image_np = np.array(page)

        # Cropping image
        # I will use the same ROI for all images - 
        # Previously I have manually checked which roi was the best for all images
        roi = image_np[350:3050, 300:2200] #
        cv2.imwrite(os.path.join(city_folder, f'page_{page_number + 1}.jpg'), roi)

# Function to convert image to JSON using Google's Generative AI
def image_to_json(image_path, prompt):
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro-vision')
        
        # Generate content
        response = model.generate_content([prompt, image])
        
        # Extract the JSON response
        if response.text:
            # Try to parse the response as JSON to validate it
            try:
                json.loads(response.text)
                return response.text
            except json.JSONDecodeError:
                print(f"Warning: Response for {image_path} is not valid JSON")
                return None
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Function to concatenate multiple JSON responses
def concatenate_json_responses(json_responses):
    if not json_responses:
        return "[]"
    
    # Remove the outer brackets from all responses except the first one
    cleaned_responses = []
    for i, response in enumerate(json_responses):
        if i == 0:
            cleaned_responses.append(response.strip())
        else:
            # Remove the outer brackets and add a comma
            cleaned = response.strip()[1:-1]
            if cleaned:
                cleaned_responses.append(cleaned)
    
    # Join all responses with commas and wrap in brackets
    return "[" + ",".join(cleaned_responses) + "]"

# Function to process images and convert to JSON
def process_images_to_json(city):
    all_json_responses = []
    cropped_folder_path = os.path.join(cropped_file_path, city)
    
    for filename in os.listdir(cropped_folder_path):
        if filename.lower().startswith('page') and filename.lower().endswith('.jpg'):
            image_path = os.path.join(cropped_folder_path, filename)    
            print(f"Processing image: {filename}")    

            json_response = image_to_json(image_path, prompt)
            if json_response:
                all_json_responses.append(json_response)
    
    # Concatenate all JSON responses after processing all images
    all_pages_content = concatenate_json_responses(all_json_responses)
    return all_pages_content

# Process all cities
for city in canadian_cities:
    print(f"\nProcessing city: {city}")
    all_pages_content = process_images_to_json(city)
    
    # Save the results for each city
    output_file = os.path.join(ai_files, city, f"{city}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_pages_content)
    print(f"Saved results for {city} to {output_file}")

# I want to run the function for all files in the input folder
for city in canadian_cities:
    input_file_path = f'input/{city}/{city}.pdf'
    print(f"Processing {input_file_path}...")
    cropping_images(input_file_path, temp_file_path, cropped_file_path, city)
    print(f"Finished processing {input_file_path}.")

# Function to process all JSON files and create a DataFrame
def process_all_json_files():
    all_data = []
    
    # Process each city's JSON file
    for city in canadian_cities:
        json_file = os.path.join(ai_files, city, f"{city}_results.json")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                city_data = json.load(f)
                
            # Add city information to each entry if not present
            for entry in city_data:
                if 'City' not in entry or not entry['City']:
                    entry['City'] = city.upper()
                all_data.append(entry)
                
        except FileNotFoundError:
            print(f"Warning: No results file found for {city}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {city}'s results file")
        except Exception as e:
            print(f"Error processing {city}'s results: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_csv = os.path.join(ai_files, 'all_results.csv')
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nSaved all results to {output_csv}")
    
    return df

# After processing all cities, create the DataFrame
print("\nCreating DataFrame from all results...")
df = process_all_json_files()

# Display basic statistics
print("\nDataFrame Summary:")
print(f"Total entries: {len(df)}")
print("\nEntries per city:")
print(df['City'].value_counts())
print("\nEntries per company type:")
print(df['Company Type'].value_counts()) 