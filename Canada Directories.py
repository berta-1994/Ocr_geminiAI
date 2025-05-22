# Importing necessary libraries
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

# Verify API key is available
try:
    from config import GOOGLE_API_KEY
except ImportError:
    print("Error: config.py file not found!")
    print("Please create a config.py file and set the GOOGLE_API_KEY variable.")
    exit(1)
# configure google api key for gemini:
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not found!")
    print("Please set it using: $env:GOOGLE_API_KEY='your-api-key' in PowerShell")
    exit(1)
else:
    print("API key found successfully!")
    genai.configure(api_key=GOOGLE_API_KEY)


########################################################
# PART 1: CROPPING IMAGES
########################################################

# Function to display images
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)
    height, width = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()

# Creating paths for directories
input_file_path = 'input'
cropped_folder_path = 'cropped'
temp_file_path = "temp"
ai_files = "ai_files"
canadian_cities = ['Calgary', 'Edmonton', 'Montreal', 'Ottawa', 'Quebec', 'Toronto', 'Vancouver', 'Victoria', 'Winnipeg']

# Create folders with city names in temp, cropped and ai_files directories
for city in canadian_cities:
    os.makedirs(os.path.join(temp_file_path, city), exist_ok=True)
    os.makedirs(os.path.join(cropped_folder_path, city), exist_ok=True)
    os.makedirs(os.path.join(ai_files, city), exist_ok=True)

# Cropping images function
def cropping_images(pdf_path, cropped_folder_path, city):
    # Create city-specific folder if it doesn't exist
    city_folder = os.path.join(cropped_folder_path, city)
    os.makedirs(city_folder, exist_ok=True)

    # import all pages
    pages = convert_from_path(pdf_path, dpi=400)

    for page_number, page in enumerate(pages, start=0):
        print(f"Processing page {page_number + 1}...")

        # Convert the PpmImageFile object to a NumPy array
        image_np = np.array(page)

        # Cropping image
        roi = image_np[650:3900, 400:2800]
        output_path = os.path.join(city_folder, f'page_{page_number + 1}.jpg')
        cv2.imwrite(output_path, roi)
        print(f"Saved to {output_path}")


for city in canadian_cities:
    input_file_path = f'input/{city}/{city}.pdf'
    print(f"Processing {input_file_path}...")
    cropping_images(input_file_path, cropped_folder_path, city)
    print(f"Finished processing {input_file_path}.")

########################################################
# PART 2: EXTRACTING TEXT FROM IMAGES
########################################################

# Read the prompt with UTF-8 encoding
with open('prompt_CA.txt', 'r', encoding='utf-8') as file:
    prompt = file.read()
print(prompt)


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
    cropped_folder_path = os.path.join(cropped_folder_path, city)
    
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
########################################################


########################################################

# Process all cities
for city in canadian_cities:
    print(f"\nProcessing city: {city}")
    all_pages_content = process_images_to_json(city)
    
    # Save the results for each city
    output_file = os.path.join(ai_files, city, f"{city}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_pages_content)
    print(f"Saved results for {city} to {output_file}")
########################################################
# some brackets are not closed, so I will manually fix them
# when loading the json data the consol shows me shich lines have an issue, then it's easy to find them and correct them with the notepad 
#Calgary
calgary_file = os.path.join(ai_files, 'Calgary', 'Calgary_results.json')
with open(calgary_file, 'r', encoding='utf-8') as f:
    calgary_data = json.load(f)

# Edmonton
edmonton_file = os.path.join(ai_files, 'Edmonton', 'Edmonton_results.json')
with open(edmonton_file, 'r', encoding='utf-8') as f:
    edmonton_data = json.load(f)

# Montreal
montreal_file = os.path.join(ai_files, 'Montreal', 'Montreal_results.json')
with open(montreal_file, 'r', encoding='utf-8') as f:
    montreal_data = json.load(f)

# Ottawa
ottawa_file = os.path.join(ai_files,'Ottawa','Ottawa_results.json')
with open(ottawa_file, 'r', encoding = 'utf-8') as f:
    ottawa_data = json.load(f)

#Quebec
quebec_file =  os.path.join(ai_files, 'Quebec', 'Quebec_results.json')
with open(quebec_file, 'r', encoding = 'utf-8') as f:
    quebec_data = json.load(f)

# Toronto
toronto_file = os.path.join(ai_files, 'Toronto', 'Toronto_results.json')
with open(toronto_file, 'r', encoding = 'utf-8') as f:
    toronto_data = json.load(f)

# Vancouver
vancouver_file = os.path.join(ai_files, 'Vancouver', 'Vancouver_results.json')
with open(vancouver_file, 'r', encoding = 'utf-8') as f:
    vancouver_data = json.load(f)
# Victoria
victoria_file = os.path.join(ai_files, 'Victoria', 'Victoria_results.json')
with open(victoria_file, 'r', encoding = 'utf-8') as f:
    victoria_data = json.load(f)
# Winnipeg
winnipeg_file = os.path.join(ai_files, 'Winnipeg', 'Winnipeg_results.json')
with open(winnipeg_file, 'r', encoding = 'utf-8') as f:
    winnipeg_data = json.load(f)




########################################################
# PART 3: CREATING A DATAFRAME
########################################################
# Function to process all JSON files and create a DataFrame
def json_to_dataframe(city):
    json_file = os.path.join(ai_files, city, f"{city}_results.json")
    with open(json_file, 'r', encoding = 'utf-8') as f:
        city_data = json.load(f)

    # Create a DataFrame
    df = pd.DataFrame(city_data)
    
    # Save each city's data to a separate CSV file
    output_csv = os.path.join(ai_files, city, f"{city}.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nSaved all results to {output_csv}")

   
    return df

# creating city dataframes:
for city in canadian_cities:
    city_df = json_to_dataframe(city)


########################################################
# PART 4: CREATING FULL ADDRESS + GEOLOCATING
########################################################

# deleate from column "city" all cities which are not part of our list

for city in canadian_cities:
    city_df = pd.read_csv(os.path.join(ai_files, city, f"{city}.csv"))
    # delete all rows where city is not part of our list
    city_df = city_df[city_df['City'] == city.upper()]
    # save the dataframe
    city_df.to_csv(os.path.join(ai_files, city, f"{city}.csv"), index=False, encoding='utf-8')
    print(f"Updated and saved {city} to {os.path.join(ai_files, city, f"{city}.csv")}")


# Dictionary with city and state
city_states = {'Calgary': 'Alberta', 'Edmonton': 'Alberta', 'Montreal': 'Quebec', 'Ottawa': 'Ontario', 'Quebec': 'Quebec', 'Toronto': 'Ontario', 'Vancouver': 'British Columbia', 'Victoria': 'British Columbia', 'Winnipeg': 'Manitoba'}
country = 'Canada'

def full_address(df, country, city_states, city):
    
    df['full_address'] = ''

    # clean empty values and replace them for na
    df['building'] = df['building'].replace(r'^\s*$', np.nan, regex=True)
    df['address'] = df['address'].replace(r'^\s*$', np.nan, regex=True)


    for index, value in df.iterrows():
        # 1-  When both building and address are present
        if pd.notna(value['building']) and pd.notna(value['address']):
            df.loc[index, 'full_address'] = value['address'] + ', ' + value['building'] + ', ' +  city + ', '+ city_states[city] + ', ' + country
        # 2 - Only building
        elif pd.notna(value['building']):
            df.loc[index, 'full_address'] = value['building'] + ', ' +  city + ', '+ city_states[city] + ', ' + country
        # only address
        elif pd.notna(value['address']):
            df.loc[index, 'full_address'] = value['address'] + ', ' +  city + ', '+ city_states[city] +', ' + country
        else:
            df.loc[index, 'full_address'] = ''
  # Final cleanup: remove any leftover parentheses and strip spaces
    df['full_address'] = df['full_address'].str.strip().str.replace(r'\(.*?\)', '', regex=True)

    return df

for city in canadian_cities:
    city_df = pd.read_csv(os.path.join(ai_files, city, f"{city}.csv"))
    city_fulladr = full_address(city_df, country, city_states, city)
    city_fulladr.to_csv(os.path.join(ai_files, city, f"{city}_fulladr.csv"), index=False, encoding='utf-8')
    print(f"Updated and saved {city} to {os.path.join(ai_files, city, f"{city}_fulladr.csv")}")

######################################################################
# PART 5: GEOLOCATING
######################################################################

def geocoding(df):
  geolocator = Nominatim(user_agent="geo_locator")  # Initialize Nominatim geocoder
  geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)  # Add rate limiting to avoid overloading the service

  for index, row in df.iterrows():
    if pd.notna(row['full_address']) and isinstance(row['full_address'], str) and 'nan' not in row['full_address']:
      location = geocode(row["full_address"])
      if location:
        df.loc[index, "Lat"] = location.latitude # create latitud column
        df.loc[index, "Lon"] = location.longitude # longitude column

      else:
        df.loc[index, "Lat"] = None
        df.loc[index, "Lon"] = None

    else:
      df.loc[index, "Lat"] = None
      df.loc[index, "Lon"] = None

  return df

for city in canadian_cities:
    city_df = pd.read_csv(os.path.join(ai_files, city, f"{city}_fulladr.csv"))
    city_df = geocoding(city_df)
    city_df.to_csv(os.path.join(ai_files, city, f"{city}_geocoded.csv"), index=False, encoding='utf-8')
    print(f"Saved {city} to {os.path.join(ai_files, city, f"{city}_geocoded.csv")}")


