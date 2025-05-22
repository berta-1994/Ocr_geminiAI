# Ocr_geminiAI  

This is an improved version of my previous repository on reading and locating old directories. 
It contains:
- Input folder with folders containing Canadian cities. Inside every folder, there is a PDF scan of one or more pages.
- Cropped folder contains the cropped pdfs at 400 dpi
- ai_files - contain all the text extracted from the scans for each city.
- prompt_CA - is the txt file with the instructions for the AI from Google (Gemini) to read the provided scans. The structure of the prompt is taken from the paper "MULTIMODAL LLMS FOR OCR, OCR POST-CORRECTION, AND NAMED ENTITY RECOGNITION IN HISTORICAL DOCUMENTS" from Gavin Greif, Niclas Griesshaber and Robin Greif.
- config_template -  where you can introduce the GOOGLE_API_KEY to replicate the code
- Canada Directories script. The script is written fully in Python and is divided in 5 parts:
    1. PART 1 - Cropping Images. Cropping the image's margins, I reduce the potential image noise and improve the OCR results.
    2. PART 2 - Extracting text from images. Feeding the prompt to Gemini I am able to extract very cohesive text in a json format. Unfortunately, there are cases where manual imput is needed to correct open brackets and commas.
    3. PART 3 - Creating a df from the json results.
    4. PART 4 - Adding a column with the full address of the company
    5. PART 5 - Geolocation. To geolocate, I have used geopy and Nominatim. It has not returned very promising results. Maybe it's a good suggestion to investigate other options. 

Running the code should be quite straightforward. 
For other scans, the roi (region of interest), which is set to crop the pages, might have to be re-adjusted.
You will have to add your GOOGLE API to the config.py file.




