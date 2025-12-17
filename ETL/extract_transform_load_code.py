import glob
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

# You also require two file paths that will be available globally in the code for all functions. 
# These are transformed_data.csv, to store the final output data that you can load to a database, and log_file.txt, 
# that stores all the logs.

log_file = "log_file.txt"
target_file = "transformed_data.csv"

# Task1: Extraction

def extract_from_csv(file_to_process):
    dataframe = pd.read_csv(file_to_process)
    return dataframe

def extract_from_json(file_to_process):
    dataframe = pd.read_json(file_to_process, lines=True)
    return dataframe

def extract_from_xml(file_to_process):
    dataframe = pd.DataFrame(columns=["name", "height", "weight"])
    tree = ET.parse(file_to_process)
    root = tree.getroot()
    for person in root:
        name = person.find("name").text
        height = float(person.find("height").text)
        weight = float(person.find("weight").text)
        dataframe = pd.concat([dataframe, pd.DataFrame([{"name": name, "height": height, "weight": weight}])], ignore_index=True)
    return dataframe


def extract():
    extracted_data = pd.DataFrame(columns = ['name', 'height', 'weight'])

    # process all csv files, except target files
    for csvfile in glob.glob("*.csv"): # check if the file is not the target file
        if csvfile != target_file:
            extracted_data = pd.concat([extracted_data, pd.DataFrame(extract_from_csv(csvfile))], ignore_index=True)
    
    # process all json files
    for jsonfile in glob.glob("*.json"):
        extracted_data = pd.concat([extracted_data, pd.DataFrame(extract_from_json(jsonfile))], ignore_index=True)

    # process all xml files
    for xmlfile in glob.glob("*.xml"):
        extracted_data = pd.concat([extracted_data, pd.DataFrame(extract_from_xml(xmlfile))], ignore_index=True)

    return extracted_data
                 


# Task 2: Transformation


# it will receive the extracted dataframe as the input. Since the dataframe is in the form of a dictionary with three keys, "name", "height", and "weight", each of them having a list of values, you can apply the transform 
# function on the entire list at one go.
def transform(data):
    '''Convert inches to meters and round off to two decimals
    1 inch is 0.0254 meters '''
    data['height'] = round(data.height * 0.0254, 2)

    '''Convert punds to kilograms and round off to two decimals
    1 pund is 0.45359237 kilograms '''
    data['weight'] = round(data.weight * 0.45359237, 2)

    return data


# Task 3 - Loading

def load_data(target_file, transfomred_data):
    transfomred_data.to_csv(target_file)


# Finally, you need to implement the logging operation to record the progress of the different operations. For this operation, you need to record a message, along with its timestamp, in the log_file.

def log_progress(message):
    timestamp_format = '%Y-%h-%d-%H:%M:%S' # Year-Monthname-Day-Hour-Minute-Second 
    now = datetime.now()
    timestamp = now.strftime(timestamp_format)
    with open(log_file, "a") as f:
        f.write(timestamp + ',' + message + '\n')

# Testing ETL operations and log progress

# Log the initialization of the ETL process
log_progress("ETL Job Started")

# Log the beginning of the Extraction process
log_progress("Extract phase Started")
extracted_data = extract()

# Log the completion of the Extraction process
log_progress("Extract phase Ended")

# Log the beginning of the Transformation process
transfomred_data = transform(extracted_data)
print("Transformed Data")
print(transfomred_data)

# Log the completion of the Transformation process
log_progress("Load phase started")
load_data(target_file, transfomred_data)

# Log the completion of the loading process
log_progress("Load phase Ended")

# Log the completion of the ETL process
log_progress("ETL Job Ended")
