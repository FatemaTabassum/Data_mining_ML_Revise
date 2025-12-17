a. Download the zip file containing the required data in multiple formats.

wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0221EN-SkillsNetwork/labs/module%206/Lab%20-%20Extract%20Transform%20Load/data/source.zip

b. Unzip the downloaded file.
unzip source.zip

The required files are now available in the project folder.

In this lab, you will extract data from CSV, JSON, and XML formats. First, you need to import the appropriate Python libraries to use the relevant functions.

The xml library can be used to parse the information from an .xml file format. The .csv and .json file formats can be read using the pandas library. You will use the pandas library to create a data frame format that will store the extracted data from any file.

To call the correct function for data extraction, you need to access the file format information. For this access, you can use the glob library.

To log the information correctly, you need the date and time information at the point of logging. For this information, you require the datetime package.

While glob, xml, and datetime are inbuilt features of Python, you need to install the pandas library to your IDE.

Install pandas
python3.11 -m pip install pandas

