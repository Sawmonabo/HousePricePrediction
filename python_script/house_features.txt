# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Written By:: Sawmon Abossedgh

# Website Conversion from StatePlane (US SURVEY TO METERS) then to Long/Lat Coordinates.
# https://www.earthpoint.us/StatePlane.aspx 
# https://www.melissa.com/v2/lookups/latlngzip4/


#####################################################################################
#################################### Modules ########################################
import pandas as pd
from pandas import read_csv
# from numpy.random import seed

# https://pypi.org/project/stateplane/
import stateplane

# https://pypi.org/project/geocoder/
import geocoder
from geopy.geocoders import Nominatim

# import pickle

import os


#####################################################################################
################################### Set Directory ###################################

if (os.getcwd() == '/Users/SawmonAbo/Desktop/housePricePred'):
    print("We are in the correct working directory.")
    
else:
    os.chdir('/Users/SawmonAbo/Desktop/UCF/housePricePred')


#####################################################################################
################################# Loading DataFrame #################################

# IF YOU HAVE PRE-SAVED a FEATURES DATASET:

# Pickle the database to load our dataset into our current console.
features_dataset = pd.read_pickle('features_df.pkl') 

# To reopen past saved home_data (File Type: document)
# with open("home_data", "rb") as fp:
#     home_data = pickle.load(fp)
    


# NOW SKIP YOU CAN SKIP THE SECTIONS YOU DON"T NEED.

#####################################################################################
################ Data Reading and StatePlane to Lat/Lon Conversion ##################



dataset = read_csv('/Users/SawmonAbo/Desktop/housePricePred/csv/sales.csv')
dataset = dataset.drop(['pid','tid'],axis=1)

   
val_set = read_csv('/Users/SawmonAbo/Desktop/housePricePred/csv/validation_set-6.csv')
val_set = val_set[['price', 'home_size', 'parcel_size', 'beds', 'age', 'pool', 'year', 'cbd_dist', 'x_coord', 'y_coord']]

        
          
dataset = val_set              
            
                

# If we were going to apply this to a model we can randomize the data.
# seed(1234)
# dataset = dataset.sample(len(dataset))


# If we want to partially check specific indicies to reduce runtime we can split
# dataset = dataset[:5000]


# US_Survey -> METERS
unit_converter = (1200/3937)


dataset['x_coord'] = dataset['x_coord'] * unit_converter
dataset['y_coord'] = dataset['y_coord'] * unit_converter


# Rename to stateplane meteric
dataset = dataset.rename(columns={"x_coord" : "E_stateplane"})
dataset = dataset.rename(columns={"y_coord" : "N_stateplane"})


# Re-adjust 'cbd_dist' to match stateplane measure from US Survey -> Meters
dataset['cbd_dist'] = dataset['cbd_dist'] * unit_converter



features_dataset = dataset[['price', 'home_size', 'parcel_size', 'beds', 'age', 'pool', \
                            'year', 'cbd_dist', 'E_stateplane', 'N_stateplane']]


# Convert dataset stateplane columns to arrays to loop through function below.
x_coord = dataset['E_stateplane'].to_numpy()
y_coord = dataset['N_stateplane'].to_numpy()


# Function to convery. The 'fips' parameter is statecode, 'epsg' is Florida's
# European Petroleum Survey Group code, and 'abbr' is flroida east abbreviation.
coordinates = []
for i in range(0, features_dataset.shape[0]):
    coordinates.append(stateplane.to_lonlat(features_dataset['E_stateplane'].iloc[i], \
                    features_dataset['N_stateplane'].iloc[i], fips='0901', abbr='FL_E'))


# Find long/lats and add to dataset featues
long = []
lat = []
for x, y in coordinates:
    long.append(x)
    lat.append(y)
    
    
features_dataset['latitude'] = lat #y 
features_dataset['longitutde'] = long #x
    

# Swapped format (Long/Lat) to (Lat/Long) coordinates to fit function parameter 
# call in geocoder for home data section below.
altered_coordinates = []
for x, y in coordinates:
    altered_coordinates.append((y, x))

#####################################################################################
############################## Lat/Lon to Home Data #################################

# About 1.4hr to run entire 10,000 dataset entries to data points.


# loop over list of lists (coordinates)
# use open street map (osm)

home_data = []
length = len(altered_coordinates)
ctr = 1
geolocator = Nominatim(user_agent="test_app") 

for i in range(len(altered_coordinates)):
    home = geolocator.reverse(altered_coordinates[i])
    home_data.append(home.raw)
    print(f"Obtaining home data from pairs... ({ctr}/{length})")
    ctr = ctr + 1
    
#####################################################################################
################################ Obtaining Zipcodes #################################


# Now that we have home data we can select specific attribute values from
# our list of dictionaries/lists. 

# 'home_data' is a 3D object therefore each sublevel has different information.
# Break down of the variable:  home_data[1][2][3]
#
#   [1] = [indexs]
#   [2] = [address, boundingbox, category, display_name, importance, lat, lon, \
#           licence, osm_id, osm_type, place_id, place_rank, type]
#   [3] = [country_code, county, house_number, postcode, residential, road, \
#           state, suburb ]


# To keep our dataset organized and formated with our other vairables
# we only add the zipcodes from the dataset by extracting it from our home_data.

# To get rid of the 'add-on' or 'plus 4 code' (i.e., 32819-6393) attached 
# to only a few of the zips and to keep it consistent with the majority of
# other zipcodes, we take off the 'add-on'.

zipcode = {}

for i in range(0, len(home_data)):
    
    # Some of the zips return nothing therefore we set those values to 0 for now.
    if (home_data[i]['address'].get('postcode') == None):
        zipcode[i] = 0
    else:
        # Only extracting the zipcode from the string
        zipcode[i] = int((home_data[i]['address']['postcode'])[0:5])
    
features_dataset['zipcode'] = zipcode.values()
features_dataset['zipcode'] = pd.to_numeric(features_dataset['zipcode'])

# Now we need to check our "features_dataset['zipcode']" and make sure it is 
# formatted and correct.

# Problems Found:
    
# 1] Above I mentioned that some of the zips returned where not given and I set 
#    them to zero to make it the same 'type' as our other zips.

# 2] I also spotted and found a zipcode '23786' that was repeated in the dataset.
#    I know this is not an Orlando zipcode so these are incorrect.


#####################################################################################
############################ Fixing Zipcode Values ##################################


# For zipcodes with '23786' its actually supposed to be 32786. We will manually 
# overwrite those indexes.
for i in range(0, len(home_data)):
    if (home_data[i]['address'].get('postcode') == '23786'):
        zipcode[i] = 32786

features_dataset['zipcode'] = zipcode.values()




# Until figure out how to webscrape we will remove the zipcodes with 0 for 
# consistency in models.

# Check shape before deletion.
features_dataset.shape

features_dataset = features_dataset.loc[features_dataset["zipcode"] != 0]

# Check shape after deletion to make sure rows were deleted.
features_dataset.shape



#####################################################################################
################################# Visualization #####################################

pd.set_option('display.max_columns', None)
features_dataset.head()


#####################################################################################
################################## Saving DataFrame #################################

# IF YOU HAVE ADDED NEW FEATURES:

# Save our new constructed features_data set and our home_data to
# avoid runtime of function.

# Pickle function to save the databas.
# features_dataset.to_pickle('features_df.pkl')  

os.chdir('/Users/SawmonAbo/Desktop/housePricePred/DataFrame')


features_dataset.to_pickle('features_wZipcode.pkl')  

# OR
features_dataset.to_pickle('features_validation.pkl')  

# To save home_data as a document for later usage:
# with open("home_data", "wb") as fp:
#     pickle.dump(home_data, fp)

   

#####################################################################################
#                                         END                                       #
#####################################################################################












































#####################################################################################
##################################### TESTING #######################################

# Testing of specific values.

zipcode = []

# (28.3481500989072, -81.42060289087314) - no zipcode
# (28.446300, -81.511430)
test_zips = geocoder.osm((28.4714, -81.589), method='reverse')
house_data = test_zips.raw

test_zips = geocoder.osm((28.4853, -81.5749), method='reverse')
house2_data = test_zips.raw



print(house_data['address']['postcode'])


# To get rid of the add-on or plus 4 codes attatched to zipcode we split it off
zipcode = home_data[1327]['address']['postcode']
zipcode = zipcode[0:5]









#####################################################################################
############################ Webscrapper Function ###################################


# For zipcodes with '23786' its actually supposed to be 32786. We could manually
# fix this by re-writting those spoecific indexes to the correct zipcode.
# But to maintain a dynamic program we will avoid hardcoding. 

# If the zipcode returns as '0' or '23786', use a webscrapper to obtain zipcodes 
# using a secondary website 'https://www.melissa.com/v2/lookups/latlngzip4/'. 



# Let's obtain all columns with zipcodes as 0:
# broken_zipcodes = {}
# broken_zipcodes_indexes = {}
# fixed_zipcodes = {}

# for i in range(0, len(features_dataset.zipcode)):
#     # i = 976
#     if (home_data[i]['address'].get('postcode') == None):
#         # broken_zipcodes_indexes[i] = home_data.index
#         broken_zipcodes[i] = home_data[i]
#         address = broken_zipcodes[i]['display_name']
#         fixed_zipcodes[i] = (geolocator.geocode(address))._address
   # if (home_data[i]['address'].get('postcode') == '23786'):
#         # broken_zipcodes_indexes[i] = home_data.index
#         broken_zipcodes[i] = home_data[i]
#         address = broken_zipcodes[i]['display_name']
#         fixed_zipcodes[i] = (geolocator.geocode(address))._address


# broken_zips = geocoder.osm()        
# fixed_zips = broken_zips.raw 




# For webscrapper
# from bs4 import BeautifulSoup
# import requests

# Not working. Needs work still.

# def find_zip(lat: float, long: float):

#     lat = str(lat)
#     long = str(long)
    
#     url = "https://www.melissa.com/v2/lookups/latlngzip4/?lat=" + lat + "&lng=" + long
    
    
#     r = requests.get(url)
#     data = r.text
#     soup = BeautifulSoup(data, 'html.parser')


#     divs = soup.findAll("table", class_="table table-hover table-striped table-bordered")
#     for div in divs:
#         row = ''
#         rows = div.findAll('tr')
#         for row in rows:
#             if(row.text.find("Postal Code" ) > -1):
#                 zippy = row.text
#                 zipcode = ""
#                 for m in zippy:
#                     if m.isdigit():
#                         zipcode = zipcode + m
#                 zipcode = zipcode[0:5]

#     return zipcode
    

    
    
# search = find_zip(28.4853,81.5749)  
    
# print(search)  


