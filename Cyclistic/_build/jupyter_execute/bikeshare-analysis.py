#!/usr/bin/env python
# coding: utf-8

# # Case Study: How Does a Bike-Share Navigate Speedy Success?

# ## Guiding Questions of Analysis?
# 
# 1. How do annual members and casual riders use Cyclistic bikes differently?

# ## Data Gathering
# 
# The data gathered for this analysis is sourced from the __[publicly open dataset](https://divvy-tripdata.s3.amazonaws.com/index.html)__ which is provided by the Company **Cyclistic** under the following __[Data Licence Agreement](https://ride.divvybikes.com/data-license-agreement)__.
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as ptly_go
import plotly.offline as ptly_off
import matplotlib.pyplot as plt
import matplotlib.ticker as mat_tick
from pathlib import Path
import math
import seaborn as sns
import locale
import datetime
from IPython.display import Markdown as md
from glob import glob
import heapq
import plotly.io as pio

pio.renderers.default = "notebook"
ptly_off.init_notebook_mode()

locale.setlocale(locale.LC_NUMERIC, 'de_DE')
locale.setlocale(locale.LC_TIME, 'en_US')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# IMPORTING DATA
csv_list = glob('*-tripdata.csv', root_dir="./../raw_data/")

dfs = [
    pd.read_csv(f'./../raw_data/{filename}', index_col = None, parse_dates = ['started_at','ended_at'],
                dtype={'rideable_type': 'category', 'casual_member': 'category'})
    for filename in csv_list
]

df = pd.concat(dfs)


# ## Data Cleaning
# 
# Start by Cleaning the data where there are any N/A values as well as rides that start and end at the exact coordinate values.

# In[3]:


# DROP NaN RIDES
# DROP ROWS IF STATIONS = NaN AND START + END LAT/LONG VALUES ARE SAME
df_cleaned = df.dropna(axis=0, how='any', 
            subset=['start_station_name','start_station_id', 'end_station_name', 'end_station_id'],
            inplace = False)

df_cleaned = df_cleaned.loc[(df_cleaned['start_lat'] != df_cleaned['end_lat']) & 
                            (df_cleaned['start_lng'] != df_cleaned['end_lng'])]

md(f"""Rows drecreased by **{locale.format_string('%d', df.shape[0]-df_cleaned.shape[0], grouping=True)}** 
   from {locale.format_string('%d', df.shape[0], grouping = True)} to {locale.format_string('%d', df_cleaned.shape[0], grouping=True)}.""")


# In[4]:


md(f"""Check for matching observations (start and end stations should have the same amount of ID's and names)

| Station Type | # of ID's | # of names | DIFF |
| --- | --- | --- | --- |
| Start | {df_cleaned['start_station_id'].nunique()} | {df_cleaned['start_station_name'].nunique()} | {abs(df_cleaned['start_station_id'].nunique() - df_cleaned['start_station_name'].nunique())} |
| End | {df_cleaned['end_station_id'].nunique()} | {df_cleaned['end_station_name'].nunique()} | {abs(df_cleaned['end_station_id'].nunique() - df_cleaned['end_station_name'].nunique())} |
""")


# Since there is a clear difference in the numnber of names to stations there seems to be some kind of irregulatory, probably typos. Sincec we do not have an easy way to filter them out lets create a station dataframe with 1:1 id to name and only take the names with the largest number of appearances. afterwards we will replace the names in the cleaned dataframe with the stations that we found to be the most common names.

# In[5]:


# create single df for stations (no difference for start or end)
station_concat = pd.concat([df_cleaned[['start_station_id','start_station_name']]\
            .rename(columns={'start_station_id':'station_id', 'start_station_name': 'station_name'}),
            df_cleaned[['end_station_id','end_station_name']]\
            .rename(columns={'end_station_id':'station_id', 'end_station_name': 'station_name'})])

# groupby stations and count station name
stations_df = station_concat.groupby(['station_id','station_name'])['station_name'].count().rename('count')

# use only speeling with occures most often
station_list = []
for idx, temp_df in stations_df.groupby(level=0):
    station_list.append(temp_df.nlargest(1))

stations_df_cleaned = pd.concat(station_list).to_frame()

stations_df_cleaned = stations_df_cleaned.reset_index(1).drop(columns='count')
stations_df_cleaned

stations_dict = {}
for idx, value in stations_df_cleaned.groupby(level=0):
    stations_dict.update({idx: value['station_name'].item()})


# In[6]:


# change names in df_cleaned

df_cleaned['start_station_name'] = df_cleaned['start_station_id'].apply(lambda x:
                                    stations_dict.get(x))

df_cleaned['end_station_name'] = df_cleaned['end_station_id'].apply(lambda x:
                                    stations_dict.get(x))


# ## Data Preparation
# 
# * Adding formatted columns (hour of ride, weekday, month) to further slice data
# * Calculating distance travlled (point-to-point)

# In[7]:


# CALCULATING TIME RELATED DATA
df_cleaned['hour_of_ride'] = df_cleaned['started_at'].dt.hour
df_cleaned['day_of_ride'] = pd.Categorical(df_cleaned['started_at'].dt.day_name(),
        categories= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
df_cleaned['month_of_ride'] = pd.Categorical(df_cleaned['started_at'].dt.month_name(),
                    categories= ['August', 'September', 'October', 'November', 'December',
                    'January', 'February', 'March', 'April', 'May', 'June', 'July'])
df_cleaned['year_of_ride'] = df_cleaned['started_at'].dt.year

# CALCULATING RIDE TIMES
df_cleaned['length_of_ride_tdelta'] = df_cleaned['ended_at'] - df_cleaned['started_at']
df_cleaned['length_of_ride_s'] = df_cleaned['length_of_ride_tdelta'].apply(lambda x: datetime.timedelta.total_seconds(x))


# ### Calculating the travel distane (Point-to-Point)
# 
# #### The haversine formula or haversine distance
# 
# To calculate the distance on a sphere we can use the formula:
# $$ d = r * acos ( sin(\Phi_{1}) * sin(\Phi_{2}) + cos(\Phi_{1}) * cos(\Phi_{1}) * cos(\Delta\lambda)) $$
# 
# 
# Where:<br>
# r = radius of Sphere, Earth = ~6.371km <br>
# $\Phi_{1}$ = Latitude of start point <br>
# $\Phi_{2}$ = Latitude of end point <br>
# $\Delta\lambda$ = Delta/Difference of longitude between end point and start point
# 
# Source: __[movable-type.co.uk](https://www.movable-type.co.uk/scripts/latlong.html)__

# In[8]:


# DISTANCE Formula
def calc_sphere_dist(start_lat:float, start_lng:float, end_lat:float, end_lng:float, R:int|float=6371000):
    """Calculate distance between two spherical points using Haversine distane.
    Takes coordinates in as angles and returns distance as default for Earth in m. 

    Args:
        start_lat (float): Latitude Point 1 in angle
        start_lng (float): Longitude Point 1 in angle
        end_lat (float): Latitude Point 2 in angle
        end_lng (float): Longitude Point 2 in angle
        R (int or float, optional): Radius of the sphere. Defaults to 6371000 (Earth radius in m).

    Returns:
        float: Returns distance as default for Earth in [m]. 
    """ 
    # CONVERT TO RADIANS
    start_lat_deg = math.radians(start_lat)
    start_lng_deg = math.radians(start_lng)
    end_lat_deg = math.radians(end_lat)
    end_lng_deg = math.radians(end_lng)
    
    # APPLY DISTANCE FORMULA LIKE OUTLINED ABOVE
    d = R * math.acos(
        math.sin(start_lat_deg) * math.sin(end_lat_deg) +
        math.cos(start_lat_deg) * math.cos(end_lat_deg) * math.cos(end_lng_deg - start_lng_deg)
    )

    return d


# In[9]:


# CALCULATE RIDE DISTANCE (POINT-TO-POINT)
df_cleaned['dist_ride'] = df_cleaned.apply(lambda x: 
    calc_sphere_dist(x['start_lat'], x['start_lng'], x['end_lat'], x['end_lng']), axis = 1)


# ## Data Analysis
# 
# Get a feel for the dataset with calculating some high-level statistics:
# * Average, Median, Max and Min for the ride time as well as the ride distance
# * most commonly observed time of day, weekday and month of ride

# In[10]:


# FUNCTION TO TURN TIMEDELTA INTO A TIME STRING (NOT BUILT IN FUNCTION)
# seen on https://stackoverflow.com/questions/8906926/formatting-timedelta-objects
from string import Template

def strfdelta(tdelta:datetime.timedelta, fmt:str="%H:%M:%S"):
    """Format a timedelta object into a time string like strf function

    Args:
        tdelta (timedelta object): Timedelta object to be converted to a time string
        fmt (str, optional): Format string to specify desired output format. Default: "%H:%M:%S"

    Returns:
        str: Returns string in timeformat.
    """ 
    
    class DeltaTemplate(Template):
        delimiter = "%"
        
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["H"] = str(d["H"]).zfill(2)
    d["M"], d["S"] = divmod(rem, 60)
    d["M"] = str(d["M"]).zfill(2)
    d["S"] = str(d["S"]).zfill(2)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


# ### High Level Stats - 1

# In[11]:


# CALCULATE DESCRIPTIVE STATISTICS
avg_ride_timedelta = strfdelta(df_cleaned['length_of_ride_tdelta'].mean(), '%H:%M:%S')
median_ride_timedelta = strfdelta(df_cleaned['length_of_ride_tdelta'].median(), '%H:%M:%S')
max_ride_timedelta = strfdelta(df_cleaned['length_of_ride_tdelta'].max(), '%D days %H:%M:%S')
min_ride_timedelta = strfdelta(df_cleaned['length_of_ride_tdelta'].min(), '%D days %H:%M:%S')

avg_ride_dist = locale.format_string('%d', round(df_cleaned['dist_ride'].mean(), 2), grouping=True)
median_ride_dist = locale.format_string('%d',round(df_cleaned['dist_ride'].median(), 2), grouping=True)
max_ride_dist = locale.format_string('%d',round(df_cleaned['dist_ride'].max(), 2), grouping=True)
min_ride_dist = locale.format_string('%d',round(df_cleaned['dist_ride'].min(), 2), grouping=True)

most_common_hour = df_cleaned['hour_of_ride'].mode()[0]
most_common_day = df_cleaned['day_of_ride'].mode()[0]
most_common_month = df_cleaned['month_of_ride'].mode()[0]
most_common_start_station = df_cleaned['start_station_name'].mode()[0]
most_common_end_station = df_cleaned['end_station_name'].mode()[0]

dist_time_corr = df_cleaned['length_of_ride_s'].corr(df_cleaned['dist_ride'])

df_sorted = df_cleaned.sort_values(by=['dist_ride'], ascending=False)

count_rides = df_cleaned['ride_id'].count()


# In[12]:


md(f"""The high-level statistics shows the following results:<br>
Number of observations: **{locale.format_string('%d', count_rides, grouping = True)}**
<br>
Correlation between time and distance is:
r = **{round(dist_time_corr,5)}**
<br>

| Variable | Average | Median | Max | Min | 
|----------| :-: | :-: | :-: | :-: |
| ride length [t] | {avg_ride_timedelta} | {median_ride_timedelta} | {max_ride_timedelta} | {min_ride_timedelta} |
| ride distance [m] | {avg_ride_dist} | {median_ride_dist} | {max_ride_dist} | {min_ride_dist} | 

<br>

| Variable | Mode (most common) |
| --- | :-: |
| hour of ride | {most_common_hour} |
| day of ride | {most_common_day} |
| month of ride | {most_common_month} |
| start station | {most_common_start_station} |
| end station | {most_common_end_station} |



<br>
The data shows that there is a Min value of ride time that is negative, indicating that the end time was before the start time. The Min value of ride dist also shows 0 distance travelled (although that was supposed to be filtered out by same Lat + Long before).

Also there seems to be a stark outlier with over {locale. format_string('%d', round(int(max_ride_dist.replace(".",""))/1000,1), grouping=True)}km travelled. On further inspection this was done with an electrik bike however the length of the bike ride was too short to make sense. Add new rule to filter out rides > 200km

Those data points are invalid and need to be filtered out before proceeding with more analysis.
 """)


# ### Data Cleaning - Step 1
# 
# Filter out rows with the following conditios:
# * length of travel as timedelta < 0 (negative) OR
# * ride distane in m is <= 0 OR
# * ride distance in m is > 200.000

# In[13]:


# MORE CLEANING ACCORDING TO CRITERIA ABOVE
df_cleaned_v2 = df_cleaned.loc[(df_cleaned['dist_ride'] > 0) & 
                               (df_cleaned['length_of_ride_tdelta']>datetime.timedelta(seconds=1)) &
                               (df_cleaned['dist_ride'] < 200000)]


# In[14]:


md(f"""Rows drecreased by **{locale.format_string('%d', df_cleaned.shape[0]-df_cleaned_v2.shape[0], grouping = True)}** 
   from {locale.format_string('%d', df_cleaned.shape[0], grouping = True)} 
   to {locale.format_string('%d', df_cleaned_v2.shape[0], grouping = True)}.""")


# ### High Level Stats - 2

# In[15]:


# RE-CALCULATE DESCRIPTIVE STAISTICS
avg_ride_timedelta_v2 = strfdelta(df_cleaned_v2['length_of_ride_tdelta'].mean(), '%H:%M:%S')
median_ride_timedelta_v2 = strfdelta(df_cleaned_v2['length_of_ride_tdelta'].median(), '%H:%M:%S')
max_ride_timedelta_v2 = strfdelta(df_cleaned_v2['length_of_ride_tdelta'].max(), '%D days %H:%M:%S')
min_ride_timedelta_v2 = strfdelta(df_cleaned_v2['length_of_ride_tdelta'].min(), '%D days %H:%M:%S')

avg_ride_dist_v2 = locale.format_string('%d', round(df_cleaned_v2['dist_ride'].mean(), 2), grouping=True)
median_ride_dist_v2 = locale.format_string('%d',round(df_cleaned_v2['dist_ride'].median(), 2), grouping=True)
max_ride_dist_v2 = locale.format_string('%d',round(df_cleaned_v2['dist_ride'].max(), 2), grouping=True)
min_ride_dist_v2 = locale.format_string('%d',round(df_cleaned_v2['dist_ride'].min(), 2), grouping=True)

most_common_hour_v2 = df_cleaned_v2['hour_of_ride'].mode()[0]
most_common_day_v2 = df_cleaned_v2['day_of_ride'].mode()[0]
most_common_month_v2 = df_cleaned_v2['month_of_ride'].mode()[0]
most_common_start_station_v2 = df_cleaned_v2['start_station_name'].mode()[0]
most_common_end_station_v2 = df_cleaned_v2['end_station_name'].mode()[0]

dist_time_corr_v2 = df_cleaned_v2['length_of_ride_s'].corr(df_cleaned_v2['dist_ride'])

df_max_ride_dist = df_cleaned_v2.loc[(df_cleaned_v2['dist_ride'] == df_cleaned_v2['dist_ride'].max())]

count_rides_v2 = df_cleaned_v2['ride_id'].count()


# In[16]:


md(f"""The high-level statistics shows the following results:<br>
Number of observations: {locale.format_string('%d', count_rides_v2, grouping = True)}
<br>
Correlation between time and distance is:
r = {round(dist_time_corr_v2,5)} 

| Variable | Average | Median | Max | Min | 
|----------| :-: | :-: | :-: | :-: |
| ride length [t] | {avg_ride_timedelta_v2} | {median_ride_timedelta_v2} | {max_ride_timedelta_v2} | {min_ride_timedelta_v2} |
| ride distance [m] | {avg_ride_dist_v2} | {median_ride_dist_v2} | {max_ride_dist_v2} | {min_ride_dist_v2} | 

<br>

| Time of ride | Mode (most common) |
| --- | :-: |
| hour of ride | {most_common_hour_v2} |
| day of ride | {most_common_day_v2} |
| month of ride | {most_common_month_v2} |
| start station | {most_common_start_station_v2} |
| end station | {most_common_end_station_v2} |

Although there all rides with same start coordinates have been filtered out, there are still rides with no or very little
distance travelled. Also there are rides that only last a few seconds. To get a beter understanding of "real" rides, e.g. rides
that are used to travel somewhere and are not "accidentally" unlocked, a new rule will be applied: rides have to be farther than 100m
and longer than 1 min.
 """)


# ### Data Cleaning - Step 2

# In[17]:


# NARROWING DOWN DATA SET TO MATCH ACTUAL DEMAND DIST > 100m and RIDE TIME > 1 MIN
df_cleaned_v3 = df_cleaned_v2.loc[(df_cleaned_v2['dist_ride'] > 100) & 
                               (df_cleaned_v2['length_of_ride_tdelta']>datetime.timedelta(minutes=1))]


# In[18]:


md(f"""Rows drecreased by **{locale.format_string('%d', df_cleaned_v2.shape[0]-df_cleaned_v3.shape[0], grouping = True)}** 
from {locale.format_string('%d', df_cleaned_v2.shape[0], grouping = True)} 
to {locale.format_string('%d', df_cleaned_v3.shape[0], grouping = True)}.<br>
   
In total Rows decreased by **{locale.format_string('%d', df.shape[0]-df_cleaned_v3.shape[0], grouping = True)}** from original dataset.
""")


# In[19]:


# CREATE SUBCATEGORY FOR DIST TRAVLLED FROM ULTRA-SHORT TO EXTRA-LONG
def map_distance(x):
    map = {'< 500': 'Short', '< 1000': 'Short-Medium', '< 2500': 'Medium',
           '< 5000': 'Medium-Long', '<10000': 'Long', '>= 10000': 'Extra-Long'}
    for key, value in map.items():
        if eval(f'{x} {key}'):
            return value
            break
            
df_cleaned_v3['dist_class'] = pd.Categorical(df_cleaned_v3['dist_ride'].apply(lambda x:map_distance(x)),
        categories=['Short','Short-Medium', 'Medium', 'Medium-Long', 'Long', 'Extra-Long'])


# In[20]:


# EpxORTING CLEANED DATA TO CSV
df_cleaned_v3.to_csv('.\..\data\cleaned_data.csv', float_format="%.2f", index=False)


# #### IMPROT DF for quick restart and PURGE OBSOLETE DATA

# In[21]:


# import df_cleaned_v3
if 'df_cleaned_v3' not in globals():
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as ptly_go
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mat_tick
    from pathlib import Path
    import math
    import seaborn as sns
    import heapq
    import nbformat as nbf
    import locale
    import datetime
    from IPython.display import Markdown as md

    locale.setlocale(locale.LC_NUMERIC, 'de_DE')
    locale.setlocale(locale.LC_TIME, 'en_US')
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    cols = ['ride_id', 'rideable_type', 'started_at', 'ended_at', 'start_station_name', 'start_station_id',
        'end_station_name', 'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual',
        'hour_of_ride', 'day_of_ride', 'month_of_ride', 'year_of_ride', 'length_of_ride_tdelta', 'length_of_ride_s',
        'dist_ride', 'dist_class']

    df_cleaned_v3 = pd.read_csv('./data/cleaned_data.csv',
            parse_dates = ['started_at','ended_at','length_of_ride_tdelta'], usecols=cols, dtype={
        # 'month_of_ride': 'category',
        # 'day_of_ride': 'category',
        # 'rideable_type': 'category',
        'start_lat': 'float64',
        'start_lng': 'float64',
        'end_lat': 'float64',
        'end_lng': 'float64',
        # 'member_casual': 'category',
        # 'ride_timedelta': 'timedelta64[ns]',
        'length_of_ride_s': 'float64',
        'dist_ride': 'float64',
        # 'dist_class': 'catgory',
        })
    
    df_cleaned_v3['length_of_ride_tdelta'] = pd.to_timedelta(df_cleaned_v3['length_of_ride_tdelta'])
    for col in ['month_of_ride', 'day_of_ride', 'rideable_type', 'member_casual', 'dist_class']:
        df_cleaned_v3[col] = df_cleaned_v3[col].astype('category')
        
    df_cleaned_v3['day_of_ride'] = pd.Categorical(df_cleaned_v3['started_at'].dt.day_name(),
        categories= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
else:
    globals().pop('dfs', None);
    globals().pop('df_sorted', None);
    globals().pop('df', None);
    globals().pop('df_cleaned', None);
    globals().pop('df_cleaned_v2', None);

df_cleaned_v3['day_of_ride'] = pd.Categorical(df_cleaned_v3['day_of_ride'],
        categories= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

df_cleaned_v3['month_of_ride'] = pd.Categorical(df_cleaned_v3['month_of_ride'],
                    categories= ['August', 'September', 'October', 'November', 'December',
                    'January', 'February', 'March', 'April', 'May', 'June', 'July'])

df_cleaned_v3['dist_class'] = pd.Categorical(df_cleaned_v3['dist_class'],
        categories=['Short','Short-Medium', 'Medium', 'Medium-Long', 'Long', 'Extra-Long'])

df_cleaned_v3['rideable_type'] = df_cleaned_v3['rideable_type'].apply(lambda x: x.replace('_',' '))
df_cleaned_v3[['member_casual', 'rideable_type']] = df_cleaned_v3[['member_casual', 'rideable_type']] \
    .astype(str).apply(lambda col: col.str.title())


# FUNCTION TO TURN TIMEDELTA INTO A TIME STRING (NOT BUILT IN FUNCTION)
# seen on https://stackoverflow.com/questions/8906926/formatting-timedelta-objects
from string import Template
class DeltaTemplate(Template):
    delimiter = "%"

def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["H"] = str(d["H"]).zfill(2)
    d["M"], d["S"] = divmod(rem, 60)
    d["M"] = str(d["M"]).zfill(2)
    d["S"] = str(d["S"]).zfill(2)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)   


# ### High Level Stats - 3

# In[22]:


# RE-CALCULATE DESCRIPTIVE STAISTICS
avg_ride_timedelta_v3 = strfdelta(df_cleaned_v3['length_of_ride_tdelta'].mean(), '%H:%M:%S')
median_ride_timedelta_v3 = strfdelta(df_cleaned_v3['length_of_ride_tdelta'].median(), '%H:%M:%S')
max_ride_timedelta_v3 = strfdelta(df_cleaned_v3['length_of_ride_tdelta'].max(), '%D days %H:%M:%S')
min_ride_timedelta_v3 = strfdelta(df_cleaned_v3['length_of_ride_tdelta'].min(), '%D days %H:%M:%S')

avg_ride_dist_v3 = locale.format_string('%d', round(df_cleaned_v3['dist_ride'].mean(), 2), grouping=True)
median_ride_dist_v3 = locale.format_string('%d',round(df_cleaned_v3['dist_ride'].median(), 2), grouping=True)
max_ride_dist_v3 = locale.format_string('%d',round(df_cleaned_v3['dist_ride'].max(), 2), grouping=True)
min_ride_dist_v3 = locale.format_string('%d',round(df_cleaned_v3['dist_ride'].min(), 2), grouping=True)

most_common_hour_v3 = df_cleaned_v3['hour_of_ride'].mode()[0]
most_common_day_v3 = df_cleaned_v3['day_of_ride'].mode()[0]
most_common_month_v3 = df_cleaned_v3['month_of_ride'].mode()[0]
most_common_start_station_v3 = df_cleaned_v3['start_station_name'].mode()[0]
most_common_end_station_v3 = df_cleaned_v3['end_station_name'].mode()[0]

dist_time_corr_v3 = df_cleaned_v3['length_of_ride_s'].corr(df_cleaned_v3['dist_ride'])

count_rides_v3= df_cleaned_v3['ride_id'].count()


# In[23]:


md(f"""The high-level statistics shows the following results:
<br>
Number of observations: **{locale.format_string('%d', count_rides_v3, grouping = True)}**
<br>
Correlation between time and distance is:
r = **{round(dist_time_corr_v3,5)}**
   
| Variable | Average | Median | Max | Min | 
|----------| :-: | :-: | :-: | :-: |
| ride length [t] | {avg_ride_timedelta_v3} | {median_ride_timedelta_v3} | {max_ride_timedelta_v3} | {min_ride_timedelta_v3} |
| ride distance [m] | {avg_ride_dist_v3} | {median_ride_dist_v3} | {max_ride_dist_v3} | {min_ride_dist_v3} | 

<br>

| Time of ride | Mode (most common) |
| --- | :-: |
| hour of ride | {most_common_hour_v3} |
| day of ride | {most_common_day_v3} |
| month of ride | {most_common_month_v3} | 
| start station | {most_common_start_station_v3} |
| end station | {most_common_end_station_v3} |
 """)


# ## Deep Dive - Statistics via Pivot
# 
# Taking a deeper look into the cleaned data and aggregating the data into different pivots
# 
# ### Pivot - Member-Type

# In[24]:


# GROUPING STATISTICS
df_groupby_member = df_cleaned_v3.groupby('member_casual')\
    .apply(lambda df: pd.Series({
    'count_of_rides': locale.format_string('%d', df['ride_id'].count(), grouping = True),
    'r' : round(df['length_of_ride_s'].corr(df['dist_ride']), 5),
    
    'common_hour': df['hour_of_ride'].mode()[0],
    'most_common_day': df['day_of_ride'].mode()[0],
    'most_common_month': df['month_of_ride'].mode()[0],
        
    'avg_time': strfdelta(df['length_of_ride_tdelta'].mean(), '%H:%M:%S'),
    'median_time': strfdelta(df['length_of_ride_tdelta'].median(), '%H:%M:%S'),
    'std_time': strfdelta(df['length_of_ride_tdelta'].std(), '%H:%M:%S'),
    'max_time': df['length_of_ride_tdelta'].max(),
    'min_time': strfdelta(df['length_of_ride_tdelta'].min(), '%H:%M:%S'),
    
    'avg_dist': round(df['dist_ride'].mean(), 0),
    'median_dist': round(df['dist_ride'].median(), 0),
    'std_dist': round(df['dist_ride'].std(), 0),
    'max_dist': round(df['dist_ride'].max(), 0),
    'min_dist': round(df['dist_ride'].min(), 0),
    }))

df_groupby_member


# ### Pivot - Rideable-Type

# In[25]:


# GROUPING STATISTICS
df_groupby_ride = df_cleaned_v3.groupby('rideable_type') \
    .apply(lambda df: pd.Series({
    'count_of_rides': locale.format_string('%d', df['ride_id'].count(), grouping = True),
    'r' : round(df['length_of_ride_s'].corr(df['dist_ride']), 5),
    
    'common_hour': df['hour_of_ride'].mode()[0],
    'most_common_day': df['day_of_ride'].mode()[0],
        
    'avg_time': strfdelta(df['length_of_ride_tdelta'].mean(), '%H:%M:%S'),
    'median_time': strfdelta(df['length_of_ride_tdelta'].median(), '%H:%M:%S'),
    'std_time': strfdelta(df['length_of_ride_tdelta'].std(), '%H:%M:%S'),
    'max_time': df['length_of_ride_tdelta'].max(),
    'min_time': strfdelta(df['length_of_ride_tdelta'].min(), '%H:%M:%S'),
    
    'avg_dist': round(df['dist_ride'].mean(), 0),
    'median_dist': round(df['dist_ride'].median(), 0),
    'std_dist': round(df['dist_ride'].std(), 0),
    'max_dist': round(df['dist_ride'].max(), 0),
    'min_dist': round(df['dist_ride'].min(), 0),
    }))

df_groupby_ride    


# ### Pivot - Month

# In[26]:


# GROUPING STATISTICS
df_groupby_month = df_cleaned_v3.groupby('month_of_ride')\
    .apply(lambda df: pd.Series({
    'count_of_rides': locale.format_string('%d', df['ride_id'].count(), grouping = True),
    'r' : round(df['length_of_ride_s'].corr(df['dist_ride']), 5),
    
    'common_hour': df['hour_of_ride'].mode()[0],
    'most_common_day': df['day_of_ride'].mode()[0],
        
    'avg_time': strfdelta(df['length_of_ride_tdelta'].mean(), '%H:%M:%S'),
    'median_time': strfdelta(df['length_of_ride_tdelta'].median(), '%H:%M:%S'),
    'std_time': strfdelta(df['length_of_ride_tdelta'].std(), '%H:%M:%S'),
    'max_time': df['length_of_ride_tdelta'].max(),
    'min_time': strfdelta(df['length_of_ride_tdelta'].min(), '%H:%M:%S'),
    
    'avg_dist': round(df['dist_ride'].mean(), 0),
    'median_dist': round(df['dist_ride'].median(), 0),
    'std_dist': round(df['dist_ride'].std(), 0),
    'max_dist': round(df['dist_ride'].max(), 0),
    'min_dist': round(df['dist_ride'].min(), 0),
    }))

df_groupby_month   


# ### Pivot - Member-Type & Month

# In[27]:


# GROUPING STATISTICS
df_groupby_member_month = df_cleaned_v3.groupby(['member_casual','month_of_ride']) \
    .apply(lambda df: pd.Series({
    'count_of_rides': locale.format_string('%d', df['ride_id'].count(), grouping = True),
    'r' : round(df['length_of_ride_s'].corr(df['dist_ride']), 5),
    
    'common_hour': df['hour_of_ride'].mode()[0],
    'most_common_day': df['day_of_ride'].mode()[0],
        
    'avg_time': strfdelta(df['length_of_ride_tdelta'].mean(), '%H:%M:%S'),
    'median_time': strfdelta(df['length_of_ride_tdelta'].median(), '%H:%M:%S'),
    'std_time': strfdelta(df['length_of_ride_tdelta'].std(), '%H:%M:%S'),
    'max_time': df['length_of_ride_tdelta'].max(),
    'min_time': strfdelta(df['length_of_ride_tdelta'].min(), '%H:%M:%S'),
    
    'avg_dist': round(df['dist_ride'].mean(), 0),
    'median_dist': round(df['dist_ride'].median(), 0),
    'std_dist': round(df['dist_ride'].std(), 0),
    'max_dist': round(df['dist_ride'].max(), 0),
    'min_dist': round(df['dist_ride'].min(), 0),
    }))


df_groupby_member_month   


# ### Pivot - Member-Type & Day

# In[28]:


# GROUPING STATISTICS
df_groupby_member_day = df_cleaned_v3.groupby(['member_casual','day_of_ride']) \
    .apply(lambda df: pd.Series({
    'count_of_rides': locale.format_string('%d', df['ride_id'].count(), grouping = True),
    'r' : round(df['length_of_ride_s'].corr(df['dist_ride']), 5),
    
    'common_hour': df['hour_of_ride'].mode()[0],
        
    'avg_time': strfdelta(df['length_of_ride_tdelta'].mean(), '%H:%M:%S'),
    'median_time': strfdelta(df['length_of_ride_tdelta'].median(), '%H:%M:%S'),
    'std_time': strfdelta(df['length_of_ride_tdelta'].std(), '%H:%M:%S'),
    'max_time': df['length_of_ride_tdelta'].max(),
    'min_time': strfdelta(df['length_of_ride_tdelta'].min(), '%H:%M:%S'),
    
    'avg_dist': round(df['dist_ride'].mean(), 0),
    'median_dist': round(df['dist_ride'].median(), 0),
    'std_dist': round(df['dist_ride'].std(), 0),
    'max_dist': round(df['dist_ride'].max(), 0),
    'min_dist': round(df['dist_ride'].min(), 0),
    }))

df_groupby_member_day    


# ### Pivot - Member-Type & Ride-Type & Month

# In[29]:


# GROUPING STATISTICS
df_groupby_member_ride_month = df_cleaned_v3.groupby(['member_casual', 
    'rideable_type','month_of_ride']).apply(lambda df: pd.Series({
    'count_of_rides': locale.format_string('%d', df['ride_id'].count(), grouping = True),
    'r' : round(df['length_of_ride_s'].corr(df['dist_ride']), 5),
    
    'common_hour': df['hour_of_ride'].mode()[0],
        
    'avg_time': strfdelta(df['length_of_ride_tdelta'].mean(), '%H:%M:%S'),
    'median_time': strfdelta(df['length_of_ride_tdelta'].median(), '%H:%M:%S'),
    'std_time': strfdelta(df['length_of_ride_tdelta'].std(), '%H:%M:%S'),
    'max_time': df['length_of_ride_tdelta'].max(),
    'min_time': strfdelta(df['length_of_ride_tdelta'].min(), '%H:%M:%S'),
    
    'avg_dist': round(df['dist_ride'].mean(), 0),
    'median_dist': round(df['dist_ride'].median(), 0),
    'std_dist': round(df['dist_ride'].std(), 0),
    'max_dist': round(df['dist_ride'].max(), 0),
    'min_dist': round(df['dist_ride'].min(), 0),
    }))

df_groupby_member_ride_month


# ### Pivot - Member-Type & most used Station

# In[30]:


station_concat = pd.concat([df_cleaned_v3[['member_casual','start_station_id','start_station_name','day_of_ride','month_of_ride','rideable_type']]\
            .rename(columns={'start_station_id':'station_id', 'start_station_name': 'station_name'}),
            df_cleaned_v3[['member_casual','end_station_id','end_station_name','day_of_ride','month_of_ride','rideable_type']]\
            .rename(columns={'end_station_id':'station_id', 'end_station_name': 'station_name'})])

most_common_station = station_concat.groupby(['member_casual','station_id', 'station_name'])['station_id'].count().\
    rename('no_of_rides').sort_values(ascending=False)

dfs = []
for idx, temp_df in most_common_station.groupby(level=0):
    dfs.append(temp_df.nlargest(5))

most_common_station = pd.concat(dfs).to_frame()

most_common_station


# ### Pivot - Member-Type & Month & most used Station

# In[31]:


df_temp = station_concat.groupby(['member_casual','month_of_ride','station_id','station_name'])['station_id'].count().\
    rename('no_of_rides').sort_values(ascending=False)

dfs = []
for idx, temp_df in df_temp.groupby(level=[0,1]):
    dfs.append(temp_df.nlargest(2))

most_common_station_month = pd.concat(dfs).to_frame()

most_common_station_month


# ### Pivot - Member-Type & Day & most used Station

# In[32]:


df_temp = station_concat.groupby(['member_casual','day_of_ride','station_id','station_name'])['station_id'].count().\
    rename('no_of_rides').sort_values(ascending=False)


dfs = []
for idx, temp_df in df_temp.groupby(level=[0,1]):
    dfs.append(temp_df.nlargest(3))

most_common_station_weekday = pd.concat(dfs).to_frame()

most_common_station_weekday


# ### Pivot - Member-Type & Ride-Type & most used Station

# In[33]:


df_temp = station_concat.groupby(['member_casual','rideable_type','station_id','station_name'])['station_id'].count().\
    rename('no_of_rides').sort_values(ascending=False)


dfs = []
for idx, temp_df in df_temp.groupby(level=[0,1]):
    dfs.append(temp_df.nlargest(5))

most_common_station_ridetype = pd.concat(dfs).to_frame()

most_common_station_ridetype


# ### EXPORT PIVOTS TO EXCEL

# In[34]:


# EXPORTING CLEANED DATA TO XLSX
float_format = '%.2f'
file_path = '.\..\data\cleaned_data.xlsx'
with pd.ExcelWriter(file_path) as writer:  
    # df_cleaned_v3.to_excel(writer, sheet_name='cleaned_data', float_format=float_format)
    df_groupby_member.to_excel(writer, sheet_name='pivot_member', float_format=float_format)
    del df_groupby_member
    df_groupby_month.to_excel(writer, sheet_name='pivot_month', float_format=float_format)
    del df_groupby_month
    df_groupby_ride.to_excel(writer, sheet_name='pivot_ridetype', float_format=float_format)
    del df_groupby_ride
    df_groupby_member_day.to_excel(writer, sheet_name='pivot_member_day', float_format=float_format)
    del df_groupby_member_day
    df_groupby_member_month.to_excel(writer, sheet_name='pivot_member_month', float_format=float_format)
    del df_groupby_member_month
    df_groupby_member_ride_month.to_excel(writer, sheet_name='pivot_member_ridetype_month', float_format=float_format)
    del df_groupby_member_ride_month
    most_common_station.to_excel(writer, sheet_name='mode_station', float_format=float_format)
    # del most_common_station - will be needed for plot
    most_common_station_month.to_excel(writer, sheet_name='mode_station_month', float_format=float_format)
    del most_common_station_month
    most_common_station_weekday.to_excel(writer, sheet_name='mode_station_wday', float_format=float_format)
    del most_common_station_weekday
    most_common_station_ridetype.to_excel(writer, sheet_name='mode_station_ridetype', float_format=float_format)
    del most_common_station_ridetype
    


# In[35]:


# STYLING PLOTS
plt.style.use('seaborn-notebook')

plt.rcParams.update({
    'xtick.labelsize' : 26,
    'ytick.labelsize' : 26,
    "savefig.dpi" : 300,
    "axes.titlesize" : 46,
    "axes.titleweight": "bold",
    "axes.labelsize" : 30,
    "axes.labelweight" : "bold",
    "lines.linewidth" : 3,
    "lines.markersize" : 10,
    "figure.figsize" : [24, 18],
    "figure.titlesize" : 46,
    "figure.titleweight" : 900,
    "font.size" : 22,
    "legend.fontsize": 26,
    "legend.title_fontsize": 32,
})

palette = sns.color_palette("tab10")
blue =  palette[0]
orange = palette[1]

color_dict = dict(
  Casual = f"rgba({blue[0]*255}, {blue[1]*255}, {blue[2]*255}, 1)",
  Member = f"rgba({orange[0]*255}, {orange[1]*255}, {orange[2]*255}, 1)",
)


# ## Deep Dive - Visualization
# 
# Let's first take a look at the distribution of rides used by member and casual riders as well as what type of ride they use
# 
# ### Distribution of rides

# In[36]:


df_member_type_distri = df_cleaned_v3.groupby(['member_casual','rideable_type'])['ride_id'].count().rename('no_of_rides').reset_index()
df_member_type_distri['Total'] = " "
# df_member_type_distri['rideable_type'] = df_member_type_distri['rideable_type'].apply(lambda x: x.replace('_',' '))
# df_member_type_distri[['member_casual', 'rideable_type']] = df_member_type_distri[['member_casual', 'rideable_type']] \
#     .astype(str).apply(lambda col: col.str.title())

fig = px.sunburst(df_member_type_distri, path=['Total','member_casual', 'rideable_type'],
            values = 'no_of_rides',
            maxdepth=3,
            template="plotly_white",
            color_discrete_sequence=['#E24A33', '#348ABD'],
            height=900
)

fig.update_traces(textinfo= "label+percent entry")

fig.update_layout(title_text="<b>Distribution by Ride-Type and Member</b>",title_x=0.5,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
)

fig.write_image('./../pictures/sunburst_member_ridetype.png')
fig.show()


# In[37]:


md(f"""
Both parties seem to have a preference for the class bike instead of an electric bike. This preference is strongest for the members who prefer
to use the classic bike for {df_member_type_distri.loc[(df_member_type_distri['member_casual']=='Member') & (df_member_type_distri['rideable_type']=='Classic Bike')]['no_of_rides'].sum() / 
df_member_type_distri.loc[(df_member_type_distri['member_casual']=='Member')]['no_of_rides'].sum():.1%} of their rides vs casual riders who use it for 
{df_member_type_distri.loc[(df_member_type_distri['member_casual']=='Casual') & (df_member_type_distri['rideable_type']=='Classic Bike')]['no_of_rides'].sum() / 
df_member_type_distri.loc[(df_member_type_distri['member_casual']=='Casual')]['no_of_rides'].sum():.1%}  of their rides.

Also casual riders are the only ones who use the option of the docked bike.
""")


# ### Ride distance
# 
# Next we will look at a quick distribution of the distance classified into 6 categories from Short to X-Long (see annotations).

# In[38]:


df_dist_class_distri = df_cleaned_v3.groupby('dist_class')['ride_id'].count().reset_index()
fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))

wedges, texts, autotexts = ax.pie(df_dist_class_distri['ride_id'], autopct='%1.1f%%', pctdistance=0.8, wedgeprops = {'width': 0.5, 'linewidth' : 1, 'edgecolor' : 'white' },
                        colors = sns.color_palette("Blues")[:len(df_dist_class_distri)], startangle= 0,
                        textprops={'fontsize':24, 'fontweight':'bold'}
)

ax.set_title('Distribution of distance travelled by classification', 
                     fontsize = 36, weight = 'bold')

annotations = ["Short\n x < 500m",
          "Short-Medium\n 500m <= x < 1.000m",
          "Medium\n 1.000m <= x < 2.500m",
          "Medium-Long\n 2.500m <= x < 5.000m",
          "Long\n 5.000m <= x < 10.000m",
          "X-Long\n 10.000m <= x"]

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(annotations[i], xy=(x, y), xytext=(1.2*np.sign(x), 1.2*y),
                horizontalalignment=horizontalalignment, **kw)

# del df_dist_class_distri

plt.savefig('./../pictures/donut_distclass.png', transparent = True)

plt.show()


# But is there a difference between the two groups and their ride distance?

# #### Histogram

# In[39]:


fig, ax = plt.subplots()

member = df_cleaned_v3.loc[df_cleaned_v3['member_casual']=='Member']['dist_ride']
casual = df_cleaned_v3.loc[df_cleaned_v3['member_casual']=='Casual']['dist_ride']
ax.hist([casual,member], bins =150, histtype='bar', stacked=True, label=['Casual', 'Member'], range=[0, 15000])
ax.set_title('Histogram of rides by distance', pad = 40)
ax.legend(title='Member type', loc='upper right')
ax.set_ylabel('# of rides', fontsize = 32, weight = 'bold')
ax.set_xlabel('Distance travelled [km]', fontsize = 32, weight = 'bold')
ax.xaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p:
    f'{locale.format_string("%d", round(x/1000,2), grouping=True)}km'))
ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p:
    f'{locale.format_string("%d", x, grouping=True)}'))

fig.subplots_adjust(top=.95)

plt.savefig('./../pictures/hist_ridedistance.png', transparent = True)
plt.show()


# The histogramm suggests that the casual riders ride slightly longer distances. Also it can be seen very well that the members use Cyclist more often.
# 
# But is there a difference in ride behaviour (number of rides, as well as distance) in the weekdays?

# #### Violinplot for member-type and weekday

# In[40]:


fig, ax = plt.subplots()

# boxplot = df_cleaned_v3.boxplot(column=['dist_ride','ride_timedelta'], by='member_casual')
ax = sns.violinplot(x="day_of_ride", y="dist_ride", hue="member_casual",  hue_order = ['Casual','Member'],
            data=df_cleaned_v3, linewidth=2.5, whis=[1,99], scale='count',
            order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday' ],
        #     height = 1200, widht= 1600
        )
ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: f'{locale.format_string("%d", round(x/1000,2), grouping=True)}km'))
ax.legend(fontsize=26, title='Type of Member')
ax.set_title('Distribution of distance travelled by Member-Type and Weekday', 
                     fontsize = 36, weight = 'bold', pad = 40)
ax.set_ylabel('Distance travelled [m]', fontsize = 28, weight = 'bold')
ax.set_xlabel('Weekday', fontsize = 28, weight = 'bold')

medians_dist = df_cleaned_v3.groupby(['day_of_ride','member_casual'])['dist_ride'].median()
max_dist = df_cleaned_v3.groupby(['day_of_ride','member_casual'])['dist_ride'].max()
vertical_offset = 1.2 # offset from median for display

i = 0
for xtick1 in ax.get_xticks():
    for n in range(2):
        i+= n%2
        iterator = xtick1 + i
        x_offset = xtick1 - 0.2 + 0.4*n
        t = ax.text(x_offset,medians_dist[iterator] * vertical_offset,f'{round(medians_dist[iterator]/1000,2)}km', 
                horizontalalignment='center',size='medium',color='black',weight='semibold')
        t.set_bbox(dict(facecolor='white', alpha=0.5))
        ax.text(x_offset,max_dist[iterator]*1.03, f'{round(max_dist[iterator]/1000,2)}km', 
                horizontalalignment='center',size='medium',color='firebrick',weight='semibold')

plt.savefig('./../pictures/violin_ridedist_weekday.png', transparent = True)
plt.show()


# The violinplot of the ridedistance grouped by weekday and member type clearly shows that the number of rides for the casual members is significantly less in the workweek days (Mo-Fr) as shown by the slimmer widths of the violins. On weekends the casual riders even outweigh the members.
# 
# For both parties it also shows that the weekend comes with increased ride distances.
# 
# Next to the weekday is there also a difference when it comes to the months or the season?

# #### Violinplot for member-type and month

# In[41]:


fig, ax = plt.subplots()

# boxplot = df_cleaned_v3.boxplot(column=['dist_ride','ride_timedelta'], by='member_casual')
ax = sns.violinplot(x="month_of_ride", y="dist_ride", hue="member_casual",  hue_order = ['Casual','Member'],
            data=df_cleaned_v3, linewidth=2.5, whis=[1,99], scale='count'
                 )
ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: f'{locale.format_string("%d", round(x/1000,2), grouping=True)}km'))
ax.legend(fontsize=26, title='Type of Member')
ax.set_title('Distribution of distance travelled by Member-Type and Month', pad = 40,
                     fontsize = 36, weight = 'bold')
ax.set_ylabel('Distance travelled [m]', fontsize = 28, weight = 'bold')
ax.set_xlabel('Month', fontsize = 28, weight = 'bold')
plt.xticks(rotation = 45)

medians_dist = df_cleaned_v3.groupby(['month_of_ride','member_casual'])['dist_ride'].median()
max_dist = df_cleaned_v3.groupby(['month_of_ride','member_casual'])['dist_ride'].max()
# print(medians)
vertical_offset = 1.2 # offset from median for display

i = 0
for xtick1 in ax.get_xticks():
    for n in range(2):
        i+= n%2
        iterator = xtick1 + i
        x_offset = xtick1 - 0.2 + 0.4*n
        t = ax.text(x_offset,medians_dist[iterator] * vertical_offset,f'{round(medians_dist[iterator]/1000,2)}km', 
                horizontalalignment='center',size='medium',color='black',weight='semibold')
        t.set_bbox(dict(facecolor='white', alpha=0.5))
        ax.text(x_offset,max_dist[iterator]*1.05, f'{round(max_dist[iterator]/1000,2)}km', 
                horizontalalignment='center',size='medium',color='firebrick',weight='semibold')

plt.savefig('./../pictures/violin_ridedist_month.png', transparent = True)
plt.show()


# The above violinplot is similar to the one before with the only diffeerence that the weekdays are substituted by the months. Here we also see a drastic reduction in rides (width of violin) for the cold months starting from October and stretching to March. January and February are especially low volume for the casual riders whereas the members still ride almost as much as during the summers.
# 
# Also for both parties the distance travelled is shorter during the winter months.

# ### Ride length [t]
# 
# Next let's look at the difference in riding behaviour for the length of the ride.

# In[42]:


#### Boxplot - Ride-length by Member-Type and Weekday


# In[43]:


ax = sns.boxplot(x="day_of_ride", y="length_of_ride_s", hue="member_casual",  hue_order = ['Casual','Member'],
            data=df_cleaned_v3, linewidth=2.5, whis=[1,99]
        )
ax.legend(fontsize=26, title='Type of Member')
ax.set_title('Distribution of length of ride by Member-Type and Weekday', 
                     fontsize = 36, weight = 'bold', pad = 40)
ax.set_ylabel('time travlled [t]', fontsize = 28, weight = 'bold')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: pd.to_timedelta(x, unit='s')))
ax.set_xlabel('Weekday', fontsize = 28, weight = 'bold')

medians_time = df_cleaned_v3.groupby(['day_of_ride','member_casual'])['length_of_ride_s'].median()
max_time = df_cleaned_v3.groupby(['day_of_ride','member_casual'])['length_of_ride_s'].max()
top_99 = df_cleaned_v3.groupby(['day_of_ride','member_casual'])['length_of_ride_s'].quantile(q=0.99, interpolation='nearest')

vertical_offset = 1.2 # offset from median for display

i = 0
for xtick1 in ax.get_xticks():
    for n in range(2):
        i+= n%2
        iterator = xtick1 + i
        x_offset = xtick1 - 0.2 + 0.4*n
        t = ax.text(x_offset,medians_time[iterator] * vertical_offset, strfdelta(datetime.timedelta(seconds=medians_time[iterator]),'%H:%M'), 
                horizontalalignment='center',size='medium',color='black',weight='semibold')
        t.set_bbox(dict(facecolor='white', alpha=0.5))
        ax.text(x_offset,max_time[iterator]*1.15, datetime.timedelta(seconds=max_time[iterator]), 
                horizontalalignment='center',size='medium',color='firebrick',weight='semibold')
        t2 = ax.text(x_offset,top_99[iterator]*1.15, datetime.timedelta(seconds=top_99[iterator]), 
                horizontalalignment='center',size='medium',color='firebrick',weight='semibold')
        t2.set_bbox(dict(facecolor='white', alpha=0.9))
        
medians_time = medians_time.reset_index()
max_time = max_time.reset_index()
top_99 = top_99.reset_index()

plt.savefig('./../pictures/boxplot_ridelength_day.png' , transparent = True)
plt.show()


# In[44]:


md(f"""
The above boxplot shows that the casual riders spent a significant time longer on the bike - 
by {(medians_time.loc[medians_time['member_casual']=='Casual']['length_of_ride_s'].mean()/medians_time.loc[medians_time['member_casual']=='Member']['length_of_ride_s'].mean() - 1):.1%} to be exact 
(Casual: {strfdelta(datetime.timedelta(seconds = medians_time.loc[medians_time['member_casual']=='Casual']['length_of_ride_s'].mean()), "%H:%M:%S")} 
vs
Member: {strfdelta(datetime.timedelta(seconds = medians_time.loc[medians_time['member_casual']=='Member']['length_of_ride_s'].mean()), "%H:%M:%S")}).

In the case of the ride length for Top 1% of rides (99th Percentile) this is even more drastic:
Casual: {strfdelta(datetime.timedelta(seconds = top_99.loc[top_99['member_casual']=='Casual']['length_of_ride_s'].mean()), "%H:%M:%S")}
vs
Member: {strfdelta(datetime.timedelta(seconds = top_99.loc[top_99['member_casual']=='Member']['length_of_ride_s'].mean()), "%H:%M:%S")}.

So far this shows a clear difference in  riding behaviour during the weekend, but is there also more differences throughout the year?
""")


# #### Boxplot - Ride-length by Member-Type and Month

# In[45]:


ax = sns.boxplot(x="month_of_ride", y="length_of_ride_s", hue="member_casual",  hue_order = ['Casual','Member'],
            data=df_cleaned_v3, linewidth=2.5, whis=[1,99]
                 )
ax.legend(fontsize=26, title='Type of Member')
ax.set_title('Distribution of length of ride by Member-Type and Month', pad = 40,
                     fontsize = 36, weight = 'bold')
ax.set_ylabel('time travlled [t]', fontsize = 28, weight = 'bold')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: pd.to_timedelta(x, unit='s')))
ax.set_xlabel('Month', fontsize = 28, weight = 'bold')
plt.xticks(rotation = 45)

medians_time = df_cleaned_v3.groupby(['month_of_ride','member_casual'])['length_of_ride_s'].median()
max_time = df_cleaned_v3.groupby(['month_of_ride','member_casual'])['length_of_ride_s'].max()
top_99 = df_cleaned_v3.groupby(['month_of_ride','member_casual'])['length_of_ride_s'].quantile(q=0.99, interpolation='nearest')

vertical_offset = 1.2 # offset from median for display

i = 0
for xtick1 in ax.get_xticks():
    for n in range(2):
        i+= n%2
        iterator = xtick1 + i
        x_offset = xtick1 - 0.2 + 0.4*n
        t = ax.text(x_offset,medians_time[iterator] * vertical_offset, strfdelta(datetime.timedelta(seconds=medians_time[iterator]),'%H:%M'), 
                horizontalalignment='center',size='medium',color='black',weight='semibold')
        t.set_bbox(dict(facecolor='white', alpha=0.5))
        ax.text(x_offset,max_time[iterator]*1.15, datetime.timedelta(seconds=max_time[iterator]), 
                horizontalalignment='center',size='medium',color='firebrick',weight='semibold')
        t2 = ax.text(x_offset,top_99[iterator]*1.15, datetime.timedelta(seconds=top_99[iterator]), 
                horizontalalignment='center',size='medium',color='firebrick',weight='semibold')
        t2.set_bbox(dict(facecolor='white', alpha=0.9))

medians_time = medians_time.reset_index()
max_time = max_time.reset_index()
top_99 = top_99.reset_index()

plt.savefig('./../pictures/boxplot_ridelength_month.png', transparent = True)
plt.show()


# In[46]:


md("""
The general difference in the ride length persisted throughout the year comparing the two groups.
There is howevere a synchronous change in ride length throughout the year for both. During the winter months (Oct - Mar)
the length of the ride drops by a significant ammount (ca. 30%) for both. The drop is slightly larger for the casual riders.

This seems to be innline with the previous findings that also the distance travelled is reduced in those months - however not by 30%.
During the winter months the riders seem to hurry more to get to their destination. Whereas in the summer
the motto seems to be "the ride is the goal" in Winter it shifts to "the destination is the goal".
""")


# ### Relation of distance and ride length
# 
# Now let's take a look at the relationship between the distance of the ride and the length.
# By now we know that the distance travelled is slightly longer for casual riders, but the time needed for this more than the distance would suggest.
# 
# If there is a correlation between the ride distance and length of the ride, I expect it to be stronger for the members who travel almost as much as the casual riders in distance, but do it in a faster manner.
# 
# Also the docked bike has some gigantic outliers with the ride length (> 20 days). To get a good grasp lets take a look at the relationship of distance and ride length grouped by member-type and ride-type.

# #### Linear Model of rides

# In[47]:


import scipy as sp
# annotate lmplots with pearson r and p-value to understand the correlation and statistical significance
# used from https://stackoverflow.com/questions/25579227/seaborn-lmplot-with-equation-and-r2-text 
def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['length_of_ride_s'], data['dist_ride'])
    ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p), size = 14,
            transform=ax.transAxes)


# In[48]:


df_implot_temp = df_cleaned_v3[['member_casual','rideable_type', 'length_of_ride_s', 'dist_ride']]
df_implot_temp['rideable_type'] = df_implot_temp['rideable_type'].apply(lambda x: x.replace('_',' '))

sns.set(rc={'figure.figsize':(64,32),
        'xtick.labelsize':16,
        'ytick.labelsize':16})

g = sns.lmplot(data = df_implot_temp, x = 'length_of_ride_s', y = 'dist_ride', row = 'member_casual',
        ci = 99, col ='rideable_type', scatter_kws = {'alpha' : 0.1},  hue = 'member_casual', hue_order = ['Casual', 'Member'],
        facet_kws = {'sharex':True, 'sharey':True, 'legend_out':True, 'margin_titles':True},
        col_order = ['Classic Bike', 'Electric Bike', 'Docked Bike'], row_order= ['Casual', 'Member'],
        truncate=False
    )

g.set(xlim=(0, 24*60**2))
g.set(ylim=(0, 30*10**3))

plt.subplots_adjust(hspace=0.35)

g.set_axis_labels("length of ride [t]", "distance travelled [km]", fontsize = 20, weight = 'bold')

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: f'{round(x/1000,1)}km'))
    ax.xaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: strfdelta(pd.to_timedelta(x, unit='s'), '%H:%M')))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        
g.map_dataframe(annotate)
g.fig.suptitle('Relation of ride length vs distance travelled', 
                     fontsize = 28, weight = 'bold')
g.set_titles(col_template="{col_name}", row_template="{row_name}", size = 20, weight = 'bold')
g.figure.subplots_adjust(top=.9)
        
plt.savefig('./../pictures/lmplot_length_vs_dist.png', transparent = True)
plt.show()


# In[49]:


# DF of classic bike
df_implot_temp = df_cleaned_v3[['member_casual','rideable_type', 'month_of_ride','length_of_ride_s', 'dist_ride']]\
    .loc[(df_cleaned_v3['rideable_type']=="Classic Bike")]

sns.set(rc={'figure.figsize':(64,32),
        'xtick.labelsize':16,
        'ytick.labelsize':16})

g = sns.lmplot(data = df_implot_temp, x = 'length_of_ride_s', y = 'dist_ride', ci = 99, col ='member_casual', col_order = ['Casual', 'Member'],
        row = 'month_of_ride', scatter_kws = {'alpha' : 0.1},  hue = 'member_casual', hue_order = ['Casual', 'Member'], aspect = 1.5,
        facet_kws = {'sharex':True, 'sharey':True, 'legend_out':True, 'margin_titles':True},
        # col_order = ['Classic Bike', 'Electric Bike', 'Docked Bike'], row_order= ['Casual', 'Member'],
        truncate=False
    )
g.set(xlim=(0, 8*60**2)) # limit to 8 hours day
g.set(ylim=(0, 30*10**3)) # limit to 30km

plt.subplots_adjust(hspace=0.25)

g.set_axis_labels("length of ride [t]", "distance travelled [km]", fontsize = 20, weight = 'bold')

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: f'{round(x/1000,1)}km'))
    ax.xaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: strfdelta(pd.to_timedelta(x, unit='s'), '%H:%M')))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        
g.map_dataframe(annotate)
g.set_titles(col_template="{col_name}", row_template="{row_name}", size = 20, weight = 'bold')
g.fig.suptitle("""Relation of ride length vs distance travelled
        Classic Bike""", 
                     fontsize = 28, weight = 'bold')
g.figure.subplots_adjust(top=.96)

plt.savefig('./../pictures/lmplot_length_vs_dist_classic_bike.png', dpi = 300, transparent = True)
plt.show()


# In[50]:


# DF of electric bike
df_implot_temp = df_cleaned_v3[['member_casual','rideable_type', 'month_of_ride','length_of_ride_s', 'dist_ride']]\
    .loc[(df_cleaned_v3['rideable_type']=="Electric Bike")]

sns.set(rc={'figure.figsize':(64,32),
        'xtick.labelsize':16,
        'ytick.labelsize':16})

g = sns.lmplot(data = df_implot_temp, x = 'length_of_ride_s', y = 'dist_ride', ci = 99, col ='member_casual', col_order = ['Casual', 'Member'],
        row = 'month_of_ride', scatter_kws = {'alpha' : 0.1},  hue = 'member_casual', hue_order = ['Casual', 'Member'], aspect = 1.5,
        facet_kws = {'sharex':True, 'sharey':True, 'legend_out':True, 'margin_titles':True},
        # col_order = ['Classic Bike', 'Electric Bike', 'Docked Bike'], row_order= ['Casual', 'Member'],
        truncate=False
    )
g.set(xlim=(0, 8*60**2)) # limit to 8 hours
g.set(ylim=(0, 30*10**3))

plt.subplots_adjust(hspace=0.25)

g.set_axis_labels("length of ride [t]", "distance travelled [km]", fontsize = 20, weight = 'bold')

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: f'{round(x/1000,1)}km'))
    ax.xaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: strfdelta(pd.to_timedelta(x, unit='s'), '%H:%M')))
    for label in ax.get_xticklabels():
        label.set_rotation(45)

g.map_dataframe(annotate)
g.set_titles(col_template="{col_name}", row_template="{row_name}", size = 20, weight = 'bold')
g.fig.suptitle("""Relation of ride length vs distance travelled
        Electric Bike""", 
                     fontsize = 28, weight = 'bold')
g.figure.subplots_adjust(top=.96)

plt.savefig('./../pictures/lmplot_length_vs_dist_electric_bike.png', dpi = 300, transparent = True)
plt.show()


# ### Stations and routes used
# 
# Now let's take a look if there is a geographical difference as well. Let's look at the most used stations for this and then plot it into the map.
# 
# #### Barchart - Visited Stations

# In[51]:


try:
    most_common_station.reset_index(inplace=True)
except:
    pass

g = sns.FacetGrid(most_common_station, row="member_casual",hue="member_casual", size=8, aspect=2, sharex=False)
g.map(sns.barplot,'station_name','no_of_rides')
g.fig.suptitle('Most Used Station by Member-Type', 
                     fontsize = 32, weight = 'bold')
g.fig.subplots_adjust(top=0.9) 
g.set_axis_labels("Station", "# of Rides", fontsize=26, weight='bold')
g.set_titles(row_template="{row_name}", size=26)

plt.subplots_adjust(hspace=0.6)

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mat_tick.FuncFormatter(lambda x, p: locale.format_string('%d', x , grouping=True)))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        
plt.savefig('./../pictures/most_station.png')
plt.show()


# #### Map Scatterplot - Visited Stations
# 
# Now let's visualize the difference in used stations and plot it onto the map. The size of the marker will indicate how frequently the station is visited by each member-type.

# In[52]:


# Create DF for the map scatterplot
scatter_map_df = df_cleaned_v3[['member_casual','start_station_id','end_station_id']].melt(id_vars=['member_casual'], var_name='station_type', value_name='station_id')\
    .groupby(['member_casual','station_id']).apply(lambda df: pd.Series({
        'no_of_rides':  df['station_type'].count()
    })).reset_index()
    
mean_lat_lng = df_cleaned_v3.groupby(['start_station_id','start_station_name'])\
    .apply(lambda df: pd.Series({
        'lat': df['start_lat'].mean(),
        'lng': df['start_lng'].mean(),
    })).reset_index().rename(columns={'start_station_id':'station_id', 'start_station_name': 'station_name'})
    
scatter_map_df = scatter_map_df.merge(mean_lat_lng, how = 'left', on = 'station_id').dropna(how='any')


# In[53]:


mapbox_acces_token = px.set_mapbox_access_token(open("../mapbox_token_public.json").read())
fig = px.scatter_mapbox(scatter_map_df,
                    lat=scatter_map_df['lat'],
                    lon=scatter_map_df['lng'],
                    size='no_of_rides',
                    color = 'member_casual',
                    hover_name="station_name",
                    hover_data = ['no_of_rides'],
                    opacity = 0.8,
                    zoom = 12,
                    labels = {'member_casual': 'Member Type'},
                    height = 1200,
                    width = 1200)


fig.update_layout(margin={"r":0,"t":60,"l":0,"b":0},  # remove the white gutter between the frame and map
    # hover appearance
    hoverlabel=dict(
        bgcolor="white",     # white background
        font_size=16,), # label font size
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="right",
        x=0.98,
        ),
    title = dict(
        x = 0.5,
        xanchor = 'center',
        text = '<b>Bike Stations Used by Member-Type and Frequency</b>',
        font = dict(
            size = 32,
        )
    )
)
fig.write_image('./../pictures/station_map.png')

fig.show()


# #### Map Scatter- and Lineplot - Visited Stations and used routes
# 
# Now let's visualize the difference in used stations and the typical routes taken by the riders. For this however the number of combinations for the routes (station to station) will be too big and the plot would end up quite confusing.
# 
# We will limit the lines plotted to visualize the routes taken therefore to the top 200 routes for each member type.

# In[54]:


# Create DF for the the different routes taken

scatter_map_rides_df = df_cleaned_v3.groupby(['member_casual','start_station_id','end_station_id'])['ride_id'].count()\
    .rename('no_of_rides').reset_index()

scatter_map_rides_df  =scatter_map_rides_df.loc[(scatter_map_rides_df['no_of_rides']!=0)]
print(scatter_map_rides_df.shape)

# Calculate the mean for Lat + Long for each station (otherwise one station would have several groups)
mean_lat_lng_start = df_cleaned_v3.groupby(['start_station_id','start_station_name'])\
    .apply(lambda df: pd.Series({
        'lat': df['start_lat'].mean(),
        'lng': df['start_lng'].mean(),
        'no_of_rides': df['ride_id'].count(),
    })).reset_index()\
        .rename(columns={'start_station_id':'station_id', 'start_station_name':'station_name'})
    
mean_lat_lng_end = df_cleaned_v3.groupby(['end_station_id','end_station_name'])\
    .apply(lambda df: pd.Series({
        'lat': df['end_lat'].mean(),
        'lng': df['end_lng'].mean(),
        'no_of_rides': df['ride_id'].count(),
    })).reset_index()\
        .rename(columns={'end_station_id':'station_id', 'end_station_name':'station_name'})

scatter_map_rides_df = scatter_map_rides_df.merge(mean_lat_lng_start, how = 'left',
        left_on='start_station_id', right_on='station_id', suffixes=(None,'_start'))\
        .merge(mean_lat_lng_end, how = 'left', left_on='end_station_id', right_on='station_id', suffixes=(None,'_end'))
scatter_map_rides_df.drop(columns=['station_id', 'station_id_end','no_of_rides_start', 'no_of_rides_end'], inplace=True)
scatter_map_rides_df.rename(columns={'lat':'lat_start', 'lng':'lng_start'}, inplace=True)

scatter_map_rides_df['line_width'] = round(scatter_map_rides_df['no_of_rides'] / scatter_map_rides_df['no_of_rides'].max() * 10,2)
scatter_map_rides_df['line_width'] = scatter_map_rides_df['line_width'].apply(lambda x: 1 if x < 1 else x)

scatter_map_rides_df.dropna(inplace=True)


# In[55]:


# Create DF to plot stations - this time unified (no start + end separation or member-type separation)
stations_df = pd.concat([mean_lat_lng_start, mean_lat_lng_end])
stations_df = stations_df.groupby(['station_id', 'station_name']).apply(lambda df: pd.Series({
    'lat': df['lat'].mean(),
    'lng': df['lng'].mean(),
    'no_of_rides': df['no_of_rides'].sum(),
})).reset_index()


# In[56]:


mapbox_acces_token = px.set_mapbox_access_token(open("../mapbox_token_public.json").read())
# mapbox_acces_token = px.set_mapbox_access_token(open("mapbox_token_public.json").read())

fig = px.scatter_mapbox(data_frame=stations_df,
    lat='lat',
    lon='lng',
    size= 'no_of_rides',
    color_discrete_sequence = ['rgba(31,31,31, 0.9)'],
    hover_name= 'station_name',
    hover_data = ['no_of_rides'],
    opacity = 0.8,
    zoom = 12,)

    
for idx, temp_df in scatter_map_rides_df.groupby('member_casual'):
    color = color_dict.get(idx, 'rgba(31, 31, 31, 1)').replace(", 1)", ", 0.5)")
    temp_df.sort_values(by='no_of_rides', ascending=False, inplace=True)
    temp_df = temp_df[:200].reset_index()
    for i in range(len(temp_df)):
        lat = [temp_df['lat_start'][i], temp_df['lat_end'][i]]
        lon = [temp_df['lng_start'][i], temp_df['lng_end'][i]]
        fig.add_trace(ptly_go.Scattermapbox(
            mode = "lines",
            lat = lat,
            lon = lon,
            name = idx,
            hovertext = f"{temp_df['station_name'][i]} - {temp_df['station_name_end'][i]} \n # Rides: {temp_df['no_of_rides'][i]} ",
            # hoverinfo = ['no_of_rides'],
            line = dict(
            color= color, 
            width = temp_df['line_width'][i]*2, # multiply by 2 because original value not big enoug
            ),
        ))

fig.update_layout(margin ={'l':0,'t':45,'b':0,'r':0},
        mapbox = {
            'zoom': 11},
            width=1200,
            height=1200,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        ),
        title = dict(
        x = 0.5,
        xanchor = 'center',
        text = '<b>Top 200 Routes by Member-Type</b>',
        font = dict(
            size = 32,
        )
    )
        )

fig.update_traces(showlegend=False)

fig.write_image('./../pictures/routes_map.png')


fig.show()


# The plot shows a clear concentration of the casual riders near the lake side and in or between the several parks.
# 
# For the members there are several little hubs or networks. One which is near downtown and a big one near the university. 

# ## Conclusion
# 
# For the members the Cyclist bike seems to be more integrated into their daily life and used for work or similar things.  They ride their bike throughout the year with little change and are not wasting time when they are riding.
# 
# In contrast the casual riders show a clear tendence towards riding as a leisure activity. They ride their bikes mostly on the weekens and during the hotter months, especially in summer. While they ride mostly in the parks they seem to stroll around which is why there is also a weaker correlation between the ride time and the distance of the ride. When winter hits, this however changes and the cold temperatures seem to shift the behaviour. For the few remaining casual riders the bike becomes more of a vehicle to get from point A to B instead of a long stroll through the new cihlly cold park.
