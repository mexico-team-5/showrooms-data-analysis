#pip install psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import numpy as np  ## Import the NumPy package
import matplotlib.pyplot as plt  ## import fundamental plotting library in Python
import seaborn as sns  ## Advanced plotting functionality with seaborn
import os


from datetime import datetime, date, time, timedelta
sns.set(style="whitegrid") # can set style depending on how you'd like it to look

## Conect postgres to python
conn = psycopg2.connect(host='external-kavak-ds4a.kavak.services',
                        port=5432, database="kavak", user="ds4a_t5",
                       password = '%$Hhp!D2y&iJwg%Jdh')  #load database
cursor1=conn.cursor()
print("Connected to Database...")


print("Saving Query...")
# Save Query into a dataframe
car_pageviews = pd.read_sql_query("select * from inv_dist.car_pageviews", conn) 
showroom = pd.read_sql_query("select * from showroom", conn)
showroom_visits = pd.read_sql_query("select a.*, b.name location_name from inv_dist.showroom_visits a LEFT JOIN inv_dist.showroom b on a.showroom_id=b.showroom_id", conn)  
car_description1 = pd.read_sql_query(
    '''SELECT c.*, d.name brand_name FROM
            (SELECT a.*, b.name location_name 
             FROM inv_dist.car_description_1 a LEFT JOIN showroom b 
        ON a.sold_location_id=b.showroom_id) c
        LEFT JOIN inv_dist.brands d ON c.brand_id=d.brand_id
    ''', conn)
car_description = car_description1

print("Starting Data Cleaning...")

### Add stock_ID to the dataset for Pageviews table
car_pageviews['stock_id'] = car_pageviews['gmb_pagepath'].apply(lambda x: 
                                                                x[(x.rfind("autos-")+6):len(x)])

##Create date columns
car_pageviews['gmb_datetime'] = pd.to_datetime(car_pageviews['gmb_date'],
                                               format="%Y-%m-%d")
car_pageviews['year'] = car_pageviews['gmb_datetime'].dt.year
car_pageviews['month'] = car_pageviews['gmb_datetime'].dt.month
car_pageviews['my'] =pd.to_datetime(("01" + "/" + car_pageviews['month'].astype(str) + "/" + car_pageviews['year'].astype(str)), format="%d/%m/%Y")

##Delete showroom equal to "FUERA DE KAVAK"
#car_description = car_description.drop(car_description[car_description['location_name']=="FUERA DE KAVAK"].index)
list_of_values = ['otro']
car_description = car_description.drop(car_description[car_description['location_name']=='otro'].index)


## Create new columns for numeric KAVAK SEGMENTS

a=('[ 79,999 - 209,999 ]','[ 212,999 - 339,999 ]','[ 342,999 - 1054,999 ]')
a=('[ 1,050 - 36,300 ]','[ 36,649 - 64,851 ]','[ 64,965 - 110,900 ]')
a=('Low','Mid','High')

car_description['new_price_segment'] = car_description['price_segment'].map({1: "[ 79,999 - 209,999 ]", 2: "[ 212,999 - 339,999 ]", 3: "[ 342,999 - 1054,999 ]"})
car_description['new_km_segment'] = car_description['km_segment'].map({1: "[ 1,050 - 36,300 ]", 2: "[ 36,649 - 64,851 ]", 3: "[ 64,965 - 110,900 ]"})
car_description['new_model_segment'] = car_description['model_segment'].map({1: "Low", 2: "Mid", 3: "High"})


##Create date columns
car_description['sold_datetime'] = pd.to_datetime(car_description['sold_date'], format="%Y-%m-%d")

car_description['bought_datetime'] = car_description['sold_datetime'] - pd.to_timedelta(car_description['inventory_days'], unit='D')

### Sum total pageviews per car ###
import datetime
car_description['tot_pageviews']=9999
n_days = 60

for indice_fila, fila in car_description.iterrows():
    stockid=car_description.loc[indice_fila,'stock_id']
    stockid=str(stockid)
    lastdate=car_description.loc[indice_fila,'sold_datetime']
    invdays=car_description.loc[indice_fila,'inventory_days']
    invdays=int(invdays)
    firstdate = lastdate-datetime.timedelta(days=invdays)
    #if invdays>=60:
    #    invdays=60
    #else:
    #    invdays
    newlastday = firstdate+datetime.timedelta(days=n_days)
    pv = car_pageviews[
        (car_pageviews['stock_id']==stockid) & (car_pageviews['gmb_datetime']<=newlastday)
    ]
    value=pv['views'].sum()
    car_description.loc[indice_fila,'tot_pageviews']=value


car_description['age']=car_description['bought_datetime'].dt.year -car_description['year']
car_description['pv_day'] = car_description['tot_pageviews']/n_days

print("Starting EDA...")

### Inventory days distribuition by showroom ordered by mean ###
order = car_description.groupby('location_name')['inventory_days'].median().sort_values(ascending=True).iloc[::1].index

fig, ax = plt.subplots(figsize=(10,4))
m = sns.violinplot(x="location_name",y="inventory_days",data=car_description,orient='vertical',showfliers=False, order=order)

# Format plot
plt.title('Inventory days distribuition by showroom ordered by mean', fontsize=20, verticalalignment='bottom')
plt.xticks(rotation=90);
plt.xlabel('Showroom location', fontsize = 15)
plt.ylabel('Inventory days', fontsize = 15);
plt.savefig("Inventory_days_distribuition_by_showroom_ordered_by_mean.jpg")


### Price distribution by showroom ordered by mean ###
order = car_description.groupby('location_name')['published_price'].median().sort_values(ascending=True).iloc[::1].index

fig, ax = plt.subplots(figsize=(10,4))
m = sns.violinplot(x="location_name",y="published_price",data=car_description,orient='vertical',showfliers=False, order=order)

# Format plot
plt.title('Price distribution by showroom ordered by mean', fontsize=20, verticalalignment='bottom')
plt.xticks(rotation=90);
plt.xlabel('Showroom location', fontsize = 15)
plt.ylabel('Price', fontsize = 15);
plt.savefig("Price_distribution_by_showroom_ordered_by_mean.jpg")


### Footfall distribution by showroom ###
#order = showroom_visits.groupby('location_name')['visits'].mean().sort_values(ascending=True).iloc[::1].index
#fig, ax = plt.subplots(figsize=(10,4))
#m = sns.violinplot(x="location_name",y="visits",data=showroom_visits,order = ['WH - LERMA', 'PLAZA FORTUNA', 'FLORENCIA', 'SANTA FE'] )
#m = sns.violinplot(x="location_name",y="visits",data=showroom_visits)
#m.set_title('Footfall distribution by showroom' , fontsize=20, verticalalignment='bottom')
#m.set_ylabel('Footfall' , fontsize = 15)
#m.set_xlabel('Showroom location' , fontsize = 15)
#plt.savefig("Footfall_distribution_by_showroom.jpg")

### Online pageviews distribution by showroom ordered by mean ###
order = car_description.groupby('location_name')['tot_pageviews'].mean().sort_values(ascending=True).iloc[::1].index
fig, ax = plt.subplots(figsize=(10,4))
#m = sns.violinplot(x="location_name",y="visits",data=showroom_visits,order = ['WH - LERMA', 'PLAZA FORTUNA', 'FLORENCIA', 'SANTA FE'] )
m = sns.violinplot(x="location_name",y="tot_pageviews",data=car_description)
m.set_title('Online pageviews distribution by showroom ordered by mean' , fontsize=20, verticalalignment='bottom')
m.set_ylabel('Online pageviews for the cars' , fontsize = 15)
m.set_xlabel('Showroom location' , fontsize = 15)
plt.savefig("Online_pageviews_distribution_by_showroom_ordered_by_mean.jpg")

### Online Pageviews  distribution by price ###
order = car_description.groupby('published_price')['tot_pageviews'].mean().sort_values(ascending=True).iloc[::1].index
fig, ax = plt.subplots(figsize=(15,7))
m = sns.scatterplot(x="published_price",y="tot_pageviews", data=car_description)
m.set_title('Online Pageviews  distribution by price', fontsize = 15)
m.set_ylabel('Online Pageviews', fontsize = 15)
m.set_xlabel('Price', fontsize = 15)
#[SS['LUGAR.DE.VENTA'] == "SANTA FE"]
plt.savefig("Online_Pageviews_distribution_by_price.jpg")

#####  Online Pageviews  distribution per Showroom  #####

fig, axes = plt.subplots(figsize=(20,10))
plt.subplot(2,2,1)
SS3=car_description[car_description['location_name']=='Florencia']
order = SS3.groupby('published_price')['tot_pageviews'].mean().sort_values(ascending=True).iloc[::1].index
m = sns.scatterplot(x="published_price",y="tot_pageviews", data=SS3)
m.set_title('Online Pageviews  distribution by price FLORENCIA', fontsize = 15)
m.set_ylabel('Online Pageviews', fontsize = 15)
m.set_xlabel('Price', fontsize = 15)
plt.savefig("Online_Pageviews_distribution_by_price_FLORENCIA.jpg")

plt.subplot(2,2,2)
SS4=car_description[car_description['location_name']=='Fortuna']
order = SS4.groupby('published_price')['tot_pageviews'].mean().sort_values(ascending=True).iloc[::1].index
n = sns.scatterplot(x="published_price",y="tot_pageviews", data=SS4)
n.set_title('Online Pageviews  distribution by price PLAZA FORTUNA', fontsize = 15)
n.set_ylabel('Online Pageviews', fontsize = 15)
n.set_xlabel('Price', fontsize = 15)
plt.savefig("Online_Pageviews_distribution_by_price_PLAZA_FORTUNA.jpg")

plt.subplot(3,2,5)
SS5=car_description[car_description['location_name']=='Lerma']
order = SS5.groupby('published_price')['tot_pageviews'].mean().sort_values(ascending=True).iloc[::1].index
o = sns.scatterplot(x="published_price",y="tot_pageviews", data=SS5)
o.set_title('Online Pageviews  distribution by price WH - LERMA', fontsize = 15)
o.set_ylabel('Online Pageviews', fontsize = 15)
o.set_xlabel('Price', fontsize = 15)
plt.savefig("Online_Pageviews_distribution_by_price_LERMA.jpg")

plt.subplot(3,2,6)
SS6=car_description[car_description['location_name']=='Santa Fe']
order = SS6.groupby('published_price')['tot_pageviews'].mean().sort_values(ascending=True).iloc[::1].index
p = sns.scatterplot(x="published_price",y="tot_pageviews", data=SS6)
p.set_title('Online Pageviews  distribution by price SANTA FE', fontsize = 15)
p.set_ylabel('Online Pageviews', fontsize = 15)
p.set_xlabel('Price', fontsize = 15);
plt.savefig("Online_Pageviews_distribution_by_price_SANTA_FE.jpg")

### Bar Graphs brand_name ###
car_description['brand_name'].value_counts().plot.bar()
plt.title("Cars sold per brand", fontsize=20, verticalalignment='bottom');
plt.ylabel("Cars sold", fontsize=15, verticalalignment='bottom');
plt.xlabel('Brand', fontsize=15, verticalalignment='bottom');
plt.savefig("Bar_Graphs_brand_name.jpg")

#####  Bar Graphs Brands per Showroom  #####

fig, axes = plt.subplots(figsize=(20,5))
plt.subplot(1,4,1)
SS3=car_description[car_description['location_name']=='Florencia']
SS3['brand_name'].value_counts().plot.bar()
plt.title("FLORENCIA", fontsize=20, verticalalignment='bottom')
plt.savefig("Bar_Graphs_brand_name_FLORENCIA.jpg")

plt.subplot(1,4,2)
SS4=car_description[car_description['location_name']=='Fortuna']
SS4['brand_name'].value_counts().plot.bar()
plt.title("PLAZA FORTUNA", fontsize=20, verticalalignment='bottom')
plt.savefig("Bar_Graphs_brand_name_PLAZA_FORTUNA.jpg")

plt.subplot(1,4,3)
SS5=car_description[car_description['location_name']=='Lerma']
SS5['brand_name'].value_counts().plot.bar()
plt.title("WH - LERMA", fontsize=20, verticalalignment='bottom')
plt.savefig("Bar_Graphs_brand_name_LERMA.jpg")

plt.subplot(1,4,4)
SS6=car_description[car_description['location_name']=='Santa Fe']
SS6['brand_name'].value_counts().plot.bar()
plt.title("SANTA FE", fontsize=20, verticalalignment='bottom');
plt.savefig("Bar_Graphs_brand_name_SANTA_FE.jpg")

### Inventory days distribution by brand ordered by mean ###
order = car_description.groupby('brand_name')['inventory_days'].mean().sort_values(ascending=True).iloc[::1].index
fig, ax = plt.subplots(figsize=(35,12))
m = sns.violinplot(x="brand_name",y="inventory_days",data=car_description,orient='vertical', order = order , fontsize = 30)
m.set_title('Inventory days distribution by brand ordered by mean', fontsize = 20)
m.set_ylabel('inventory days', fontsize = 20)
m.set_xlabel('brand', fontsize = 20)
plt.savefig("Inventory_days_distribution_by_brand_ordered_by_mean.jpg")

### Price distribution by brand ordered by mean ###
order = car_description.groupby('brand_name')['published_price'].mean().sort_values(ascending=True).iloc[::1].index
fig, ax = plt.subplots(figsize=(35,12))
m = sns.violinplot(x="brand_name",y="published_price",data=car_description,orient='vertical', order = order , fontsize = 30)
m.set_title('Price distribution by brand ordered by mean', fontsize = 20)
m.set_ylabel('Price', fontsize = 20)
m.set_xlabel('brand', fontsize = 20)
plt.savefig("Price_distribution_by_brand_ordered_by_mean.jpg")

### Online pageviews distribution by brand ordered by mean ###
order = car_description.groupby('brand_name')['tot_pageviews'].mean().sort_values(ascending=True).iloc[::1].index
fig, ax = plt.subplots(figsize=(35,12))
m = sns.violinplot(x="brand_name",y="tot_pageviews",data=car_description,orient='vertical', order = order , fontsize = 30)
m.set_title('Online pageviews distribution by brand ordered by mean', fontsize = 20)
m.set_ylabel('Online Pageviews', fontsize = 20)
m.set_xlabel('brand', fontsize = 20)
plt.savefig("Online_pageviews_distribution_by_brand_ordered_by_mean.jpg")




