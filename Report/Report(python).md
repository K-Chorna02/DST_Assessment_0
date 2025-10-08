#Section: Regional Analysis -Python

This analysis continues from the section on regional data. The code below has been taken from a public Git repository which contains examples of EDA on the exact same dataset used for this report.
The Git repository can be accessed at :  https://github.com/Abel-2005/EDA-FOR-AIR-QUALITY-INDEX.git


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyprojroot import here

#data_path = here("data/who_data.csv")

# Read the CSV
#df = pd.read_csv(data_path)
file_path = "\\Users\\ilc04\\OneDrive - University of Bristol\\DST\\DST_Assessment_0\\Isobelle Clemmens\\who_data.csv"
df = pd.read_csv(file_path, encoding='latin1')

```


```python
numeric_cols = [
    'year', 'pm10_concentration', 'pm25_concentration', 'no2_concentration',
    'pm10_tempcov', 'pm25_tempcov', 'no2_tempcov', 'population',
    'latitude', 'longitude', 
]

non_numeric_cols = [
    'who_region', 'iso3', 'country_name', 'city', 'version',
    'type_of_stations'
]
numeric_df = df[numeric_cols]


```

The following plot was adapted to show the top 10 most polluting cities in the WHO region rather than filtering by country in the orignal code.


```python
year_filter = 2020
region_filter = '1_Afr'

top_cities = df[
    (df['year'] == year_filter) & 
    (df['who_region'] == region_filter)
][['city', 'pm25_concentration']] \
.sort_values(by='pm25_concentration', ascending=False) \
.dropna().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_cities, x='pm25_concentration', y='city', palette='mako')
plt.title(f'Top 10 Most Polluted Cities in {region_filter} (PM2.5) - {year_filter}')
plt.xlabel('PM2.5 Concentration (µg/m³)')
plt.ylabel('City')
plt.tight_layout()
plt.show()
```

    C:\Users\ilc04\AppData\Local\Temp\ipykernel_22504\4166675728.py:12: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=top_cities, x='pm25_concentration', y='city', palette='mako')
    


    
![png](output_4_1.png)
    


The following code was adapted from the repository. This plot is useful for spotting temporal trends and comparing regions but as shown previously, the proportion of missing data varies greatly between regions, and so it is difficult to draw conclusions when we cannot have the same confidence in each region's plot being accurate.


```python
eu_avg = df[df['who_region']=='4_Eur'].groupby('year')['pm10_concentration'].mean()
amr_avg = df[df['who_region']=='2_Amr'].groupby('year')['pm10_concentration'].mean()
afr_avg = df[df['who_region']=='1_Afr'].groupby('year')['pm10_concentration'].mean()
sear_avg = df[df['who_region']=='3_Sear'].groupby('year')['pm10_concentration'].mean()
emr_avg = df[df['who_region']=='5_Emr'].groupby('year')['pm10_concentration'].mean()
wpr_avg = df[df['who_region']=='6_Wpr'].groupby('year')['pm10_concentration'].mean()


plt.figure(figsize=(14,6))
plt.plot(eu_avg.index, eu_avg.values, label='Europe', marker='o', linewidth=2)
plt.plot(amr_avg.index, amr_avg.values, label='Americas', marker='o', linewidth=2)
plt.plot(afr_avg.index, afr_avg.values, label='West Africa', marker='o', linewidth=2)
plt.plot(sear_avg.index, sear_avg.values, label='South East Asia', marker='o', linewidth=2)
plt.plot(emr_avg.index, emr_avg.values, label='East Mediterranean', marker='o', linewidth=2)
plt.plot(wpr_avg.index, wpr_avg.values, label='West Pacific', marker='o', linewidth=2)
plt.xlim(2008,2023)
plt.title('NO2 Trends Over Years: WHO Regions')
plt.xlabel('Year')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
```


    
![png](output_6_0.png)
    


Now we explore how correlation coefficients vary between regions. The repository provided data for a correlation heatmap for the whole numerical dataset which was then used to plot the correlatino heatmap for each individual region.


```python
for region, data in df.groupby("who_region"):
    
    num_data = data[numeric_cols]

    corr = num_data.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(f"Correlation Heatmap – {region}")
    plt.tight_layout()
    plt.show()

```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



    
![png](output_8_3.png)
    



    
![png](output_8_4.png)
    



    
![png](output_8_5.png)
    



    
![png](output_8_6.png)
    


This proved to be quite useful in a number of ways. Firstly Europe, already identified as having the most complete dataset also has little to no correlation between any of its features. This is useful from machine learning standpoint if we were to take the use of this dataset futher into a machine learning model as effects of multicollinearity would be of little concern. With a large and uncorrelated dataset, Europe would be a suitable subset of data to train a machine learning model on.
Secondly, other regions have very correlated data; a pattern that stands out as a potential relationship is the significant correlation coefficients between PM10 and the geographical coordinates for several of the regions. To explore this we plot PM10 concentration against longitude.


```python

for region, data in df.groupby("who_region"):
    
    num_data = data[numeric_cols]
   
    plt.figure(figsize=(6, 5))
    plt.scatter(num_data['longitude'],num_data['pm10_concentration'],)
    plt.title(f"PM10 Concetration against Longitude – {region}")
    plt.tight_layout()
    plt.show()
    
```


    
![png](output_10_0.png)
    



    
![png](output_10_1.png)
    



    
![png](output_10_2.png)
    



    
![png](output_10_3.png)
    



    
![png](output_10_4.png)
    



    
![png](output_10_5.png)
    



    
![png](output_10_6.png)
    


While some region have little to no visible correlation due to sparsity of data, we can see that for eaxmple in the South East Asia region there is a clear correlation between longitude and PM10 concentration, perhaps suggesting that the cities and towns further north are more developed or more populated hence we'd expect more pollution from those cities and towns.
We could potentially take this further in the future to predict a country or city's pollutant concentration based on its region and geographical coordinates.

##Conclusion
To conclude our section on linear regression, our exploratory analysis investigated how population size, time, station type, and regional factors influence air pollution levels across cities.
Across all three pollutants, PM_2.5, PM_10, and NO_10,population and urbanisation emerged as strong, statistically significant predictors of higher pollution.
Regional effects also proved important: while some regions (such as the Eastern Mediterranean) experienced higher pollution, others (like Europe and the Western Pacific) recorded significantly lower levels.

However, across the three final models for PM_2.5, PM_10, and NO_10, the value R2
 did not exceed 0.239, indicating that while these investigated predictors do explain a part of variation in pollution levels, a very large proportion still remains unexplained.

To amend this and further our analysis, we would need to consider additional explanatory variables. These could include:

Whether conditions - temperature, wind speed, humidity.
Industrial emissions.
Traffic density.
Similarly, we could consider exploring non-linear relationships. This could include squared or log-transformed terms for population or year, as an attempt to capture diminishing or accelerating effects.

In section 4 we have seen how the population size impacts the average concentration of each pollutant, as the population size increases the average concentration of pollutants increases. As we have such a spread in the values of the population we have transformed our data using a log transformation and this made it visually better to see the trends in the data, in addition after using the R^2 test we saw that the log model for pm25 and NO2 concentrations were better models.

After extensive exploration of our data, we can conclude that there is a general decline in pollutant concentrations over time in most European countries (from 2015 onwards). We explored 2 different methods when creating time plots, by using ggplot and by using built in R packages. The ggplots were easier to read when picking out key information such as year and specific concentration. However for exploring trends, openair was very useful as it provided a clear visual representation of all concentrations- making it easy to compare between countries and across different particle types.

In section explored global city air quality data from WHO. PM2.5 and PM10 are strongly correlated, while NO₂ shows a moderate correlation with both. Limiting data to cities with over 75% coverage doesn’t change these correlations much, but it removes outliers and reduces the dataset, making some early years, like 2010–2012, unreliable due to very few records and noticeable spikes. After 2013, pollutant levels show a steady decrease, with PM10 consistently the highest, followed by NO₂ and PM2.5. Coverage is uneven across regions and most high-coverage data comes from Europe, with some from North America, while Africa and the Eastern Mediterranean are greatly underrepresented. This means the trends mainly reflect well-covered regions like Europe, where pollution is decreasing due to legislation and other measures, but these patterns may not represent global trends.

In the regional analysis section we have identified areas where data is much more sparse than others. The Git repository provided useful visualisations but some were not useful enough to draw conclusions from because we have to consider the impact of the missing data on any trends or patterns that we see. The correlation heatmaps for each region provided useful information we wouldn't have otherwise seen in the correlation matrix for the entire dataset.

Overall our extensive EDA of this dataset has provided us with all the information we need to use this data for modelling or prediction purposes.
