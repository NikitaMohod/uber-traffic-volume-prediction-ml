# Uber Traffic Volume Prediction using Machine Learning


## Project Overview
- The aim of this project is to analyze **traffic volume data** and evaluate how various factors like weather, time, and calendar attributes affect traffic.  
- We build and compare **Machine Learning models** (Linear Regression, Random Forest, XGBoost) to predict traffic volume and generate actionable insights for **businesses like Uber** to optimize driver allocation, surge pricing, and demand forecasting.

---


## Dataset Overview
- **Rows:** 48,187  
- **Columns:** 15
- **Features Include:**  
  - Temperature, Rain, Snow, Clouds  
  - Hour, Day of Week, Month, Rush Hour Indicator  
  - Target: **Traffic Volume**

---

 
## Exploratory Data Analysis (EDA)
### 1. Univariate Analysis
![Univariate Analysis Hist Plot](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223221.png)

### Univariate Analysis Insights
#### 1> Distribution of Temperature (temp) (This Graph show That mostly Days in which Temprature)
- If the bar is tall at a certain temperature, it means there were more days with that temperature.
- Mostly between 272–292 K (normal range) means Most of the days were neither too hot nor too cold it is medium temperature (25-30C).

#### 2> Distribution of Rain (rain_1h) (This Graph Shows that How much rain occurred every hour and how many times it happened.)
- If the bar is tall at 0 mm, it means that most of the time there was no rainfall.
- Mostly 0, but extreme outliers (9831 mm) → unrealistic means Rain occurs for a very short time, but when it does, it is heavy.

#### 3> Distribution of Snow (snow_1h) (This Graph show that How much snow fell in each hour)
- If all the bars are small or at zero, then it means there is no snowfall.
- It is almost impossible for snow to fall at this place.

#### 4> Distribution of Clouds (clouds_all) (This Graph is show that How much the sky was covered with clouds (0 = clear, 100 = fully covered).)
- If there are more bars in the 80–100% range, it means most of the days were cloudy.
- Spread across 0–100%, with peaks at clear and overcast means During the monsoon, the sky remains covered with clouds most of the time.
- Can affect traffic visibility.

#### 5> Distribution of Hour of Day (hour) (This Graph show that At which hour of the day was the traffic highest?)
- So that which Hour has long Graph.
- Strong daily pattern, traffic depends on time means This graph shows that the data is unbiased and balanced — data for every hour is available, which allows for accurate comparison across the entire day.
- Very important predictor.

#### 6> Distribution of Day of Week (This Graph Show That On which day of the week was the traffic highest?)
- All days balanced means  Counts are nearly equal across all days (0 to 6), indicating consistent activity throughout the week.
- Needed to capture weekday vs weekend difference.
- Cyclical Trend: The sinusoidal line suggests a periodic pattern—behavior or intensity varies rhythmically across days, even though total volume remains stable.

- 7> Distribution of Month (This Graph Show that How was the traffic in each month of the year?)
    - Covers all seasons means The bar heights are fairly consistent across all months, indicating that the overall activity remains stable throughout the year.
 
#### 8> Distribution of Weekend (is_weekend) (This Graph show that Whether that day was a weekend or not.)
- The count of is_weekend = 0 (workdays) is very high, indicating that most of the data is based on workdays.
- The data is skewed towards weekdays compared to weekends, indicating that most activities or events occur on working days.

#### 9> Distribution of Rainy(is_rainy) (This Graph Show that How many times did it rain?)
- So that is_rainy = 0 means Rain Free day and is_rainy = 1 means rainy day
- so there is show is_rainy = 0 bar is higher than is_rainy = 1 so that Which indicates that most days are rain-free.
- This variable takes only two values (0 and 1), clearly distinguishing between rain and no rain.

#### 10> Distribution of Snowy (is_snowy) (This Graph Show that How many times did it snow?)
- The count of is_snowy = 0 is very high, indicating that most days had no snowfall.
- This variable takes only two values (0 and 1), clearly distinguishing between snowy and non-snowy days.

#### 11> Distribution of Rush Hour (is_rush_hour) (This Graph show that How many times data was recorded during rush hours(Busy Time).)
- Most of the data has is_rush_hour = 0, indicating that the majority of activities do not occur during rush hours.
- This variable takes only two values (0 and 1), clearly distinguishing between rush hours and non-rush hours.

#### 12> Distribution of Traffic Volume (Target) (This Graph Show that How many vehicles were on the road.)
- Several common traffic levels: The graph shows multiple peaks—especially around low volume, 3000, and 5000—indicating that traffic frequently occurs at these levels.
- Low traffic occurs most frequently: The highest peak is at a lower traffic volume, indicating that low traffic conditions happen most often.
- High variability in traffic: The spread of the graph and multiple peaks indicate that traffic volume varies considerably, i.e., it changes across different times or conditions.

---

### 2. Bivariate Analysis
![Bivariate Analysis Scatter Plot](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223319.png)

### Bivariate Analysis Insights
#### 1> temp vs Traffic Volume
- This  scatter plot show That  shows the traffic volume at different temperatures.
- Most of the data is between 250 and 300 temperature units, indicating that traffic volume is mostly recorded within this temperature range.
- Even at the same temperature, traffic volume ranges from 0 to over 6000, indicating that traffic levels are not determined solely by temperature.
- "Very few data points are near 0 temperature, but even among them, traffic volume varies widely—indicating that traffic behavior can be unpredictable in extreme cold.


#### 2> rain_1h vs Traffic Volume
- This  scatter plot show That Relationship between rainfall and traffic — how much it rained in an hour and the corresponding traffic volume.
- Most traffic occurs during low or no rainfall. Most data points are around rain_1h = 0, indicating that traffic volume is predominantly recorded during dry or lightly rainy conditions.
- Rainfall has a limited impact on traffic. Even during light rain, traffic volume does not decrease significantly, suggesting that rain does not greatly affect traffic flow.
- An unusual data point shows extremely high rainfall (~10,000 units) with traffic volume around 6000. This may be an outlier, possibly due to a data error or a special event.


#### 3> snow_1h vs Traffic Volume
- This  scatter plot show That How traffic changes when it snows.
- Snowfall occurs very rarely. Most data points are around snow_1h = 0, indicating that snowfall is infrequent in the dataset.
- Even with minimal snowfall, traffic volume ranges from 0 to over 7000, suggesting that snow does not have a significant impact on traffic.
- No clear pattern is visible in the graph, indicating that there is no strong relationship between snowfall and traffic volume.


#### 4> clouds_all vs Traffic Volume
- This  scatter plot show That Relationship between cloud cover and traffic.
- There is no clear relationship. Data points are spread across the entire cloud coverage range (0 to 100), indicating that cloudiness does not have a strong correlation with traffic volume.
- Traffic volume varies under all conditions, ranging from 0 to over 7000, suggesting that clouds do not significantly affect traffic.
- Traffic is observed across the entire x-axis, showing that it occurs under all levels of cloud cover, whether the sky is clear or fully overcast.


#### 5> hour vs Traffic Volume
- This  scatter plot show That Traffic volume for each hour.
- Traffic peaks during rush hours — traffic volume increases in the morning (7–9 AM) and evening (4–6 PM), reflecting typical office or school timings.
- Traffic is low at night — from 12 AM to 5 AM, traffic volume is minimal, indicating reduced activity during these hours.
- Traffic remains moderate and stable from 10 AM to 3 PM, reflecting non-rush hour activity.


#### 6> day_of_week vs Traffic Volume
- This  scatter plot show That On which day of the week traffic is higher or lower.
- Traffic volume is almost similar on all days (0 to 6). Data points are evenly distributed, indicating that there is no significant change in traffic on any single day.
- No clear weekly pattern is observed. There is no consistent rise or fall in traffic volume throughout the week, suggesting that traffic behavior remains relatively stable.
- Traffic volume varies each day, ranging from 0 to over 7000, indicating that traffic conditions can differ from day to day.

  
#### 7> month vs Traffic Volume
- This  scatter plot show That Traffic volume across different months of the year.
- Traffic is fairly consistent throughout the year. Data is evenly distributed across months 1 to 12, indicating that traffic volume remains relatively stable year-round.
- No clear seasonal pattern is visible. The graph shows no consistent rise or fall, suggesting that weather has little effect on traffic.
- Traffic volume varies across months, ranging from 0 to over 7000, indicating that traffic conditions can differ from month to month.


#### 8> is_weekend vs Traffic Volume
- This  scatter plot show That Difference in traffic between weekends (Saturday–Sunday) and weekdays.
- Traffic is observed on all days of the week. Data points are dense for both is_weekend = 0 and is_weekend = 1, indicating that traffic persists throughout the week.
- Traffic is higher on weekdays. The spread of traffic volume is greater on workdays, suggesting that traffic is more frequent and consistent during working days.
- Although data is slightly lower on weekends, traffic volume still reaches high levels, indicating that traffic does not drop significantly during weekends.


#### 9> is_rainy vs Traffic Volume
- This  scatter plot show That Difference in traffic between rainy days and non-rainy days.
- Traffic persists whether it rains or not. Data points exist for both is_rainy = 0 and is_rainy = 1, indicating that traffic occurs even during rainy conditions.
- More traffic is recorded during dry periods. A higher number of data points correspond to non-rainy times, suggesting that most traffic occurs when it is not raining.
- Traffic volume can still be high during rainfall, reaching over 7000, indicating that rain does not completely reduce traffic.


#### 10> is_snowy vs Traffic Volume
- This  scatter plot show That Difference in traffic between snowy days and clear days.
- Traffic persists whether it snows or not. Data points exist for both is_snowy = 0 and is_snowy = 1, indicating that traffic occurs even during snowy conditions.
- Most traffic occurs when there is no snowfall. A higher number of data points correspond to is_snowy = 0, suggesting that traffic is generally higher on non-snowy days.
- Traffic volume can still be high during snowfall, reaching over 7000, indicating that snow does not completely reduce traffic.

  
#### 11> is_rush_hour vs Traffic Volume
- This  scatter plot show That How much traffic increases during rush hours.
- Traffic exists during both rush hours and non-rush hours. Data points are dense for both is_rush_hour = 0 and is_rush_hour = 1, indicating that traffic persists throughout the day.
- More data points correspond to non-rush hours, suggesting that the majority of traffic occurs outside peak times.
- Traffic volume can reach over 7000 during rush hours, indicating significantly higher traffic during peak times.

---


### 3. Multivariate Analysis
![Multivariate Analysis heat map](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223410.png)

###  Multivariate Analysis Insights
#### 1. Traffic peaks during rush hours:
- The correlation between is_rush_hour and traffic_volume is +0.45 → indicating that traffic volume is highest during rush hours.

#### 2. Time of day affects traffic:
- The correlation between hour and traffic_volume is +0.40 → traffic volume varies according to the time of day.

#### 3. Lower traffic on weekends:
- The correlation between is_weekend and traffic_volume is −0.33 → traffic volume decreases on weekends.

#### 4. Rain and snowfall indicators are accurate:
- The correlation of is_rainy with rain_1h is +0.79 and is_snowy with snow_1h is +0.75 → these binary variables accurately reflect weather conditions.

#### 5. Clouds and rain are related:
- The correlation between clouds_all and is_rainy is +0.29 → higher cloud cover increases the likelihood of rain.

#### 6. Rush hour occurs at specific times:
- The correlation between hour and is_rush_hour is +0.35 → rush hours occur at particular times of the day.

---


## Machine Learning Models
### 1. **Linear Regression**  
![Linear Regression](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223428.png)
### Model 1 - Linear Regression Insights
- 1> R² Score = -5.77
    - This means the Linear Regression model is doing very badly.
    - ven just guessing the average traffic would be better than this model.
    - Traffic depends on complex factors, so a simple straight-line model cannot capture it.

- 2> RMSE = 5204
    - This is a big error compared to the average traffic (~3259).
    - It shows the model predictions are often far from the real traffic numbers.

- 3> Why it happens
    - Traffic changes are non-linear (rush hours, weather, weekend effects).
    - Linear Regression cannot handle these patterns well.

- 4> Business Insight
    - You cannot rely on Linear Regression for Uber traffic prediction.
    - Tree-based models like Random Forest or XGBoost are better for accurate predictions and business decisions.

- Linear Regression did very poorly (R² = -5.77, RMSE = 5204). Traffic volume cannot be predicted accurately using a simple linear model because it depends on many factors like hour, day, weather, and rush hours. Advanced models like Random Forest or XGBoost are better for real-world predictions

--- 


### 2. **Random Forest**  
![Random Forest](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223437.png)
### Model 2 - Random Forest Insights
- 1> R² = 0.9549 (very high)
    - The model explains 95% of the variance in traffic volume.
    - This means the predictions are very close to the actual traffic values.

- 2> RMSE = 424.66
    - The average prediction error is around 425 vehicles per hour.
    - Considering the mean traffic volume is about 3259, this error is acceptable and indicates good accuracy.

- 3> Conclusion
    - Random Forest is highly accurate and reliable for this dataset.
    - It clearly captures the impact of features like hour, rush hour, rainy conditions, and weekends.

- 4> Business Insight
    - For Uber, this model can be used for real-time surge pricing, driver allocation, and demand forecasting.
    - It can accurately predict patterns during rush hours, rainy days, and weekends.

- Random Forest performed extremely well (R² = 0.95, RMSE = 424.66), showing that it can accurately predict traffic volume. This makes it a reliable model for Uber’s operational decisions like surge pricing and dynamic driver allocation, capturing the effects of hour, rush hour, weekends, and weather.

---

### 3. **XGBoost**
![XGBoost](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223515.png)
### Model 3 - XGBoost Insights
- 1> R² = 0.9533 (very high)
    - The model explains around 95% of the variance in traffic volume.
    - This means the predictions are very close to the actual traffic values.

- 2> RMSE = 432.13
    - The average prediction error is approximately 432 vehicles per hour.
    - Considering the mean traffic volume of about 3259, this error is acceptable and accurate.

- 3> Conclusion
    - XGBoost, like Random Forest, is highly accurate and reliable.
    - It clearly captures the impact of important features such as hour, rush hour, rainy conditions, and weekends.

- 4> Business Insight
    - For Uber, this model can be used for real-time surge pricing, driver allocation, and demand forecasting.
    - Patterns during rush hours, rainy days, and weekends can be accurately predicted.

- XGBoost performed very well (R² = 0.953, RMSE = 432.13), indicating that it can accurately predict traffic volume. This makes it a strong model for Uber’s operational decisions like surge pricing and dynamic driver allocation, capturing the effects of hour, rush hour, weekends, and weather.

---


## Model Comparison
| Model              | RMSE       | R²        |
|--------------------|-----------:|----------:|
| Linear Regression  | 5204.47    | -5.77     |
| Random Forest      | 424.65     | 0.95      |
| XGBoost            | 432.13     | 0.95      |

### Model Comparison & Business Insights
- 1> Linear Regression
    - RMSE = 5204, R² = -5.77
    - This model is performing very poorly.
    - A negative R² means the model is worse than a baseline (mean) prediction.
    - The traffic volume pattern does not fit linear assumptions → it has non-linear and complex dependencies.

- 2> Random Forest
    - RMSE = 424.66, R² = 0.9549
    - This model is highly accurate and explains 95% of the variance.
    - It clearly captures the impact of features like hour, rush hour, rainy conditions, and weekends.
    - It can be directly used for Uber’s surge pricing and driver allocation decisions.

- 3> XGBoost
    - RMSE = 432.13, R² = 0.9533
    - This model is also highly accurate, performing close to Random Forest.
    - It is an industry-level predictive model and efficiently handles complex feature interactions.

- 4> Business Insight Summary
    - Tree-based models (Random Forest & XGBoost) are the best choice for traffic prediction.
    - Linear Regression is inadequate.
    - Rush hours, rainy days, and weekends have the highest impact on Uber demand.
    - Based on these predictions, dynamic driver allocation, real-time surge pricing, and operational planning can be improved.
 
---

### Bar chart for RMSE
![Bar chart for RMSE](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223557.png)
### Bar chart for RMSE Insights

#### 1. XGBoost performs the best:
- XGBoost has the lowest RMSE (~400), indicating that it makes the most accurate predictions.

#### 2. Random Forest also performs well:
- Random Forest has an RMSE of around ~500, showing that it is also fairly accurate, but slightly behind XGBoost.

#### 3. Linear Regression performs poorly:
- Linear Regression has a very high RMSE (above 5000), indicating that its predictions are not accurate.

---

### Bar chart for R2
![Bar chart for R2](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223617.png)
### Bar chart for R2 Insights

#### 1. XGBoost performs the best:
- XGBoost has the highest R² score (~0.0), indicating that it explains the variance in the data better than the other models.

#### 2. Random Forest performs poorly:
- Random Forest has an R² score of around (~−0.5), showing that it fails to capture the data patterns accurately.

#### 3. Linear Regression performs the worst:
- Linear Regression has an R² score of (~−5.5), indicating that it fits the data very poorly and makes large prediction errors.

---


## Feature Importance
### 2> Random Forest Feature Importance
![ Random Forest](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223502.png)
### Feature importance of random forest Insights

#### 1. Hour has the greatest impact:
- The importance score of hour is the highest (~0.8), indicating that time of day has the strongest influence on traffic volume.

#### 2. Other features have minimal impact:
- Features such as day_of_week, is_weekend, temp, month, clouds_all, is_rush_hour, rain_1h, is_rainy, is_snowy, and snow_1h have very low importance, mostly below 0.05.

#### 3. Weather-related features are the least useful:
- Weather-related features—including rain_1h, snow_1h, clouds_all, is_rainy, and is_snowy—have almost no effect on the model.

---

### 3> XGBoost Feature Importance
![XGBoost](https://github.com/Anshpatel1825/uber-traffic-volume-prediction-ml/blob/main/Screenshot%202025-09-10%20223534.png)
### Feature importance XGBoost Insights
#### 1. Hour has the greatest impact:
- The importance score of hour is the highest (~0.85) → time of day has the strongest influence on traffic volume.

#### 2. Day of the week also matters:
- The importance score of day_of_week is (~0.10) → traffic patterns vary slightly across different days.

#### 3. Rush hour has a minor contribution:
- The importance score of is_rush_hour is (~0.03) → rush hour slightly affects traffic, but much less than hour.

#### 4. All other features have minimal effect:
- Features related to weather and calendar (month, rain_1h, temp, clouds_all, is_rainy, snow_1h, is_weekend, is_snowy) have very low scores (~0.005) → these features do not significantly influence the model.

---

## Challenges Faced
- Handling **outliers in rainfall** (~9831 mm unrealistic).  
- Balancing **weekday vs weekend data**.  
- Ensuring model doesn’t overfit on time-related features.

---

##  Key Responsibilities & My Role
- Collected, cleaned, and prepared dataset.  
- Performed **EDA (Univariate, Bivariate, Multivariate)** with visualizations.  
- Built & compared **ML models** with hyperparameter tuning.  
- Extracted **feature importance** & business insights.  
- Documented the entire pipeline for **GitHub & resume showcase**.

  ---


## Benefits of the Project
- Demonstrates ability to **handle real-world data** with anomalies.  
- Proves skills in **EDA, visualization, ML modeling, evaluation**.  
- Can be directly used in **ride-hailing, logistics, or traffic management businesses**.  
- Strong **resume project** to highlight Data Science & ML skills.
  
---

## Tech Stack
- **Python**: Pandas, Numpy, Matplotlib, Seaborn  
- **ML Models**: Scikit-learn  
- **Visualization**: Matplotlib & Seaborn  
- **Version Control**: Git & GitHub

--- 
