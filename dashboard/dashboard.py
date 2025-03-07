import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

def total_registered_df(day_df):
   reg_df =  day_df.groupby(by="date").agg({
      "registered_users": "sum"
    })
   reg_df = reg_df.reset_index()
   reg_df.rename(columns={
        "registered_users": "register_sum"
    }, inplace=True)
   return reg_df

def total_casual_df(day_df):
   cas_df =  day_df.groupby(by="date").agg({
      "casual_users": ["sum"]
    })
   cas_df = cas_df.reset_index()
   cas_df.rename(columns={
        "casual_users": "casual_sum"
    }, inplace=True)
   return cas_df

def sum_order (hour_df):
    sum_order_items_df = hour_df.groupby("hour").total_rental.sum().sort_values(ascending=False).reset_index()
    return sum_order_items_df

def create_hourly_rentals_df(hour_df):
    hourly_rentals_df = hour_df.groupby('hour').total_rental.sum().reset_index()
    hourly_rentals_df.rename(columns={"total_rental": "total_rental"}, inplace=True)
    return hourly_rentals_df

def create_daily_rentals_df(df):
    daily_rentals_df = df.groupby('weekday')['total_rental'].mean().reset_index()
    daily_rentals_df.rename(columns={'total_rental': 'average_rentals_daily'}, inplace=True)
    return daily_rentals_df

def create_monthly_rentals_df(day_df):
    expected_months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
    monthly_rentals = day_df.groupby('month')['total_rental'].mean().reset_index()
    monthly_rentals.rename(columns={'total_rental': 'average_rentals_daily'}, inplace=True)
    monthly_rentals = monthly_rentals.sort_values(by='month')
    return monthly_rentals.sort_values('month')

def create_byseasons_df(day_df):
    byseason_df = day_df.groupby('season')['total_rental'].mean().reset_index()
    season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    byseason_df['season'] = byseason_df['season'].map(season_labels)
    byseason_df.rename(columns={"total_rental": "average_rental"}, inplace=True)
    return byseason_df

def create_byweather_df(day_df):
    byweather_df = day_df.groupby('weathersit').agg({
        "date": "nunique",
        "total_rental": ["max", "min", "mean", "std"]
    }).reset_index()
    weather_labels = {1: "Cerah/Berawan", 2: "Mendung", 3: "Hujan Ringan", 4: "Hujan Deras/Salju"}
    byweather_df['weathersit'] = byweather_df['weathersit'].map(weather_labels)
    byweather_df.columns = ['weathersit', 'unique_days', 'max_rental', 'min_rental', 'mean_rental', 'std_rental']
    return byweather_df

def create_weekend_weekday_df(day_df):
    weekend_weekday_df = day_df.groupby(
        day_df["workingday"].map({0: "Weekend", 1: "Weekday"})
    ).agg({
        "casual_users": "sum",
        "registered_users": "sum",
        "total_rental": "sum"
    }).reset_index()
    weekend_weekday_df.rename(columns={"total_rental": "total_rental"}, inplace=True)
    return weekend_weekday_df

def create_weather_stats(day_df):
    weather_factors = ["humidity", "windspeed", "temperature", "feeling_temperature"]
    weather_stats_df = (
        pd.concat([
            day_df.groupby(factor)['total_rental'].sum().reset_index().assign(Factor=factor)
            for factor in weather_factors
        ])
    )
    return weather_stats_df

def create_rfm_analysis(day_df):
    last_date = day_df['date'].max()
    recency_df = day_df.groupby('registered_users')['date'].max().reset_index()
    recency_df['recency'] = (pd.to_datetime(last_date) - pd.to_datetime(recency_df['date'])).dt.days
    recency_df = recency_df[['registered_users', 'recency']]
    frequency_df = day_df.groupby('registered_users')['total_rental'].count().reset_index()
    frequency_df.columns = ['registered_users', 'frequency']
    monetary_df = day_df.groupby('registered_users')['total_rental'].sum().reset_index()
    monetary_df.columns = ['registered_users', 'monetary']
    rfm_df = recency_df.merge(frequency_df, on='registered_users').merge(monetary_df, on='registered_users')
    return rfm_df

def create_rfm_summary(rfm_df):
    rfm_summary = {
        "total_customers": rfm_df.shape[0],
        "avg_recency": rfm_df['recency'].mean(),
        "avg_frequency": rfm_df['frequency'].mean(),
        "total_monetary": rfm_df['monetary'].sum()
    }
    return rfm_summary

day_df = pd.read_csv("dashboard/day_cleaned.csv")
hour_df = pd.read_csv("dashboard/hour_cleaned.csv")

day_df['date'] = pd.to_datetime(day_df['date'])
day_df['weekday'] = day_df['date'].dt.weekday
weekday_labels = ['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min']

datetime_columns = ["date"]
day_df.sort_values(by="date", inplace=True)
day_df.reset_index(inplace=True)   

df_filtered = day_df.copy()

hour_df.sort_values(by="date", inplace=True)
hour_df.reset_index(inplace=True)
 
for column in datetime_columns:
     day_df[column] = pd.to_datetime(day_df[column])
     hour_df[column] = pd.to_datetime(hour_df[column])

min_date_day = day_df["date"].min()
max_date_day = day_df["date"].max()

min_date_hour = hour_df["date"].min()
max_date_hour = hour_df["date"].max()

st.set_page_config(layout="wide")
with st.sidebar:
    st.image("https://raw.githubusercontent.com/cloudyafilia/bike-sharing-dataset/e4e46a784323e096a95db215dc9093b072421935/dashboard/logo.png")
    
    start_date, end_date = st.date_input(
        label='Time Range',
        min_value=min_date_day,
        max_value=max_date_day,
        value=[min_date_day, max_date_day])

main_df_day = day_df[(day_df["date"] >= str(start_date)) & 
                     (day_df["date"] <= str(end_date))]

main_df_hour = hour_df[(hour_df["date"] >= str(start_date)) & 
                       (hour_df["date"] <= str(end_date))]

st.title('Bike Sharing Dashboard 🚲')

reg_df = total_registered_df(main_df_day)
cas_df = total_casual_df(main_df_day)
sum_order_items_df = sum_order(main_df_hour)
hourly_rentals_df = create_hourly_rentals_df(main_df_hour)
daily_rentals_df = create_daily_rentals_df(main_df_hour)
monthly_rentals_df = create_monthly_rentals_df(main_df_day)
byseason_df = create_byseasons_df(main_df_day)
byweather_df = create_byweather_df(main_df_day)
weekend_weekday_df = create_weekend_weekday_df(main_df_day)
weather_stats_df = create_weather_stats(main_df_day)
rfm_df = create_rfm_analysis(main_df_day)
rfm_summary = create_rfm_summary(rfm_df)

col1, col2, col3 = st.columns(3)
with col1:
    total_orders = day_df.total_rental.sum()
    st.metric("Total Sharing Bike", value=total_orders)

with col2:
    total_sum = reg_df.register_sum.sum()
    st.metric("Total Registered Users", value=total_sum)

with col3:
    total_sum = cas_df.casual_sum.sum()
    st.metric("Total Casual Users", value=total_sum)

st.header("Best and Worst Performing Time Periods")
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader('Hourly Rentals')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(hourly_rentals_df['hour'], hourly_rentals_df['total_rental'], color='lightblue', alpha=0.5)
    ax.plot(hourly_rentals_df['hour'], hourly_rentals_df['total_rental'], marker='o', linestyle='-', color='blue')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average Rental")
    st.pyplot(fig, use_container_width=True)

weekly_rentals = create_daily_rentals_df(day_df)
with col5:
    st.subheader('Daily Rentals')
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_color = "lightblue"
    line_color = "blue"
    sns.barplot(x='weekday', y='average_rentals_daily', data=weekly_rentals, color=bar_color, alpha=0.7, ax=ax)
    bar_positions = np.arange(len(weekly_rentals))
    ax.plot(bar_positions, weekly_rentals['average_rentals_daily'], marker='o', linewidth=2, color=line_color)
    ax.set_xlabel("Day")
    ax.set_ylabel("Average Rental")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(weekday_labels)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig, use_container_width=True)

monthly_rentals = create_monthly_rentals_df(day_df)
with col6:
    st.subheader('Monthly Rentals')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='month', y='average_rentals_daily', data=monthly_rentals_df, color="lightblue", alpha=0.7, ax=ax)
    ax.plot(monthly_rentals['month'] - 1,  
        monthly_rentals['average_rentals_daily'],
        color=line_color, marker='o', linestyle='-', linewidth=2, markersize=6)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Rental")
    ax.set_xticks(range(12))  
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Agt', 'Sep', 'Okt', 'Nov', 'Des'])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig, use_container_width=True)

st.header("Weather Impact on Bike Rentals")

with st.container():
    col7, col8 = st.columns(2)

with col7:
    st.subheader('Bike Rentals by Season')
    max_value = byseason_df['average_rental'].max()
    colors = ['lightblue' if val == max_value else 'lightgrey' for val in byseason_df['average_rental']]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(y='season', x='average_rental', data=byseason_df, palette=colors, ax=ax)
    ax.set_xlabel("Average Rental")
    ax.set_ylabel("Season")
    st.pyplot(fig, use_container_width=True)

with col8:
    st.subheader('Bike Rentals by Weather')
    max_value = byweather_df['mean_rental'].max()
    colors = ['lightblue' if val == max_value else 'lightgrey' for val in byweather_df['mean_rental']]
    weather_labels = {
        "Cerah/Berawan": "Clear/Cloudy",
        "Mendung": "Overcast",
        "Hujan Ringan": "Light Rain",
        "Hujan Deras/Salju": "Heavy Rain/Snow"
    }
    byweather_df['weather_sit'] = byweather_df['weathersit'].replace(weather_labels)
    order = [weather_labels[label] for label in ["Cerah/Berawan", "Mendung", "Hujan Ringan", "Hujan Deras/Salju"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y='weather_sit', x='mean_rental', data=byweather_df, palette=colors, ax=ax, order=order)
    ax.set_xlabel("Average Rental")
    ax.set_ylabel("Weather")
    ax.set_xlim([0, byweather_df['mean_rental'].max() * 1.1])
    st.pyplot(fig, use_container_width=True)

st.header("Customer Demographics")

st.subheader('Bike Rentals by Casual Vs Registered Users on Weekdays and Weekends')
categories = weekend_weekday_df["workingday"].values 
casual_users = weekend_weekday_df["casual_users"].values
registered_users = weekend_weekday_df["registered_users"].values
x = np.arange(len(categories))
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - 0.2, registered_users, width=0.4, label="Registered Users", color="blue")
ax.bar(x + 0.2, casual_users, width=0.4, label="Casual Users", color="lightblue")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Total Rental")
ax.set_xlabel("Day Category")
ax.legend()
ax.ticklabel_format(style="plain", axis="y")
for i in range(len(categories)):
        ax.text(x[i] - 0.2, registered_users[i] + 50000, f"{registered_users[i]:,}", ha='center', fontsize=10, color='black')
        ax.text(x[i] + 0.2, casual_users[i] + 50000, f"{casual_users[i]:,}", ha='center', fontsize=10, color='black')
st.pyplot(fig, use_container_width=True)


st.header("Weather Factor Comparison")

col9, col10 = st.columns(2)
with col9:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(x=day_df['temperature'], y=day_df['total_rental'], ax=ax, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    ax.set_title("Effect of Temperature on Bike Sharing")
    ax.set_xlabel("Temperature (Normalized)")
    ax.set_ylabel("Total Rental")
    st.pyplot(fig, use_container_width=True)  

with col10:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(x=day_df['feeling_temperature'], y=day_df['total_rental'], ax=ax, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    ax.set_title("Effect of Feeling Temperature on Bike Sharing")
    ax.set_xlabel("Feeling Temperature (Normalized)")
    ax.set_ylabel("Total Rental")
    st.pyplot(fig, use_container_width=True)

with col9:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(x=day_df['humidity'], y=day_df['total_rental'], ax=ax, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    ax.set_title("Effect of Humidity on Bike Sharing")
    ax.set_xlabel("Humidity (Normalized)")
    ax.set_ylabel("Total Rental")
    st.pyplot(fig, use_container_width=True)

with col10:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(x=day_df['windspeed'], y=day_df['total_rental'], ax=ax, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    ax.set_title("Effect of Windspeed on Bike Sharing")
    ax.set_xlabel("Windspeed (Normalized)")
    ax.set_ylabel("Total Rental")
    st.pyplot(fig, use_container_width=True)

st.header("Visualisasi RFM Analysis")
with st.container():
        col11, col12, col13 = st.columns(3)
        with col11:
            st.metric(label="Average Recency", value=f"{rfm_summary['avg_recency']:.1f} days")
        
        with col12:
            st.metric(label="Average Frequency", value=f"{rfm_summary['avg_frequency']:.1f} times")
        
        with col13:
            st.metric(label="Total Revenue", value=f"AUD {rfm_summary['total_monetary']:,.0f}")

col14, col15, col16 = st.columns(3)
figsize = (5, 4)  
with col14:
    st.subheader("Recency Distribution")
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(rfm_df['recency'], bins=20, kde=True, color='blue', ax=ax)
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Total Rental")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col15:
    st.subheader("Frequency Distribution")
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(rfm_df['frequency'], bins=20, kde=True, color='green', ax=ax)
    ax.set_xlabel("Frequency (times)")
    ax.set_ylabel("Total Rental")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with col16:
    st.subheader("Monetary Distribution")
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(rfm_df['monetary'], bins=20, kde=True, color='red', ax=ax)
    ax.set_xlabel("Monetary (total rentals)")
    ax.set_ylabel("JTotal Rental")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
