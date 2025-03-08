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

def create_byweather_df(hour_df):
    byweather_df = hour_df.groupby("weathersit").agg({
        "date": "nunique",
        "total_rental": ["max", "min", "mean", "std"]
    }).reset_index()
    byweather_df.columns = ["weathersit", "unique_days", "max_rental", "min_rental", "mean_rental", "std_rental"]
    weather_mapping = {
        1: "Cerah/Berawan",
        2: "Mendung",
        3: "Hujan Ringan",
        4: "Hujan Deras/Salju"
    }
    byweather_df["weathersit"] = byweather_df["weathersit"].map(weather_mapping)
    weather_order = ["Cerah/Berawan", "Mendung", "Hujan Ringan", "Hujan Deras/Salju"]
    byweather_df["weathersit"] = pd.Categorical(byweather_df["weathersit"], categories=weather_order, ordered=True)
    return byweather_df

def create_weather_stats(day_df):
    weather_factors = ["humidity", "windspeed", "temperature", "feeling_temperature"]
    weather_stats_df = (
        pd.concat([
            day_df.groupby(factor)['total_rental'].sum().reset_index().assign(Factor=factor)
            for factor in weather_factors
        ])
    )
    return weather_stats_df

def create_rental_user_clustering(hour_df):
    rental_user_clustering = hour_df.groupby(['weekday', 'hour'])[['total_rental', 'casual_users', 'registered_users']].sum().unstack()
    return rental_user_clustering

day_df = pd.read_csv("day_cleaned.csv")
hour_df = pd.read_csv("hour_cleaned.csv")

day_df['date'] = pd.to_datetime(day_df['date'])
day_df['weekday'] = day_df['date'].dt.weekday
weekday_labels = ['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min']

df_filtered = day_df.copy()

hour_df.sort_values(by="date", inplace=True)
hour_df.reset_index(inplace=True)


st.set_page_config(layout="wide")
with st.sidebar:
    st.image("https://raw.githubusercontent.com/cloudyafilia/bike-sharing-dataset/e4e46a784323e096a95db215dc9093b072421935/dashboard/logo.png")
    
    min_date_day = day_df["date"].min()
    max_date_day = day_df["date"].max()

    start_date = st.sidebar.date_input("Start Date", min_date_day, min_value=min_date_day, max_value=max_date_day)
    end_date = st.sidebar.date_input("End Date", max_date_day, min_value=min_date_day, max_value=max_date_day)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    main_df_day = day_df[(day_df["date"] >= start_date) & (day_df["date"] <= end_date)]
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
byweather_df = create_byweather_df(main_df_hour)
weather_stats_df = create_weather_stats(main_df_day)
rental_user_clustering_df = create_rental_user_clustering(main_df_hour)

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

st.header("Best and Worst Performing Time Periods⏰")
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

st.header("Weather Impact on Bike Rentals🌤️")

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
    byweather_df = create_byweather_df(hour_df)
    byweather_df = byweather_df.sort_values("weathersit")
    colors = ["lightblue" if condition == "Cerah/Berawan" else "lightgrey" for condition in byweather_df["weathersit"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="mean_rental", y="weathersit", data=byweather_df, palette=colors, ax=ax)
    ax.set_xlabel("Average Rental")
    ax.set_ylabel("Weather Conditions")
    st.pyplot(fig, use_container_width=True)

st.header("Weather Factor Comparison☀️❄️")

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

st.header("Clustering Analysis with Manual Grouping👥")
col11, col12 = st.columns(2)

with col11:
    st.subheader('Heatmap of Bike Rentals by Casual Users')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(rental_user_clustering_df['casual_users'], cmap="YlGnBu", annot=False, ax=ax)  # annot=False untuk menghilangkan angka
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day")
    st.pyplot(fig, use_container_width=True)

with col12:
    st.subheader('Heatmap of Bike Rentals by Registered Users')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(rental_user_clustering_df['registered_users'], cmap="YlOrBr", annot=False, ax=ax)  # annot=False untuk menghilangkan angka
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day")
    st.pyplot(fig, use_container_width=True)