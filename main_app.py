import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('cars.csv')
st.markdown('# Car prices')


## data-processing ###################################################
######################################################################
# rozdelit znacka-model

df = df.rename(columns={
    'CarName':'car_name',
    'fueltype': 'fuel_type',
    'doornumber': 'door_number',
    'carbody':'car_body',
    'enginelocation':'engine_location',
    'wheelbase': 'wheel_base',
    'carlength': 'car_length',
    'carwidth': 'car_width',
    'curbweight':'curb_weight',
    'enginetype': 'engine_type',
    'cylindernumber': 'cylinder_number',
    'enginesize':'engine_size',
    'fuelsystem': 'fuel_system',
    'boreratio': 'bore_ratio',
    'compressionratio':'compression_ratio',
    'horsepower':'horse_power',
    'peakrpm':'peak_rpm',
    'citympg':'city_mpg',
    'highwaympg':'highway_mpg',
    'drivewheel':'wheel_drive'
    })

# rozdelit jmena na brand - model
df['car_name_split'] = df['car_name'].str.split(' ')
df['brand'] = df['car_name_split'].apply(lambda x: x[0])
df['model'] = df['car_name_split'].apply(lambda x: " ".join(x[1:]))

# # nulove hodnoty
# st.write(df.isna().sum().sort_values(ascending=False))

## uprava spatnych jmen
df['brand'] = df['brand'].replace({
    'alfa-romero':'alfa-romeo',
    'vokswagen':'volkswagen',
    'toyouta':'toyota',
    'Nissan':'nissan',
    'maxda':'mazda',
    'porcshce':'porsche'
    })

## mapping pro cylinder number
df['cylinder_number'] = df['cylinder_number'].map({
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'eight': 8,
    'twelve': 12
})

## stranky pro jednotlive pohledy
page = st.sidebar.radio('Page', ['Visual', 'Regression'])

if page == 'Visual':
    
    # grouping - zbavit se duplicitnich modelu, bran prumer hodnot
    df = df[['brand', 'model', 'wheel_drive', 'fuel_type', 'cylinder_number', 'price', 'horse_power']]
    df = df.groupby(['brand', 'model', 'wheel_drive', 'fuel_type']).mean().reset_index()

    # # nefunguje info, zjistime datatypy jinak
    # for column in df.columns:
    #     st.write(f'{column}:{df[column].dtype}')


    # # filters #########################################################
    #####################################################################

    ## models and brand
    brand_opt = df['brand'].unique()
    brand_pick = st.sidebar.multiselect('Brand', brand_opt)

    if brand_pick:
        df = df[df['brand'].isin(brand_pick)]

    ## wheel drive
    st.sidebar.write('Wheel drive')
    wheel_drive_filter_base = []
    wheel_drive_filter_base.append(st.sidebar.checkbox('4wd', value=True))
    wheel_drive_filter_base.append(st.sidebar.checkbox('fwd', value=True))
    wheel_drive_filter_base.append(st.sidebar.checkbox('rwd', value=True))

    wheel_drive_filter = []
    for position, value in zip(wheel_drive_filter_base, ('4wd', 'fwd', 'rwd')):
        
        if position:
            wheel_drive_filter.append(value)

    df = df[df['wheel_drive'].isin(wheel_drive_filter)]

    ## cylinders
    min_c, max_c = st.sidebar.select_slider('Cylinders', options=range(2,13), value=[2,12])

    df = df[(df['cylinder_number'] >= min_c) & (df['cylinder_number'] <= max_c)]

    ## charts ######################################################
    ################################################################

    # # price distribution
    pr_dst = alt.Chart(df).mark_bar().encode(
        x=alt.X(
            'price:Q', 
            bin=alt.Bin(maxbins=25),
            axis=alt.Axis(title='price')
        ),
        y=alt.Y(
            'count()',
            axis=alt.Axis(title='count'),
        )
    )

    # # entries per brand / price vs hp per models
    if brand_pick:
        base = alt.Chart(df).mark_point().encode(
            x='price',
            y='horse_power',
            tooltip=['brand', 'model', 'price', 'horse_power']
        )
        
        text = base.mark_text(
            align='left',
            baseline='middle',
            dx=7
        ).encode(
            text='model'
        )

        second_chart = (base + text).interactive()

    else:
        second_chart = alt.Chart(df).mark_bar(size=10).encode(
            x='count()',
            y=alt.Y('brand', sort='-x')
        ).properties(
            height=300
        )



    # # fuel type
    fuel_source = df.groupby('fuel_type').count().reset_index()[['fuel_type', 'brand']]
    fuels = fuel_source['fuel_type'].to_list()
    counts = fuel_source['brand'].to_list()

    fuel_pie = px.pie(values=counts, names=fuels)

    # # cylinders
    cylinder_source = df.groupby('cylinder_number').count().reset_index()[['cylinder_number', 'brand']]
    cylinders = cylinder_source['cylinder_number'].to_list()
    counts = cylinder_source['brand'].to_list()

    cylinder_pie = px.pie(values=counts, names=cylinders)


    # # power per price
    hp = alt.Chart(df).mark_circle(size=20).encode(
        x=alt.X('price:Q'),
        y=alt.Y('horse_power:Q'),
        tooltip=['brand', 'model', 'price', 'horse_power']
    ).interactive()



    # # hp density plot
    fig = plt.figure()
    sns.displot(df, x="horse_power", kind="kde")

    c1, c2 = st.columns((1,1))

    c1.markdown('## Price distribution')
    c1.altair_chart(pr_dst, use_container_width=True)

    if brand_pick:
        c2.markdown(f'## {", ".join(brand_pick)}')
    else:
        c2.markdown('## Models per brand')
    c2.altair_chart(second_chart, use_container_width=True)

    c21, c22, c23 = st.columns((1,1,1))

    c21.markdown('## Fuel types')
    c21.plotly_chart(fuel_pie, use_container_width=True)

    c22.markdown('## Cylinders')
    c22.plotly_chart(cylinder_pie, use_container_width=True)

    c23.markdown('## Horsepower distribution')
    c23.pyplot(plt.gcf(), use_container_width=True)


## skocit do jupyter notebooku predtim, vysvetlit regresi
elif page == 'Regression':

    cols = ['wheel_base', 'car_length', 'car_width', 'curb_weight', 'cylinder_number', 'engine_size', 'bore_ratio', 'horse_power', 'city_mpg', 'highway_mpg', 'price']
    df_reg = df[cols]

    ## train test split
    df_train, df_test = train_test_split(df_reg, train_size=0.7, test_size=0.3, random_state=100)

    y_train = df_train.pop('price')
    x_train = df_train

    y_test = df_test.pop('price')
    x_test = df_test

    ## scaling and training
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_train_df = pd.DataFrame(x_train_scaled, index=x_train.index, columns=x_train.columns)

    reg = linear_model.LinearRegression()
    reg.fit(x_train_df, y_train)


    c1, c2 = st.columns((1,3))
    with c1.form("Price predictor"):
        wheel_base = st.slider('Wheel base', min_value=70, max_value=140)
        car_length = st.slider('Car length', min_value=130, max_value=230)
        car_width = st.slider('Car width', min_value=50, max_value=90)
        weight = st.slider('Weight', min_value=1300, max_value=4400)
        cylinder_number = st.slider('Cylinders', min_value=2, max_value=12)
        engine_size = st.slider('Engine size', min_value=50, max_value=350)
        bore_ratio = st.slider('Bore ratio', min_value=2.4, max_value=4.2)
        horse_power = st.slider('Horse power', min_value=40, max_value=350)
        city_mpg = st.slider('City mpg', min_value=10, max_value=55)
        highway_mpg = st.slider('Highway mpg', min_value=10, max_value=55)

        submitted = st.form_submit_button("Predict!")
        if submitted:
            prediction_features = [wheel_base, car_length, car_width, weight, cylinder_number, engine_size, bore_ratio, horse_power, city_mpg, highway_mpg]

            regressor_prep = pd.DataFrame(prediction_features, index=x_test.columns).T
            reg_prep_scaled = pd.DataFrame(scaler.transform(regressor_prep), columns=x_test.columns)
            
            prediction_result = round(reg.predict(reg_prep_scaled)[0])

            final_text = f"Car with given parameters could be worth around ${prediction_result}."
            markd = c2.markdown(f'## {final_text}')








