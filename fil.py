import pandas as pd
df=pd.read_csv("used-car-dataset.csv")
df["Brand"]=df["Name"].str.split(expand=True)[0]
df["lenght_of_names"]=df["Name"].apply(lambda x: len(x.split()))
def extract_brand_model(car_name):
    # Split the string and take the most words that can describe the model of the vehicle
    words = car_name.split()
    if len(words) == 3:
        return words[1]
    elif len(words) > 3:
        return ' '.join(words[1:3])
    else:
        return car_name
df["Model"]=df["Name"].apply(extract_brand_model)
df.to_csv('car-dataset.csv')