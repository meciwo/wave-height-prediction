import pandas as pd
import numpy as np


def preprocessing(df):

    """
    df["Heading_sin"] = df["Heading"].apply(np.sin)
    df["Heading_cos"] = df["Heading"].apply(np.cos)
    df["Wave_Direction(Primary)_sin"] = df["Wave Direction(Primary)"].apply(np.sin)
    df["Wave_Direction(Primary)_cos"] = df["Wave Direction(Primary)"].apply(np.cos)
    df["Wave_Direction(2nd)_sin"] = df["Wave Direction(2nd)"].apply(np.sin)
    df["Wave_Direction(2nd)_cos"] = df["Wave Direction(2nd)"].apply(np.cos)
    df["Wave_Direction(3rd)_sin"] = df["Wave Direction(3rd)"].apply(np.sin)
    df["Wave_Direction(3rd)_cos"] = df["Wave Direction(3rd)"].apply(np.cos)
    df["Wave_Direction(Radar)_sin"] = df["Wave Direction(Radar)"].apply(np.sin)
    df["Wave_Direction(Radar)_cos"] = df["Wave Direction(Radar)"].apply(np.cos)
    df["Speed(Water)_sin"] = df["Speed(Water)"] * df["Heading_sin"]
    df["Speed(Water)_cos"] = df["Speed(Water)"] * df["Heading_cos"]
    df["Wind Angle(REL_Mean)_sin"] = df["Wind Angle(REL)"].apply(np.sin)
    df["Wind Angle(REL)_cos"] = df["Wind Angle(REL)"].apply(np.cos)

    

    df["us"]=df["Wind Speed(REL)"]*(
        df["Wind Angle(REL)"].apply(lambda x:np.sin(x*np.pi/180)))+df["Speed(Ground)"]*(
        df["Heading"].apply(lambda x:np.sin(x*np.pi/180)))
    df["vs"]=df["Wind Speed(REL)"]*(
        df["Wind Angle(REL)"].apply(lambda x:np.cos(x*np.pi/180)))+df["Speed(Ground)"]*(
        df["Heading"].apply(lambda x:np.cos(x*np.pi/180)))
    df["windspeed_ship"]=np.sqrt((df.us)**2+(df.vs)**2)
    df["Water_speed"] = df["Speed(Water)"]-df["Speed(Ground)"]
 """ 
    
    
    
    
    
    df=df.replace('********', np.nan)
    df=df.replace("--",np.nan)
    df = df.replace('%%%%%%%%',np.nan)
    df = df.replace("NaN",np.nan)
    df = df.replace("nan",np.nan)
    df = df.replace("*", np.nan)
    df=df.replace('****************',np.nan)
    
    df["u10"] = df["u10"].astype("float")
    df["v10"] = df["v10"].astype("float")
    df["windspeed_era5"] = np.sqrt((df.u10**2)+(df.v10)**2)
    
    df["mwd"] = df["mwd"].astype("float")
    df["mwp"] = df["mwp"].astype("float")
    df["swh"] = df["swh"].astype("float")
    df["shww"] = df["shww"].astype("float")
    df["shts"] = df["shts"].astype("float")
    

    df = df.rename(columns={'Significant Wave Height（Fore）': "swh_fore", 'Significant Wave Height(Port)': "swh_port",
                           'Significant Wave Height(Stbd.)': "swh_stbd", "Significant Wave Period(Port)":"swp_port", 'Significant Wave Period(Fore)':"swp_fore",
                           'Significant Wave Period(Stbd.)':"swp_stbd",
                            
                            "Pitching(Motion_Mean)":"pitching_mean",
                             "Pitching(Motion_Standard)":"pitching_standard",
                            "Pitching(Motion_Max)":"pitching_max",
                            "Pitching(Motion_Min)":"pitching_min",
                            
                            "Heaving(Motion_Mean)":"heaving_mean",
                            "Heaving(Motion_Standard)":"heaving_standard",
                            "Heaving(Motion_Max)":"heaving_max",
                            "Heaving(Motion_Min)":"heaving_min",
                            
                            "Rolling(Motion_Mean)":"rolling_mean",
                           "Rolling(Motion_Standard)":"rolling_standard",
                            "Rolling(Motion_Max)":"rolling_max",
                            "Rolling(Motion_Min)":"rolling_min",
                            })
    
    df["swh_max"] = df[["swh_fore","swh_stbd","swh_port"]].max(axis=1)
    df["swp_mean"] = df[["swp_fore" ,"swp_stbd","swp_port"]].mean(axis=1)
    
    return df