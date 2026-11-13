
import pandas as pd

def minutes(value):
    h,m=map(int,value.split(":"))
    return h*60+m

def find_clashes(df):
    required={"faculty","day","start","end","room","course"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")
    found=[]
    for i in range(len(df)):
        for j in range(i+1,len(df)):
            x,y=df.iloc[i],df.iloc[j]
            if x["day"].strip().lower()!=y["day"].strip().lower():
                continue
            overlap=max(minutes(x["start"]),minutes(y["start"])) < min(minutes(x["end"]),minutes(y["end"]))
            if overlap and x["faculty"].strip().lower()==y["faculty"].strip().lower():
                found.append(("Faculty",x["faculty"]))
            if overlap and x["room"].strip().lower()==y["room"].strip().lower():
                found.append(("Room",x["room"]))
    return found

def test_sample_timetable_has_two_conflicts():
    df=pd.read_csv("data/demo_timetable.csv",dtype=str).fillna("")
    assert len(find_clashes(df)) == 2
