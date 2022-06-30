import os

import pandas as pd

# orig_path = '../../../../../RDB_data/raw/airline'
# target_path = '../../../../../zero-shot-data/datasets/airline'

orig_path = target_path = '../'

df_orig_data = pd.read_csv(os.path.join(orig_path, 'On_Time_On_Time_Performance_2016_1.csv'), sep='\t')
old_columns = df_orig_data.columns

columns_new_dataset = ["Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate", "Reporting_Airline",
                       "DOT_ID_Reporting_Airline", "IATA_CODE_Reporting_Airline", "Tail_Number",
                       "Flight_Number_Reporting_Airline", "OriginAirportID", "OriginAirportSeqID", "OriginCityMarketID",
                       "Origin", "OriginCityName", "OriginState", "OriginStateFips", "OriginStateName", "OriginWac",
                       "DestAirportID", "DestAirportSeqID", "DestCityMarketID", "Dest", "DestCityName", "DestState",
                       "DestStateFips", "DestStateName", "DestWac", "CRSDepTime", "DepTime", "DepDelay",
                       "DepDelayMinutes", "DepDel15", "DepartureDelayGroups", "DepTimeBlk", "TaxiOut", "WheelsOff",
                       "WheelsOn", "TaxiIn", "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrDel15",
                       "ArrivalDelayGroups", "ArrTimeBlk", "Cancelled", "CancellationCode", "Diverted",
                       "CRSElapsedTime", "ActualElapsedTime", "AirTime", "Flights", "Distance", "DistanceGroup",
                       "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "FirstDepTime",
                       "TotalAddGTime", "LongestAddGTime", "DivAirportLandings", "DivReachedDest",
                       "DivActualElapsedTime", "DivArrDelay", "DivDistance", "Div1Airport", "Div1AirportID",
                       "Div1AirportSeqID", "Div1WheelsOn", "Div1TotalGTime", "Div1LongestGTime", "Div1WheelsOff",
                       "Div1TailNum", "Div2Airport", "Div2AirportID", "Div2AirportSeqID", "Div2WheelsOn",
                       "Div2TotalGTime", "Div2LongestGTime", "Div2WheelsOff", "Div2TailNum", "Div3Airport",
                       "Div3AirportID", "Div3AirportSeqID", "Div3WheelsOn", "Div3TotalGTime", "Div3LongestGTime",
                       "Div3WheelsOff", "Div3TailNum", "Div4Airport", "Div4AirportID", "Div4AirportSeqID",
                       "Div4WheelsOn", "Div4TotalGTime", "Div4LongestGTime", "Div4WheelsOff", "Div4TailNum",
                       "Div5Airport", "Div5AirportID", "Div5AirportSeqID", "Div5WheelsOn", "Div5TotalGTime",
                       "Div5LongestGTime", "Div5WheelsOff", "Div5TailNum", ""]

old_columns_mapped = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
                      'Reporting_Airline', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline', 'Tail_Number',
                      'Flight_Number_Reporting_Airline',
                      'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin',
                      'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName',
                      'OriginWac', 'DestAirportID', 'DestAirportSeqID', 'DestCityMarketID',
                      'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestStateName',
                      'DestWac', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes',
                      'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'TaxiOut',
                      'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrTime', 'ArrDelay',
                      'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk',
                      'Cancelled', 'CancellationCode', 'Diverted', 'CRSElapsedTime',
                      'ActualElapsedTime', 'AirTime', 'Flights', 'Distance', 'DistanceGroup',
                      'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay',
                      'LateAircraftDelay', 'FirstDepTime', 'TotalAddGTime', 'LongestAddGTime',
                      'DivAirportLandings', 'DivReachedDest', 'DivActualElapsedTime',
                      'DivArrDelay', 'DivDistance', 'Div1Airport', 'Div1AirportID',
                      'Div1AirportSeqID', 'Div1WheelsOn', 'Div1TotalGTime',
                      'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum', 'Div2Airport',
                      'Div2AirportID', 'Div2AirportSeqID', 'Div2WheelsOn', 'Div2TotalGTime',
                      'Div2LongestGTime']

for c in old_columns_mapped:
    assert c in columns_new_dataset, f"{c} not found"

col_selected = [columns_new_dataset.index(c) for c in old_columns_mapped]

print("Reading combined_csv")
# combined_csv = pd.read_csv(os.path.join(target_path, 'all_flights_small.csv'), sep=',')
combined_csv = pd.read_csv(os.path.join(target_path, 'all_flights.csv'), sep=',')
print(f"With {len(combined_csv)} rows")
combined_csv = combined_csv.iloc[:, col_selected]
combined_csv.columns = old_columns

columns_sql = [
    '"Year" integer DEFAULT NULL,',
    '"Quarter" integer DEFAULT NULL,',
    '"Month" integer DEFAULT NULL,',
    '"DayofMonth" integer DEFAULT NULL,',
    '"DayOfWeek" integer DEFAULT NULL,',
    '"FlightDate" date DEFAULT NULL,',
    '"UniqueCarrier" varchar(255) DEFAULT \'NULL\',',
    '"AirlineID" integer NOT NULL,',
    '"Carrier" char(2) DEFAULT NULL,',
    '"TailNum" varchar(6) DEFAULT NULL,',
    '"FlightNum" integer DEFAULT NULL,',
    '"OriginAirportID" integer DEFAULT NULL,',
    '"OriginAirportSeqID" integer DEFAULT NULL,',
    '"OriginCityMarketID" integer DEFAULT NULL,',
    '"Origin" char(3) DEFAULT NULL,',
    '"OriginCityName" varchar(34) DEFAULT NULL,',
    '"OriginState" char(2) DEFAULT NULL,',
    '"OriginStateFips" integer DEFAULT NULL,',
    '"OriginStateName" varchar(46) DEFAULT NULL,',
    '"OriginWac" integer DEFAULT NULL,',
    '"DestAirportID" integer DEFAULT NULL,',
    '"DestAirportSeqID" integer DEFAULT NULL,',
    '"DestCityMarketID" integer DEFAULT NULL,',
    '"Dest" char(3) DEFAULT NULL,',
    '"DestCityName" varchar(34) DEFAULT NULL,',
    '"DestState" char(2) DEFAULT NULL,',
    '"DestStateFips" integer DEFAULT NULL,',
    '"DestStateName" varchar(46) DEFAULT NULL,',
    '"DestWac" integer DEFAULT NULL,',
    '"CRSDepTime" integer DEFAULT NULL,',
    '"DepTime" integer DEFAULT NULL,',
    '"DepDelay" decimal(65,2) DEFAULT NULL,',
    '"DepDelayMinutes" float DEFAULT NULL,',
    '"DepDel15" integer DEFAULT NULL,',
    '"DepartureDelayGroups" integer DEFAULT NULL,',
    '"DepTimeBlk" char(9) DEFAULT NULL,',
    '"TaxiOut" float DEFAULT NULL,',
    '"WheelsOff" integer DEFAULT NULL,',
    '"WheelsOn" integer DEFAULT NULL,',
    '"TaxiIn" float DEFAULT NULL,',
    '"CRSArrTime" integer DEFAULT NULL,',
    '"ArrTime" integer DEFAULT NULL,',
    '"ArrDelay" decimal(65,2) DEFAULT NULL,',
    '"ArrDelayMinutes" float DEFAULT NULL,',
    '"ArrDel15" integer DEFAULT NULL,',
    '"ArrivalDelayGroups" integer DEFAULT NULL,',
    '"ArrTimeBlk" char(9) DEFAULT NULL,',
    '"Cancelled" integer DEFAULT NULL,',
    '"CancellationCode" char(1) DEFAULT NULL,',
    '"Diverted" integer DEFAULT NULL,',
    '"CRSElapsedTime" float DEFAULT NULL,',
    '"ActualElapsedTime" float DEFAULT NULL,',
    '"AirTime" float DEFAULT NULL,',
    '"Flights" float DEFAULT NULL,',
    '"Distance" float DEFAULT NULL,',
    '"DistanceGroup" integer DEFAULT NULL,',
    '"CarrierDelay" decimal(65,2) DEFAULT NULL,',
    '"WeatherDelay" decimal(65,2) DEFAULT NULL,',
    '"NASDelay" decimal(65,2) DEFAULT NULL,',
    '"SecurityDelay" decimal(65,2) DEFAULT NULL,',
    '"LateAircraftDelay" decimal(65,2) DEFAULT NULL,',
    '"FirstDepTime" decimal(65,2) DEFAULT NULL,',
    '"TotalAddGTime" decimal(65,2) DEFAULT NULL,',
    '"LongestAddGTime" decimal(65,2) DEFAULT NULL,',
    '"DivAirportLandings" integer DEFAULT NULL,',
    '"DivReachedDest" decimal(65,2) DEFAULT NULL,',
    '"DivActualElapsedTime" decimal(65,2) DEFAULT NULL,',
    '"DivArrDelay" decimal(65,2) DEFAULT NULL,',
    '"DivDistance" decimal(65,2) DEFAULT NULL,',
    '"Div1Airport" char(3) DEFAULT NULL,',
    '"Div1AirportID" integer DEFAULT NULL,',
    '"Div1AirportSeqID" integer DEFAULT NULL,',
    '"Div1WheelsOn" decimal(65,2) DEFAULT NULL,',
    '"Div1TotalGTime" decimal(65,2) DEFAULT NULL,',
    '"Div1LongestGTime" decimal(65,2) DEFAULT NULL,',
    '"Div1WheelsOff" decimal(65,2) DEFAULT NULL,',
    '"Div1TailNum" varchar(6) DEFAULT NULL,',
    '"Div2Airport" char(3) DEFAULT NULL,',
    '"Div2AirportID" integer DEFAULT NULL,',
    '"Div2AirportSeqID" integer DEFAULT NULL,',
    '"Div2WheelsOn" decimal(65,2) DEFAULT NULL,',
    '"Div2TotalGTime" decimal(65,2) DEFAULT NULL,',
    '"Div2LongestGTime" decimal(65,2) DEFAULT NULL']

for c in columns_sql:
    col_name = c.split('" ')[0].strip('"')
    if ' integer ' in c:
        combined_csv[col_name] = combined_csv[col_name].astype('Int64')

combined_csv.to_csv(os.path.join(target_path, 'On_Time_On_Time_Performance_2016_1_new.csv'),
                    index=False,
                    header=True, sep='\t')

# old_columns = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
# '             'UniqueCarrier', 'AirlineID', 'Carrier', 'TailNum', 'FlightNum',
# '             'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin',
# '             'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName',
# '             'OriginWac', 'DestAirportID', 'DestAirportSeqID', 'DestCityMarketID',
# '             'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestStateName',
# '             'DestWac', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes',
# '             'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'TaxiOut',
# '             'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrTime', 'ArrDelay',
# '             'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk',
# '             'Cancelled', 'CancellationCode', 'Diverted', 'CRSElapsedTime',
# '             'ActualElapsedTime', 'AirTime', 'Flights', 'Distance', 'DistanceGroup',
# '             'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay',
# '             'LateAircraftDelay', 'FirstDepTime', 'TotalAddGTime', 'LongestAddGTime',
# '             'DivAirportLandings', 'DivReachedDest', 'DivActualElapsedTime',
# '             'DivArrDelay', 'DivDistance', 'Div1Airport', 'Div1AirportID',
# '             'Div1AirportSeqID', 'Div1WheelsOn', 'Div1TotalGTime',
# '             'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum', 'Div2Airport',
# '             'Div2AirportID', 'Div2AirportSeqID', 'Div2WheelsOn', 'Div2TotalGTime',
# '             'Div2LongestGTime']
