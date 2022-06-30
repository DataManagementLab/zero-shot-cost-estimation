

DROP TABLE IF EXISTS "L_AIRLINE_ID";

CREATE TABLE "L_AIRLINE_ID" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_AIRPORT";

CREATE TABLE "L_AIRPORT" (
  "Code" char(3) NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_AIRPORT_ID";

CREATE TABLE "L_AIRPORT_ID" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_AIRPORT_SEQ_ID";

CREATE TABLE "L_AIRPORT_SEQ_ID" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_CANCELLATION";

CREATE TABLE "L_CANCELLATION" (
  "Code" char(1) NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_CITY_MARKET_ID";

CREATE TABLE "L_CITY_MARKET_ID" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_DEPARRBLK";

CREATE TABLE "L_DEPARRBLK" (
  "Code" char(9) NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_DISTANCE_GROUP_250";

CREATE TABLE "L_DISTANCE_GROUP_250" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_DIVERSIONS";

CREATE TABLE "L_DIVERSIONS" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_MONTHS";

CREATE TABLE "L_MONTHS" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_ONTIME_DELAY_GROUPS";

CREATE TABLE "L_ONTIME_DELAY_GROUPS" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_QUARTERS";

CREATE TABLE "L_QUARTERS" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_STATE_ABR_AVIATION";

CREATE TABLE "L_STATE_ABR_AVIATION" (
  "Code" char(2) NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_STATE_FIPS";

CREATE TABLE "L_STATE_FIPS" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_UNIQUE_CARRIERS";

CREATE TABLE "L_UNIQUE_CARRIERS" (
  "Code" varchar(255) NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_WEEKDAYS";

CREATE TABLE "L_WEEKDAYS" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_WORLD_AREA_CODES";

CREATE TABLE "L_WORLD_AREA_CODES" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "L_YESNO_RESP";

CREATE TABLE "L_YESNO_RESP" (
  "Code" integer NOT NULL,
  "Description" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("Code")
);

DROP TABLE IF EXISTS "On_Time_On_Time_Performance_2016_1";

CREATE TABLE "On_Time_On_Time_Performance_2016_1" (
  "Year" integer DEFAULT NULL,
  "Quarter" integer DEFAULT NULL,
  "Month" integer DEFAULT NULL,
  "DayofMonth" integer DEFAULT NULL,
  "DayOfWeek" integer DEFAULT NULL,
  "FlightDate" date DEFAULT NULL,
  "UniqueCarrier" varchar(255) DEFAULT 'NULL',
  "AirlineID" integer NOT NULL,
  "Carrier" char(2) DEFAULT NULL,
  "TailNum" varchar(6) DEFAULT NULL,
  "FlightNum" integer DEFAULT NULL,
  "OriginAirportID" integer DEFAULT NULL,
  "OriginAirportSeqID" integer DEFAULT NULL,
  "OriginCityMarketID" integer DEFAULT NULL,
  "Origin" char(3) DEFAULT NULL,
  "OriginCityName" varchar(34) DEFAULT NULL,
  "OriginState" char(2) DEFAULT NULL,
  "OriginStateFips" integer DEFAULT NULL,
  "OriginStateName" varchar(46) DEFAULT NULL,
  "OriginWac" integer DEFAULT NULL,
  "DestAirportID" integer DEFAULT NULL,
  "DestAirportSeqID" integer DEFAULT NULL,
  "DestCityMarketID" integer DEFAULT NULL,
  "Dest" char(3) DEFAULT NULL,
  "DestCityName" varchar(34) DEFAULT NULL,
  "DestState" char(2) DEFAULT NULL,
  "DestStateFips" integer DEFAULT NULL,
  "DestStateName" varchar(46) DEFAULT NULL,
  "DestWac" integer DEFAULT NULL,
  "CRSDepTime" integer DEFAULT NULL,
  "DepTime" integer DEFAULT NULL,
  "DepDelay" decimal(65,2) DEFAULT NULL,
  "DepDelayMinutes" float DEFAULT NULL,
  "DepDel15" integer DEFAULT NULL,
  "DepartureDelayGroups" integer DEFAULT NULL,
  "DepTimeBlk" char(9) DEFAULT NULL,
  "TaxiOut" float DEFAULT NULL,
  "WheelsOff" integer DEFAULT NULL,
  "WheelsOn" integer DEFAULT NULL,
  "TaxiIn" float DEFAULT NULL,
  "CRSArrTime" integer DEFAULT NULL,
  "ArrTime" integer DEFAULT NULL,
  "ArrDelay" decimal(65,2) DEFAULT NULL,
  "ArrDelayMinutes" float DEFAULT NULL,
  "ArrDel15" integer DEFAULT NULL,
  "ArrivalDelayGroups" integer DEFAULT NULL,
  "ArrTimeBlk" char(9) DEFAULT NULL,
  "Cancelled" integer DEFAULT NULL,
  "CancellationCode" char(1) DEFAULT NULL,
  "Diverted" integer DEFAULT NULL,
  "CRSElapsedTime" float DEFAULT NULL,
  "ActualElapsedTime" float DEFAULT NULL,
  "AirTime" float DEFAULT NULL,
  "Flights" float DEFAULT NULL,
  "Distance" float DEFAULT NULL,
  "DistanceGroup" integer DEFAULT NULL,
  "CarrierDelay" decimal(65,2) DEFAULT NULL,
  "WeatherDelay" decimal(65,2) DEFAULT NULL,
  "NASDelay" decimal(65,2) DEFAULT NULL,
  "SecurityDelay" decimal(65,2) DEFAULT NULL,
  "LateAircraftDelay" decimal(65,2) DEFAULT NULL,
  "FirstDepTime" decimal(65,2) DEFAULT NULL,
  "TotalAddGTime" decimal(65,2) DEFAULT NULL,
  "LongestAddGTime" decimal(65,2) DEFAULT NULL,
  "DivAirportLandings" integer DEFAULT NULL,
  "DivReachedDest" decimal(65,2) DEFAULT NULL,
  "DivActualElapsedTime" decimal(65,2) DEFAULT NULL,
  "DivArrDelay" decimal(65,2) DEFAULT NULL,
  "DivDistance" decimal(65,2) DEFAULT NULL,
  "Div1Airport" char(3) DEFAULT NULL,
  "Div1AirportID" integer DEFAULT NULL,
  "Div1AirportSeqID" integer DEFAULT NULL,
  "Div1WheelsOn" decimal(65,2) DEFAULT NULL,
  "Div1TotalGTime" decimal(65,2) DEFAULT NULL,
  "Div1LongestGTime" decimal(65,2) DEFAULT NULL,
  "Div1WheelsOff" decimal(65,2) DEFAULT NULL,
  "Div1TailNum" varchar(6) DEFAULT NULL,
  "Div2Airport" char(3) DEFAULT NULL,
  "Div2AirportID" integer DEFAULT NULL,
  "Div2AirportSeqID" integer DEFAULT NULL,
  "Div2WheelsOn" decimal(65,2) DEFAULT NULL,
  "Div2TotalGTime" decimal(65,2) DEFAULT NULL,
  "Div2LongestGTime" decimal(65,2) DEFAULT NULL
);

