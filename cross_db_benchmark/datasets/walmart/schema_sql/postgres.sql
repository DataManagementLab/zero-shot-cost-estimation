DROP TABLE IF EXISTS "key";

CREATE TABLE "key"
(
    "store_nbr"   integer,
    "station_nbr" integer DEFAULT NULL,
    PRIMARY KEY ("store_nbr")
);

DROP TABLE IF EXISTS "station";

CREATE TABLE "station"
(
    "station_nbr" integer,
    PRIMARY KEY ("station_nbr")
);

DROP TABLE IF EXISTS "train";

CREATE TABLE "train"
(
    "date"      varchar(12),
    "store_nbr" integer,
    "item_nbr"  integer,
    "units"     integer DEFAULT NULL,
    PRIMARY KEY ("store_nbr", "date", "item_nbr")
);

