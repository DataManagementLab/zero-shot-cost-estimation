

DROP TABLE IF EXISTS "client";

CREATE TABLE "client" (
  "client_id" integer ,
  "kraj" varchar(255) DEFAULT NULL,
  "obor" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("client_id")
) ;

DROP TABLE IF EXISTS "dobito";

CREATE TABLE "dobito" (
  "client_id" integer DEFAULT NULL,
  "month_year_datum_transakce" varchar(255) ,
  "sluzba" varchar(255) ,
  "kc_dobito" decimal(10,2) 
) ;

DROP TABLE IF EXISTS "probehnuto";

CREATE TABLE "probehnuto" (
  "client_id" integer DEFAULT NULL,
  "month_year_datum_transakce" varchar(255) ,
  "sluzba" varchar(255) DEFAULT NULL,
  "kc_proklikano" decimal(10,2) 
) ;

DROP TABLE IF EXISTS "probehnuto_mimo_penezenku";

CREATE TABLE "probehnuto_mimo_penezenku" (
  "client_id" integer ,
  "Month/Year" varchar(12) ,
  "probehla_inzerce_mimo_penezenku" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("client_id","Month/Year")
) ;

