

DROP TABLE IF EXISTS "atom";

CREATE TABLE "atom" (
  "atomid" char(13) ,
  "drug" char(10) DEFAULT NULL,
  "atomtype" char(100) DEFAULT NULL,
  "charge" char(100) DEFAULT NULL,
  "name" char(2) DEFAULT NULL,
  PRIMARY KEY ("atomid")
) ;

DROP TABLE IF EXISTS "canc";

CREATE TABLE "canc" (
  "drug_id" char(10) ,
  "class" char(1) DEFAULT NULL,
  PRIMARY KEY ("drug_id")
) ;

DROP TABLE IF EXISTS "sbond_1";

CREATE TABLE "sbond_1" (
  "id" integer ,
  "drug" char(10) DEFAULT NULL,
  "atomid" char(100) DEFAULT NULL,
  "atomid_2" char(100) DEFAULT NULL,
  PRIMARY KEY ("id")
) ;

DROP TABLE IF EXISTS "sbond_2";

CREATE TABLE "sbond_2" (
  "id" integer ,
  "drug" char(10) DEFAULT NULL,
  "atomid" char(100) DEFAULT NULL,
  "atomid_2" char(100) DEFAULT NULL,
  PRIMARY KEY ("id")
) ;

DROP TABLE IF EXISTS "sbond_3";

CREATE TABLE "sbond_3" (
  "id" integer ,
  "drug" char(8) DEFAULT NULL,
  "atomid" char(100) DEFAULT NULL,
  "atomid_2" char(100) DEFAULT NULL,
  PRIMARY KEY ("id")
) ;

DROP TABLE IF EXISTS "sbond_7";

CREATE TABLE "sbond_7" (
  "id" integer ,
  "drug" char(9) DEFAULT NULL,
  "atomid" char(100) DEFAULT NULL,
  "atomid_2" char(100) DEFAULT NULL,
  PRIMARY KEY ("id")
) ;

