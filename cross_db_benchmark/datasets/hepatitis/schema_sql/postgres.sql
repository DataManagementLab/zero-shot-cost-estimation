

DROP TABLE IF EXISTS "Bio";

CREATE TABLE "Bio" (
  "fibros" varchar(45) ,
  "activity" varchar(45) ,
  "b_id" integer ,
  PRIMARY KEY ("b_id")
) ;

DROP TABLE IF EXISTS "dispat";

CREATE TABLE "dispat" (
  "m_id" integer  DEFAULT 0,
  "sex" varchar(45) DEFAULT NULL,
  "age" varchar(45) DEFAULT NULL,
  "Type" varchar(45) DEFAULT NULL,
  PRIMARY KEY ("m_id")
) ;

DROP TABLE IF EXISTS "indis";

CREATE TABLE "indis" (
  "got" varchar(10) DEFAULT NULL,
  "gpt" varchar(10) DEFAULT NULL,
  "alb" varchar(45) DEFAULT NULL,
  "tbil" varchar(45) DEFAULT NULL,
  "dbil" varchar(45) DEFAULT NULL,
  "che" varchar(45) DEFAULT NULL,
  "ttt" varchar(45) DEFAULT NULL,
  "ztt" varchar(45) DEFAULT NULL,
  "tcho" varchar(45) DEFAULT NULL,
  "tp" varchar(45) DEFAULT NULL,
  "in_id" integer ,
  PRIMARY KEY ("in_id")
) ;

DROP TABLE IF EXISTS "inf";

CREATE TABLE "inf" (
  "dur" varchar(45) DEFAULT NULL,
  "a_id" integer  DEFAULT 0,
  PRIMARY KEY ("a_id")
) ;

DROP TABLE IF EXISTS "rel11";

CREATE TABLE "rel11" (
  "b_id" integer  DEFAULT 0,
  "m_id" integer  DEFAULT 0,
  PRIMARY KEY ("b_id","m_id")
) ;

DROP TABLE IF EXISTS "rel12";

CREATE TABLE "rel12" (
  "in_id" integer  DEFAULT 0,
  "m_id" integer  DEFAULT 0,
  PRIMARY KEY ("in_id","m_id")
) ;

DROP TABLE IF EXISTS "rel13";

CREATE TABLE "rel13" (
  "a_id" integer  DEFAULT 0,
  "m_id" integer  DEFAULT 0,
  PRIMARY KEY ("a_id","m_id")
) ;

