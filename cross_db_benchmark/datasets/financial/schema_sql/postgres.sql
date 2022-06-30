

DROP TABLE IF EXISTS "account";

CREATE TABLE "account" (
  "account_id" integer  DEFAULT 0,
  "district_id" integer  DEFAULT 0,
  "frequency" varchar(18) ,
  "date" varchar(255) ,
  PRIMARY KEY ("account_id")
) ;

DROP TABLE IF EXISTS "card";

CREATE TABLE "card" (
  "card_id" integer  DEFAULT 0,
  "disp_id" integer ,
  "type" varchar(7) ,
  "issued" varchar(255) ,
  PRIMARY KEY ("card_id")
) ;

DROP TABLE IF EXISTS "client";

CREATE TABLE "client" (
  "client_id" integer ,
  "gender" varchar(1) ,
  "birth_date" varchar(255) ,
  "district_id" integer ,
  PRIMARY KEY ("client_id")
) ;

DROP TABLE IF EXISTS "disp";

CREATE TABLE "disp" (
  "disp_id" integer ,
  "client_id" integer ,
  "account_id" integer ,
  "type" varchar(9) ,
  PRIMARY KEY ("disp_id")
) ;

DROP TABLE IF EXISTS "district";

CREATE TABLE "district" (
  "district_id" integer  DEFAULT 0,
  "A2" varchar(19) ,
  "A3" varchar(15) ,
  "A4" integer ,
  "A5" integer ,
  "A6" integer ,
  "A7" integer ,
  "A8" integer ,
  "A9" integer ,
  "A10" decimal(4,1) ,
  "A11" integer ,
  "A12" decimal(4,1) DEFAULT NULL,
  "A13" decimal(3,2) ,
  "A14" integer ,
  "A15" integer DEFAULT NULL,
  "A16" integer ,
  PRIMARY KEY ("district_id")
) ;

DROP TABLE IF EXISTS "loan";

CREATE TABLE "loan" (
  "loan_id" integer  DEFAULT 0,
  "account_id" integer ,
  "date" varchar(255) ,
  "amount" integer ,
  "duration" integer ,
  "payments" decimal(6,2) ,
  "status" varchar(1) ,
  PRIMARY KEY ("loan_id")
) ;

DROP TABLE IF EXISTS "order";

CREATE TABLE "order" (
  "order_id" integer  DEFAULT 0,
  "account_id" integer ,
  "bank_to" varchar(2) ,
  "account_to" integer ,
  "amount" decimal(6,1) ,
  "k_symbol" varchar(8) ,
  PRIMARY KEY ("order_id")
) ;

DROP TABLE IF EXISTS "trans";

CREATE TABLE "trans" (
  "trans_id" integer  DEFAULT 0,
  "account_id" integer  DEFAULT 0,
  "date" varchar(255) ,
  "type" varchar(6) ,
  "operation" varchar(14) DEFAULT NULL,
  "amount" integer ,
  "balance" integer ,
  "k_symbol" varchar(11) DEFAULT NULL,
  "bank" varchar(2) DEFAULT NULL,
  "account" integer DEFAULT NULL,
  PRIMARY KEY ("trans_id")
) ;

