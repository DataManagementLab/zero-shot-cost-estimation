

DROP TABLE IF EXISTS "actors";

CREATE TABLE "actors" (
  "actorid" integer ,
  "a_gender" varchar(255) ,
  "a_quality" integer ,
  PRIMARY KEY ("actorid")
) ;

DROP TABLE IF EXISTS "directors";

CREATE TABLE "directors" (
  "directorid" integer ,
  "d_quality" integer ,
  "avg_revenue" integer ,
  PRIMARY KEY ("directorid")
) ;

DROP TABLE IF EXISTS "movies";

CREATE TABLE "movies" (
  "movieid" integer  DEFAULT 0,
  "year" integer ,
  "isEnglish" varchar(255) ,
  "country" varchar(50) ,
  "runningtime" integer ,
  PRIMARY KEY ("movieid")
) ;

DROP TABLE IF EXISTS "movies2actors";

CREATE TABLE "movies2actors" (
  "movieid" integer ,
  "actorid" integer ,
  "cast_num" integer ,
  PRIMARY KEY ("movieid","actorid")
) ;

DROP TABLE IF EXISTS "movies2directors";

CREATE TABLE "movies2directors" (
  "movieid" integer ,
  "directorid" integer ,
  "genre" varchar(15) ,
  PRIMARY KEY ("movieid","directorid")
) ;

DROP TABLE IF EXISTS "u2base";

CREATE TABLE "u2base" (
  "userid" integer  DEFAULT 0,
  "movieid" integer ,
  "rating" varchar(45) ,
  PRIMARY KEY ("userid","movieid")
) ;

DROP TABLE IF EXISTS "users";

CREATE TABLE "users" (
  "userid" integer  DEFAULT 0,
  "age" varchar(5) ,
  "u_gender" varchar(5) ,
  "occupation" varchar(45) ,
  PRIMARY KEY ("userid")
) ;

