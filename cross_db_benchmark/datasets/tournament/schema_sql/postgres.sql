

DROP TABLE IF EXISTS "regular_season_compact_results";

CREATE TABLE "regular_season_compact_results" (
  "season" integer ,
  "daynum" integer ,
  "wteam" integer ,
  "wscore" integer DEFAULT NULL,
  "lteam" integer ,
  "lscore" integer DEFAULT NULL,
  "wloc" varchar(255) DEFAULT NULL,
  "numot" integer DEFAULT NULL,
  PRIMARY KEY ("season","daynum","wteam","lteam")
) ;

DROP TABLE IF EXISTS "regular_season_detailed_results";

CREATE TABLE "regular_season_detailed_results" (
  "season" integer ,
  "daynum" integer ,
  "wteam" integer ,
  "wscore" integer ,
  "lteam" integer ,
  "lscore" integer ,
  "wloc" varchar(255) DEFAULT NULL,
  "numot" integer DEFAULT NULL,
  "wfgm" integer DEFAULT NULL,
  "wfga" integer DEFAULT NULL,
  "wfgm3" integer DEFAULT NULL,
  "wfga3" integer DEFAULT NULL,
  "wftm" integer DEFAULT NULL,
  "wfta" integer DEFAULT NULL,
  "wor" integer DEFAULT NULL,
  "wdr" integer DEFAULT NULL,
  "wast" integer DEFAULT NULL,
  "wto" integer DEFAULT NULL,
  "wstl" integer DEFAULT NULL,
  "wblk" integer DEFAULT NULL,
  "wpf" integer DEFAULT NULL,
  "lfgm" integer DEFAULT NULL,
  "lfga" integer DEFAULT NULL,
  "lfgm3" integer DEFAULT NULL,
  "lfga3" integer DEFAULT NULL,
  "lftm" integer DEFAULT NULL,
  "lfta" integer DEFAULT NULL,
  "lor" integer DEFAULT NULL,
  "ldr" integer DEFAULT NULL,
  "last" integer DEFAULT NULL,
  "lto" integer DEFAULT NULL,
  "lstl" integer DEFAULT NULL,
  "lblk" integer DEFAULT NULL,
  "lpf" integer DEFAULT NULL,
  PRIMARY KEY ("season","daynum","wteam","lteam")
) ;

DROP TABLE IF EXISTS "seasons";

CREATE TABLE "seasons" (
  "season" integer ,
  "dayzero" varchar(255) DEFAULT NULL,
  "regionW" varchar(255) DEFAULT NULL,
  "regionX" varchar(255) DEFAULT NULL,
  "regionY" varchar(255) DEFAULT NULL,
  "regionZ" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("season")
) ;

DROP TABLE IF EXISTS "target";

CREATE TABLE "target" (
  "id" varchar(16) ,
  "season" integer DEFAULT NULL,
  "team_id1" integer DEFAULT NULL,
  "team_id2" integer DEFAULT NULL,
  "pred" float DEFAULT NULL,
  "team_id1_wins" integer DEFAULT NULL,
  "team_id2_wins" integer DEFAULT NULL,
  PRIMARY KEY ("id")
) ;

DROP TABLE IF EXISTS "teams";

CREATE TABLE "teams" (
  "team_id" integer ,
  "team_name" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("team_id")
) ;

DROP TABLE IF EXISTS "tourney_compact_results";

CREATE TABLE "tourney_compact_results" (
  "season" integer ,
  "daynum" integer ,
  "wteam" integer ,
  "wscore" integer DEFAULT NULL,
  "lteam" integer ,
  "lscore" integer DEFAULT NULL,
  "wloc" varchar(255) DEFAULT NULL,
  "numot" integer DEFAULT NULL,
  PRIMARY KEY ("season","daynum","wteam","lteam")
) ;

DROP TABLE IF EXISTS "tourney_detailed_results";

CREATE TABLE "tourney_detailed_results" (
  "season" integer ,
  "daynum" integer ,
  "wteam" integer ,
  "wscore" integer DEFAULT NULL,
  "lteam" integer ,
  "lscore" integer DEFAULT NULL,
  "wloc" varchar(255) DEFAULT NULL,
  "numot" integer DEFAULT NULL,
  "wfgm" integer DEFAULT NULL,
  "wfga" integer DEFAULT NULL,
  "wfgm3" integer DEFAULT NULL,
  "wfga3" integer DEFAULT NULL,
  "wftm" integer DEFAULT NULL,
  "wfta" integer DEFAULT NULL,
  "wor" integer DEFAULT NULL,
  "wdr" integer DEFAULT NULL,
  "wast" integer DEFAULT NULL,
  "wto" integer DEFAULT NULL,
  "wstl" integer DEFAULT NULL,
  "wblk" integer DEFAULT NULL,
  "wpf" integer DEFAULT NULL,
  "lfgm" integer DEFAULT NULL,
  "lfga" integer DEFAULT NULL,
  "lfgm3" integer DEFAULT NULL,
  "lfga3" integer DEFAULT NULL,
  "lftm" integer DEFAULT NULL,
  "lfta" integer DEFAULT NULL,
  "lor" integer DEFAULT NULL,
  "ldr" integer DEFAULT NULL,
  "last" integer DEFAULT NULL,
  "lto" integer DEFAULT NULL,
  "lstl" integer DEFAULT NULL,
  "lblk" integer DEFAULT NULL,
  "lpf" integer DEFAULT NULL,
  PRIMARY KEY ("season","daynum","wteam","lteam")
) ;

DROP TABLE IF EXISTS "tourney_seeds";

CREATE TABLE "tourney_seeds" (
  "season" integer ,
  "seed" varchar(6) ,
  "team" integer DEFAULT NULL,
  PRIMARY KEY ("season","seed")
) ;

DROP TABLE IF EXISTS "tourney_slots";

CREATE TABLE "tourney_slots" (
  "season" integer ,
  "slot" varchar(6) ,
  "strongseed" varchar(255) DEFAULT NULL,
  "weakseed" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("slot","season")
) ;

