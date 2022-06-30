

DROP TABLE IF EXISTS "allstarfull";

CREATE TABLE "allstarfull" (
  "playerID" varchar(11) ,
  "yearID" integer ,
  "gameNum" integer ,
  "gameID" varchar(12) DEFAULT NULL,
  "teamID" varchar(5) DEFAULT NULL,
  "lgID" varchar(2) DEFAULT NULL,
  "GP" integer DEFAULT NULL,
  "startingPos" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","yearID","gameNum")
) ;

DROP TABLE IF EXISTS "appearances";

CREATE TABLE "appearances" (
  "yearID" integer ,
  "teamID" varchar(5) ,
  "lgID" varchar(2) DEFAULT NULL,
  "playerID" varchar(11) ,
  "G_all" integer DEFAULT NULL,
  "G_batting" integer DEFAULT NULL,
  "G_defense" integer DEFAULT NULL,
  "G_p" integer DEFAULT NULL,
  "G_c" integer DEFAULT NULL,
  "G_1b" integer DEFAULT NULL,
  "G_2b" integer DEFAULT NULL,
  "G_3b" integer DEFAULT NULL,
  "G_ss" integer DEFAULT NULL,
  "G_lf" integer DEFAULT NULL,
  "G_cf" integer DEFAULT NULL,
  "G_rf" integer DEFAULT NULL,
  "G_of" integer DEFAULT NULL,
  "G_dh" integer DEFAULT NULL,
  "G_ph" integer DEFAULT NULL,
  "G_pr" integer DEFAULT NULL,
  PRIMARY KEY ("yearID","teamID","playerID")
) ;

DROP TABLE IF EXISTS "awardsmanagers";

CREATE TABLE "awardsmanagers" (
  "managerID" varchar(12) ,
  "awardID" varchar(27) ,
  "yearID" integer ,
  "lgID" varchar(4) ,
  "tie" varchar(1) DEFAULT NULL,
  "notes" varchar(100) DEFAULT NULL,
  PRIMARY KEY ("yearID","awardID","lgID","managerID")
) ;

DROP TABLE IF EXISTS "awardsplayers";

CREATE TABLE "awardsplayers" (
  "playerID" varchar(11) ,
  "awardID" varchar(37) ,
  "yearID" integer ,
  "lgID" varchar(4) ,
  "tie" varchar(1) DEFAULT NULL,
  "notes" varchar(100) DEFAULT NULL,
  PRIMARY KEY ("yearID","awardID","lgID","playerID")
) ;

DROP TABLE IF EXISTS "awardssharemanagers";

CREATE TABLE "awardssharemanagers" (
  "awardID" varchar(17) ,
  "yearID" integer ,
  "lgID" varchar(4) ,
  "managerID" varchar(12) ,
  "pointsWon" integer DEFAULT NULL,
  "pointsMax" integer DEFAULT NULL,
  "votesFirst" integer DEFAULT NULL,
  PRIMARY KEY ("awardID","yearID","lgID","managerID")
) ;

DROP TABLE IF EXISTS "awardsshareplayers";

CREATE TABLE "awardsshareplayers" (
  "awardID" varchar(20) ,
  "yearID" integer ,
  "lgID" varchar(4) ,
  "playerID" varchar(11) ,
  "pointsWon" integer DEFAULT NULL,
  "pointsMax" integer DEFAULT NULL,
  "votesFirst" integer DEFAULT NULL,
  PRIMARY KEY ("awardID","yearID","lgID","playerID")
) ;

DROP TABLE IF EXISTS "batting";

CREATE TABLE "batting" (
  "playerID" varchar(11) ,
  "yearID" integer ,
  "stint" integer ,
  "teamID" varchar(5) DEFAULT NULL,
  "lgID" varchar(2) DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "G_batting" integer DEFAULT NULL,
  "AB" integer DEFAULT NULL,
  "R" integer DEFAULT NULL,
  "H" integer DEFAULT NULL,
  "2B" integer DEFAULT NULL,
  "3B" integer DEFAULT NULL,
  "HR" integer DEFAULT NULL,
  "RBI" integer DEFAULT NULL,
  "SB" integer DEFAULT NULL,
  "CS" integer DEFAULT NULL,
  "BB" integer DEFAULT NULL,
  "SO" integer DEFAULT NULL,
  "IBB" integer DEFAULT NULL,
  "HBP" integer DEFAULT NULL,
  "SH" integer DEFAULT NULL,
  "SF" integer DEFAULT NULL,
  "GIDP" integer DEFAULT NULL,
  "G_old" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","yearID","stint")
) ;

DROP TABLE IF EXISTS "battingpost";

CREATE TABLE "battingpost" (
  "yearID" integer ,
  "round" varchar(7) ,
  "playerID" varchar(11) ,
  "teamID" varchar(3) DEFAULT NULL,
  "lgID" varchar(2) DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "AB" integer DEFAULT NULL,
  "R" integer DEFAULT NULL,
  "H" integer DEFAULT NULL,
  "2B" integer DEFAULT NULL,
  "3B" integer DEFAULT NULL,
  "HR" integer DEFAULT NULL,
  "RBI" integer DEFAULT NULL,
  "SB" integer DEFAULT NULL,
  "CS" integer DEFAULT NULL,
  "BB" integer DEFAULT NULL,
  "SO" integer DEFAULT NULL,
  "IBB" integer DEFAULT NULL,
  "HBP" integer DEFAULT NULL,
  "SH" integer DEFAULT NULL,
  "SF" integer DEFAULT NULL,
  "GIDP" integer DEFAULT NULL,
  PRIMARY KEY ("yearID","round","playerID")
) ;

DROP TABLE IF EXISTS "els_teamnames";

CREATE TABLE "els_teamnames" (
  "id" integer ,
  "lgid" varchar(2) ,
  "teamid" varchar(5) ,
  "franchid" varchar(5) DEFAULT NULL,
  "name" varchar(50) DEFAULT NULL,
  "park" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "fielding";

CREATE TABLE "fielding" (
  "playerID" varchar(11) ,
  "yearID" integer ,
  "stint" integer ,
  "teamID" varchar(3) DEFAULT NULL,
  "lgID" varchar(2) DEFAULT NULL,
  "POS" varchar(4) ,
  "G" integer DEFAULT NULL,
  "GS" integer DEFAULT NULL,
  "InnOuts" integer DEFAULT NULL,
  "PO" integer DEFAULT NULL,
  "A" integer DEFAULT NULL,
  "E" integer DEFAULT NULL,
  "DP" integer DEFAULT NULL,
  "PB" integer DEFAULT NULL,
  "WP" integer DEFAULT NULL,
  "SB" integer DEFAULT NULL,
  "CS" integer DEFAULT NULL,
  "ZR" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","yearID","stint","POS")
) ;

DROP TABLE IF EXISTS "fieldingof";

CREATE TABLE "fieldingof" (
  "playerID" varchar(11) ,
  "yearID" integer ,
  "stint" integer ,
  "Glf" integer DEFAULT NULL,
  "Gcf" integer DEFAULT NULL,
  "Grf" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","yearID","stint")
) ;

DROP TABLE IF EXISTS "fieldingpost";

CREATE TABLE "fieldingpost" (
  "playerID" varchar(11) ,
  "yearID" integer ,
  "teamID" varchar(3) DEFAULT NULL,
  "lgID" varchar(2) DEFAULT NULL,
  "round" varchar(7) ,
  "POS" varchar(4) ,
  "G" integer DEFAULT NULL,
  "GS" integer DEFAULT NULL,
  "InnOuts" integer DEFAULT NULL,
  "PO" integer DEFAULT NULL,
  "A" integer DEFAULT NULL,
  "E" integer DEFAULT NULL,
  "DP" integer DEFAULT NULL,
  "TP" integer DEFAULT NULL,
  "PB" integer DEFAULT NULL,
  "SB" integer DEFAULT NULL,
  "CS" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","yearID","round","POS")
) ;

DROP TABLE IF EXISTS "halloffame";

CREATE TABLE "halloffame" (
  "hofID" varchar(12) ,
  "yearID" integer ,
  "votedBy" varchar(64) DEFAULT NULL,
  "ballots" integer DEFAULT NULL,
  "needed" integer DEFAULT NULL,
  "votes" integer DEFAULT NULL,
  "inducted" varchar(1) DEFAULT NULL,
  "category" varchar(20) DEFAULT NULL,
  PRIMARY KEY ("hofID","yearID")
) ;

DROP TABLE IF EXISTS "managers";

CREATE TABLE "managers" (
  "managerID" varchar(12) DEFAULT NULL,
  "yearID" integer ,
  "teamID" varchar(5) ,
  "lgID" varchar(2) DEFAULT NULL,
  "inseason" integer ,
  "G" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "rank" integer DEFAULT NULL,
  "plyrMgr" varchar(1) DEFAULT NULL,
  PRIMARY KEY ("yearID","teamID","inseason")
) ;

DROP TABLE IF EXISTS "managershalf";

CREATE TABLE "managershalf" (
  "managerID" varchar(12) ,
  "yearID" integer ,
  "teamID" varchar(5) ,
  "lgID" varchar(2) DEFAULT NULL,
  "inseason" integer DEFAULT NULL,
  "half" integer ,
  "G" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "rank" integer DEFAULT NULL,
  PRIMARY KEY ("yearID","teamID","managerID","half")
) ;

DROP TABLE IF EXISTS "pitching";

CREATE TABLE "pitching" (
  "playerID" varchar(11) ,
  "yearID" integer ,
  "stint" integer ,
  "teamID" varchar(3) DEFAULT NULL,
  "lgID" varchar(2) DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "GS" integer DEFAULT NULL,
  "CG" integer DEFAULT NULL,
  "SHO" integer DEFAULT NULL,
  "SV" integer DEFAULT NULL,
  "IPouts" integer DEFAULT NULL,
  "H" integer DEFAULT NULL,
  "ER" integer DEFAULT NULL,
  "HR" integer DEFAULT NULL,
  "BB" integer DEFAULT NULL,
  "SO" integer DEFAULT NULL,
  "BAOpp" integer DEFAULT NULL,
  "ERA" integer DEFAULT NULL,
  "IBB" integer DEFAULT NULL,
  "WP" integer DEFAULT NULL,
  "HBP" integer DEFAULT NULL,
  "BK" integer DEFAULT NULL,
  "BFP" integer DEFAULT NULL,
  "GF" integer DEFAULT NULL,
  "R" integer DEFAULT NULL,
  "SH" integer DEFAULT NULL,
  "SF" integer DEFAULT NULL,
  "GIDP" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","yearID","stint")
) ;

DROP TABLE IF EXISTS "pitchingpost";

CREATE TABLE "pitchingpost" (
  "playerID" varchar(11) ,
  "yearID" integer ,
  "round" varchar(7) ,
  "teamID" varchar(3) DEFAULT NULL,
  "lgID" varchar(2) DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "GS" integer DEFAULT NULL,
  "CG" integer DEFAULT NULL,
  "SHO" integer DEFAULT NULL,
  "SV" integer DEFAULT NULL,
  "IPouts" integer DEFAULT NULL,
  "H" integer DEFAULT NULL,
  "ER" integer DEFAULT NULL,
  "HR" integer DEFAULT NULL,
  "BB" integer DEFAULT NULL,
  "SO" integer DEFAULT NULL,
  "BAOpp" integer DEFAULT NULL,
  "ERA" integer DEFAULT NULL,
  "IBB" integer DEFAULT NULL,
  "WP" integer DEFAULT NULL,
  "HBP" integer DEFAULT NULL,
  "BK" integer DEFAULT NULL,
  "BFP" integer DEFAULT NULL,
  "GF" integer DEFAULT NULL,
  "R" integer DEFAULT NULL,
  "SH" integer DEFAULT NULL,
  "SF" integer DEFAULT NULL,
  "GIDP" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","yearID","round")
) ;

DROP TABLE IF EXISTS "players";

CREATE TABLE "players" (
  "lahmanID" integer ,
  "playerID" varchar(11) DEFAULT NULL,
  "managerID" varchar(10) DEFAULT NULL,
  "hofID" varchar(12) DEFAULT NULL,
  "birthYear" integer DEFAULT NULL,
  "birthMonth" integer DEFAULT NULL,
  "birthDay" integer DEFAULT NULL,
  "birthCountry" varchar(50) DEFAULT NULL,
  "birthState" varchar(2) DEFAULT NULL,
  "birthCity" varchar(50) DEFAULT NULL,
  "deathYear" integer DEFAULT NULL,
  "deathMonth" integer DEFAULT NULL,
  "deathDay" integer DEFAULT NULL,
  "deathCountry" varchar(50) DEFAULT NULL,
  "deathState" varchar(2) DEFAULT NULL,
  "deathCity" varchar(50) DEFAULT NULL,
  "nameFirst" varchar(50) DEFAULT NULL,
  "nameLast" varchar(50) DEFAULT NULL,
  "nameNote" varchar(255) DEFAULT NULL,
  "nameGiven" varchar(255) DEFAULT NULL,
  "nameNick" varchar(255) DEFAULT NULL,
  "weight" integer DEFAULT NULL,
  "height" integer DEFAULT NULL,
  "bats" varchar(1) DEFAULT NULL,
  "throws" varchar(1) DEFAULT NULL,
  "debut" varchar(10) DEFAULT NULL,
  "finalGame" varchar(10) DEFAULT NULL,
  "college" varchar(50) DEFAULT NULL,
  "lahman40ID" varchar(9) DEFAULT NULL,
  "lahman45ID" varchar(9) DEFAULT NULL,
  "retroID" varchar(9) DEFAULT NULL,
  "holtzID" varchar(9) DEFAULT NULL,
  "bbrefID" varchar(9) DEFAULT NULL,
  PRIMARY KEY ("lahmanID")
) ;

DROP TABLE IF EXISTS "salaries";

CREATE TABLE "salaries" (
  "yearID" integer ,
  "teamID" varchar(5) ,
  "lgID" varchar(4) ,
  "playerID" varchar(11) ,
  "salary" integer DEFAULT NULL,
  PRIMARY KEY ("yearID","teamID","lgID","playerID")
) ;

DROP TABLE IF EXISTS "schools";

CREATE TABLE "schools" (
  "schoolID" varchar(13) ,
  "schoolName" varchar(255) DEFAULT NULL,
  "schoolCity" varchar(55) DEFAULT NULL,
  "schoolState" varchar(55) DEFAULT NULL,
  "schoolNick" varchar(55) DEFAULT NULL,
  PRIMARY KEY ("schoolID")
) ;

DROP TABLE IF EXISTS "schoolsplayers";

CREATE TABLE "schoolsplayers" (
  "playerID" varchar(11) ,
  "schoolID" varchar(13) ,
  "yearMin" integer DEFAULT NULL,
  "yearMax" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","schoolID")
) ;

DROP TABLE IF EXISTS "seriespost";

CREATE TABLE "seriespost" (
  "yearID" integer ,
  "round" varchar(7) ,
  "teamIDwinner" varchar(5) DEFAULT NULL,
  "lgIDwinner" varchar(2) DEFAULT NULL,
  "teamIDloser" varchar(3) DEFAULT NULL,
  "lgIDloser" varchar(2) DEFAULT NULL,
  "wins" integer DEFAULT NULL,
  "losses" integer DEFAULT NULL,
  "ties" integer DEFAULT NULL,
  PRIMARY KEY ("yearID","round")
) ;

DROP TABLE IF EXISTS "teams";

CREATE TABLE "teams" (
  "yearID" integer ,
  "lgID" varchar(4) ,
  "teamID" varchar(5) ,
  "franchID" varchar(3) DEFAULT NULL,
  "divID" varchar(1) DEFAULT NULL,
  "Rank" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "Ghome" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  "DivWin" varchar(1) DEFAULT NULL,
  "WCWin" varchar(1) DEFAULT NULL,
  "LgWin" varchar(1) DEFAULT NULL,
  "WSWin" varchar(1) DEFAULT NULL,
  "R" integer DEFAULT NULL,
  "AB" integer DEFAULT NULL,
  "H" integer DEFAULT NULL,
  "2B" integer DEFAULT NULL,
  "3B" integer DEFAULT NULL,
  "HR" integer DEFAULT NULL,
  "BB" integer DEFAULT NULL,
  "SO" integer DEFAULT NULL,
  "SB" integer DEFAULT NULL,
  "CS" integer DEFAULT NULL,
  "HBP" integer DEFAULT NULL,
  "SF" integer DEFAULT NULL,
  "RA" integer DEFAULT NULL,
  "ER" integer DEFAULT NULL,
  "ERA" integer DEFAULT NULL,
  "CG" integer DEFAULT NULL,
  "SHO" integer DEFAULT NULL,
  "SV" integer DEFAULT NULL,
  "IPouts" integer DEFAULT NULL,
  "HA" integer DEFAULT NULL,
  "HRA" integer DEFAULT NULL,
  "BBA" integer DEFAULT NULL,
  "SOA" integer DEFAULT NULL,
  "E" integer DEFAULT NULL,
  "DP" integer DEFAULT NULL,
  "FP" integer DEFAULT NULL,
  "name" varchar(50) DEFAULT NULL,
  "park" varchar(255) DEFAULT NULL,
  "attendance" integer DEFAULT NULL,
  "BPF" integer DEFAULT NULL,
  "PPF" integer DEFAULT NULL,
  "teamIDBR" varchar(3) DEFAULT NULL,
  "teamIDlahman45" varchar(3) DEFAULT NULL,
  "teamIDretro" varchar(3) DEFAULT NULL,
  PRIMARY KEY ("yearID","lgID","teamID")
) ;

DROP TABLE IF EXISTS "teamsfranchises";

CREATE TABLE "teamsfranchises" (
  "franchID" varchar(5) ,
  "franchName" varchar(50) DEFAULT NULL,
  "active" varchar(2) DEFAULT NULL,
  "NAassoc" varchar(3) DEFAULT NULL,
  PRIMARY KEY ("franchID")
) ;

DROP TABLE IF EXISTS "teamshalf";

CREATE TABLE "teamshalf" (
  "yearID" integer ,
  "lgID" varchar(4) ,
  "teamID" varchar(5) ,
  "Half" varchar(2) ,
  "divID" varchar(1) DEFAULT NULL,
  "DivWin" varchar(1) DEFAULT NULL,
  "Rank" integer DEFAULT NULL,
  "G" integer DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  PRIMARY KEY ("yearID","teamID","lgID","Half")
) ;

