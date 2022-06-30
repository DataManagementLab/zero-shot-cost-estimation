

DROP TABLE IF EXISTS "awards_coaches";

CREATE TABLE "awards_coaches" (
  "id" integer ,
  "year" integer DEFAULT NULL,
  "coachID" varchar(13) DEFAULT NULL,
  "award" varchar(255) DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "note" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("id")
);

DROP TABLE IF EXISTS "awards_players";

CREATE TABLE "awards_players" (
  "playerID" varchar(13) ,
  "award" varchar(39) ,
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "note" varchar(255) DEFAULT NULL,
  "pos" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("playerID","year","award")
);

DROP TABLE IF EXISTS "coaches";

CREATE TABLE "coaches" (
  "coachID" varchar(14) ,
  "year" integer ,
  "tmID" varchar(7) ,
  "lgID" varchar(255) DEFAULT NULL,
  "stint" integer ,
  "won" integer DEFAULT NULL,
  "lost" integer DEFAULT NULL,
  "post_wins" integer DEFAULT NULL,
  "post_losses" integer DEFAULT NULL,
  PRIMARY KEY ("coachID","year","tmID","stint")
);

DROP TABLE IF EXISTS "draft";

CREATE TABLE "draft" (
  "id" integer  DEFAULT 0,
  "draftYear" integer DEFAULT NULL,
  "draftRound" integer DEFAULT NULL,
  "draftSelection" integer DEFAULT NULL,
  "draftOverall" integer DEFAULT NULL,
  "tmID" varchar(7) DEFAULT NULL,
  "firstName" varchar(255) DEFAULT NULL,
  "lastName" varchar(255) DEFAULT NULL,
  "suffixName" varchar(255) DEFAULT NULL,
  "playerID" varchar(255) DEFAULT NULL,
  "draftFrom" varchar(255) DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("id")
);

DROP TABLE IF EXISTS "player_allstar";

CREATE TABLE "player_allstar" (
  "playerID" varchar(13) ,
  "last_name" varchar(255) DEFAULT NULL,
  "first_name" varchar(255) DEFAULT NULL,
  "season_id" integer ,
  "conference" varchar(255) DEFAULT NULL,
  "league_id" varchar(255) DEFAULT NULL,
  "games_played" integer DEFAULT NULL,
  "minutes" integer DEFAULT NULL,
  "points" integer DEFAULT NULL,
  "o_rebounds" integer DEFAULT NULL,
  "d_rebounds" integer DEFAULT NULL,
  "rebounds" integer DEFAULT NULL,
  "assists" integer DEFAULT NULL,
  "steals" integer DEFAULT NULL,
  "blocks" integer DEFAULT NULL,
  "turnovers" integer DEFAULT NULL,
  "personal_fouls" integer DEFAULT NULL,
  "fg_attempted" integer DEFAULT NULL,
  "fg_made" integer DEFAULT NULL,
  "ft_attempted" integer DEFAULT NULL,
  "ft_made" integer DEFAULT NULL,
  "three_attempted" integer DEFAULT NULL,
  "three_made" integer DEFAULT NULL,
  PRIMARY KEY ("playerID","season_id")
);

DROP TABLE IF EXISTS "players";

CREATE TABLE "players" (
  "playerID" varchar(13) ,
  "useFirst" varchar(255) DEFAULT NULL,
  "firstName" varchar(255) DEFAULT NULL,
  "middleName" varchar(255) DEFAULT NULL,
  "lastName" varchar(255) DEFAULT NULL,
  "nameGiven" varchar(255) DEFAULT NULL,
  "fullGivenName" varchar(255) DEFAULT NULL,
  "nameSuffix" varchar(255) DEFAULT NULL,
  "nameNick" varchar(255) DEFAULT NULL,
  "pos" varchar(255) DEFAULT NULL,
  "firstseason" integer DEFAULT NULL,
  "lastseason" integer DEFAULT NULL,
  "height" float DEFAULT NULL,
  "weight" integer DEFAULT NULL,
  "college" varchar(255) DEFAULT NULL,
  "collegeOther" varchar(255) DEFAULT NULL,
  "birthDate" varchar(255) DEFAULT NULL,
  "birthCity" varchar(255) DEFAULT NULL,
  "birthState" varchar(255) DEFAULT NULL,
  "birthCountry" varchar(255) DEFAULT NULL,
  "highSchool" varchar(255) DEFAULT NULL,
  "hsCity" varchar(255) DEFAULT NULL,
  "hsState" varchar(255) DEFAULT NULL,
  "hsCountry" varchar(255) DEFAULT NULL,
  "deathDate" varchar(255) DEFAULT NULL,
  "race" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("playerID")
);

DROP TABLE IF EXISTS "players_teams";

CREATE TABLE "players_teams" (
  "id" integer ,
  "playerID" varchar(13) ,
  "year" integer DEFAULT NULL,
  "stint" integer DEFAULT NULL,
  "tmID" varchar(7) DEFAULT NULL,
  "lgID" varchar(255) DEFAULT NULL,
  "GP" integer DEFAULT NULL,
  "GS" integer DEFAULT NULL,
  "minutes" integer DEFAULT NULL,
  "points" integer DEFAULT NULL,
  "oRebounds" integer DEFAULT NULL,
  "dRebounds" integer DEFAULT NULL,
  "rebounds" integer DEFAULT NULL,
  "assists" integer DEFAULT NULL,
  "steals" integer DEFAULT NULL,
  "blocks" integer DEFAULT NULL,
  "turnovers" integer DEFAULT NULL,
  "PF" integer DEFAULT NULL,
  "fgAttempted" integer DEFAULT NULL,
  "fgMade" integer DEFAULT NULL,
  "ftAttempted" integer DEFAULT NULL,
  "ftMade" integer DEFAULT NULL,
  "threeAttempted" integer DEFAULT NULL,
  "threeMade" integer DEFAULT NULL,
  "PostGP" integer DEFAULT NULL,
  "PostGS" integer DEFAULT NULL,
  "PostMinutes" integer DEFAULT NULL,
  "PostPoints" integer DEFAULT NULL,
  "PostoRebounds" integer DEFAULT NULL,
  "PostdRebounds" integer DEFAULT NULL,
  "PostRebounds" integer DEFAULT NULL,
  "PostAssists" integer DEFAULT NULL,
  "PostSteals" integer DEFAULT NULL,
  "PostBlocks" integer DEFAULT NULL,
  "PostTurnovers" integer DEFAULT NULL,
  "PostPF" integer DEFAULT NULL,
  "PostfgAttempted" integer DEFAULT NULL,
  "PostfgMade" integer DEFAULT NULL,
  "PostftAttempted" integer DEFAULT NULL,
  "PostftMade" integer DEFAULT NULL,
  "PostthreeAttempted" integer DEFAULT NULL,
  "PostthreeMade" integer DEFAULT NULL,
  "note" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("id")
);

DROP TABLE IF EXISTS "series_post";

CREATE TABLE "series_post" (
  "id" integer ,
  "year" integer DEFAULT NULL,
  "round" varchar(255) DEFAULT NULL,
  "series" varchar(255) DEFAULT NULL,
  "tmIDWinner" varchar(7) DEFAULT NULL,
  "lgIDWinner" varchar(255) DEFAULT NULL,
  "tmIDLoser" varchar(255) DEFAULT NULL,
  "lgIDLoser" varchar(255) DEFAULT NULL,
  "W" integer DEFAULT NULL,
  "L" integer DEFAULT NULL,
  PRIMARY KEY ("id")
);

DROP TABLE IF EXISTS "teams";

CREATE TABLE "teams" (
  "year" integer ,
  "lgID" varchar(255) DEFAULT NULL,
  "tmID" varchar(7) ,
  "franchID" varchar(255) DEFAULT NULL,
  "confID" varchar(255) DEFAULT NULL,
  "divID" varchar(255) DEFAULT NULL,
  "rank" integer DEFAULT NULL,
  "confRank" integer DEFAULT NULL,
  "playoff" varchar(255) DEFAULT NULL,
  "name" varchar(255) DEFAULT NULL,
  "o_fgm" integer DEFAULT NULL,
  "o_fga" integer DEFAULT NULL,
  "o_ftm" integer DEFAULT NULL,
  "o_fta" integer DEFAULT NULL,
  "o_3pm" integer DEFAULT NULL,
  "o_3pa" integer DEFAULT NULL,
  "o_oreb" integer DEFAULT NULL,
  "o_dreb" integer DEFAULT NULL,
  "o_reb" integer DEFAULT NULL,
  "o_asts" integer DEFAULT NULL,
  "o_pf" integer DEFAULT NULL,
  "o_stl" integer DEFAULT NULL,
  "o_to" integer DEFAULT NULL,
  "o_blk" integer DEFAULT NULL,
  "o_pts" integer DEFAULT NULL,
  "d_fgm" integer DEFAULT NULL,
  "d_fga" integer DEFAULT NULL,
  "d_ftm" integer DEFAULT NULL,
  "d_fta" integer DEFAULT NULL,
  "d_3pm" integer DEFAULT NULL,
  "d_3pa" integer DEFAULT NULL,
  "d_oreb" integer DEFAULT NULL,
  "d_dreb" integer DEFAULT NULL,
  "d_reb" integer DEFAULT NULL,
  "d_asts" integer DEFAULT NULL,
  "d_pf" integer DEFAULT NULL,
  "d_stl" integer DEFAULT NULL,
  "d_to" integer DEFAULT NULL,
  "d_blk" integer DEFAULT NULL,
  "d_pts" integer DEFAULT NULL,
  "o_tmRebound" integer DEFAULT NULL,
  "d_tmRebound" integer DEFAULT NULL,
  "homeWon" integer DEFAULT NULL,
  "homeLost" integer DEFAULT NULL,
  "awayWon" integer DEFAULT NULL,
  "awayLost" integer DEFAULT NULL,
  "neutWon" integer DEFAULT NULL,
  "neutLoss" integer DEFAULT NULL,
  "confWon" integer DEFAULT NULL,
  "confLoss" integer DEFAULT NULL,
  "divWon" integer DEFAULT NULL,
  "divLoss" integer DEFAULT NULL,
  "pace" integer DEFAULT NULL,
  "won" integer DEFAULT NULL,
  "lost" integer DEFAULT NULL,
  "games" integer DEFAULT NULL,
  "min" integer DEFAULT NULL,
  "arena" varchar(255) DEFAULT NULL,
  "attendance" integer DEFAULT NULL,
  "bbtmID" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("year","tmID")
);

