

DROP TABLE IF EXISTS "bod_schuze";

CREATE TABLE "bod_schuze" (
  "id_bod" integer ,
  "id_schuze" integer ,
  "id_tisk" varchar(255) DEFAULT NULL,
  "id_typ" varchar(255) DEFAULT NULL,
  "bod" integer ,
  "uplny_naz" text DEFAULT NULL,
  "uplny_kon" varchar(255) DEFAULT NULL,
  "poznamka" varchar(255) DEFAULT NULL,
  "id_bod_stav" integer ,
  "pozvanka" varchar(255) DEFAULT NULL,
  "rj" varchar(255) DEFAULT NULL,
  "pozn2" varchar(255) DEFAULT NULL,
  "druh_bodu" varchar(255) DEFAULT NULL,
  "id_sd" varchar(255) DEFAULT NULL,
  "zkratka" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "bod_stav";

CREATE TABLE "bod_stav" (
  "id_bod_stav" integer ,
  "popis" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("id_bod_stav")
) ;

DROP TABLE IF EXISTS "funkce";

CREATE TABLE "funkce" (
  "id_funkce" integer ,
  "id_organ" integer DEFAULT NULL,
  "id_typ_funkce" integer DEFAULT NULL,
  "nazev_funkce_cz" text DEFAULT NULL,
  "priorita" integer DEFAULT NULL,
  PRIMARY KEY ("id_funkce")
) ;

DROP TABLE IF EXISTS "hl_check";

CREATE TABLE "hl_check" (
  "id_hlasovani" integer DEFAULT NULL,
  "turn" integer DEFAULT NULL,
  "mode" integer DEFAULT NULL,
  "id_h2" varchar(255) DEFAULT NULL,
  "id_h3" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "hl_hlasovani";

CREATE TABLE "hl_hlasovani" (
  "id_hlasovani" integer ,
  "id_organ" integer DEFAULT NULL,
  "schuze" integer DEFAULT NULL,
  "cislo" integer DEFAULT NULL,
  "bod" integer DEFAULT NULL,
  "datum" varchar(255) DEFAULT NULL,
  "cas" time DEFAULT NULL,
  "pro" integer DEFAULT NULL,
  "proti" integer DEFAULT NULL,
  "zdrzel" integer DEFAULT NULL,
  "nehlasoval" integer DEFAULT NULL,
  "prihlaseno" integer DEFAULT NULL,
  "kvorum" integer DEFAULT NULL,
  "druh_hlasovani" varchar(255) DEFAULT NULL,
  "vysledek" varchar(255) DEFAULT NULL,
  "nazev_dlouhy" text DEFAULT NULL,
  "nazev_kratky" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("id_hlasovani")
) ;

DROP TABLE IF EXISTS "hl_poslanec";

CREATE TABLE "hl_poslanec" (
  "id_poslanec" integer ,
  "id_hlasovani" integer ,
  "vysledek" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "hl_vazby";

CREATE TABLE "hl_vazby" (
  "id_hlasovani" integer DEFAULT NULL,
  "turn" integer DEFAULT NULL,
  "typ" integer DEFAULT NULL
) ;

DROP TABLE IF EXISTS "hl_zposlanec";

CREATE TABLE "hl_zposlanec" (
  "id_hlasovani" integer DEFAULT NULL,
  "id_osoba" integer DEFAULT NULL,
  "mode" integer DEFAULT NULL
) ;

DROP TABLE IF EXISTS "omluvy";

CREATE TABLE "omluvy" (
  "id_organ" integer ,
  "id_poslanec" integer ,
  "den" varchar(255) ,
  "od" varchar(255) DEFAULT NULL,
  "do" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "organy";

CREATE TABLE "organy" (
  "id_organ" integer ,
  "organ_id_organ" integer DEFAULT NULL,
  "id_typ_organu" integer DEFAULT NULL,
  "zkratka" varchar(255) DEFAULT NULL,
  "nazev_organu_cz" text DEFAULT NULL,
  "nazev_organu_en" text DEFAULT NULL,
  "od_organ" varchar(255) DEFAULT NULL,
  "do_organ" varchar(255) DEFAULT NULL,
  "priorita" varchar(255) DEFAULT NULL,
  "cl_organ_base" integer DEFAULT NULL,
  PRIMARY KEY ("id_organ")
) ;

DROP TABLE IF EXISTS "osoby";

CREATE TABLE "osoby" (
  "id_osoba" integer ,
  "pred" varchar(255) DEFAULT NULL,
  "jmeno" varchar(255) DEFAULT NULL,
  "prijmeni" varchar(255) DEFAULT NULL,
  "za" varchar(255) DEFAULT NULL,
  "narozeni" varchar(255) DEFAULT NULL,
  "pohlavi" varchar(255) DEFAULT NULL,
  "zmena" varchar(255) DEFAULT NULL,
  "umrti" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("id_osoba")
) ;

DROP TABLE IF EXISTS "pkgps";

CREATE TABLE "pkgps" (
  "id_poslanec" integer DEFAULT NULL,
  "adresa" varchar(255) DEFAULT NULL,
  "sirka" float DEFAULT NULL,
  "delka" float DEFAULT NULL
) ;

DROP TABLE IF EXISTS "poslanec";

CREATE TABLE "poslanec" (
  "id_poslanec" integer ,
  "id_osoba" integer DEFAULT NULL,
  "id_kraj" integer DEFAULT NULL,
  "id_kandidatka" integer DEFAULT NULL,
  "id_obdobi" integer DEFAULT NULL,
  "web" varchar(255) DEFAULT NULL,
  "ulice" varchar(255) DEFAULT NULL,
  "obec" varchar(255) DEFAULT NULL,
  "psc" varchar(255) DEFAULT NULL,
  "email" varchar(255) DEFAULT NULL,
  "telefon" varchar(255) DEFAULT NULL,
  "fax" varchar(255) DEFAULT NULL,
  "psp_telefon" varchar(255) DEFAULT NULL,
  "facebook" varchar(255) DEFAULT NULL,
  "foto" integer DEFAULT NULL,
  PRIMARY KEY ("id_poslanec")
) ;

DROP TABLE IF EXISTS "schuze";

CREATE TABLE "schuze" (
  "id_schuze" integer ,
  "id_organ" integer DEFAULT NULL,
  "schuze" integer DEFAULT NULL,
  "od_schuze" varchar(255) DEFAULT NULL,
  "do_schuze" varchar(255) DEFAULT NULL,
  "aktualizace" varchar(255) DEFAULT NULL,
  PRIMARY KEY ("id_schuze")
) ;

DROP TABLE IF EXISTS "schuze_stav";

CREATE TABLE "schuze_stav" (
  "id_schuze" integer DEFAULT NULL,
  "stav" integer DEFAULT NULL,
  "typ" varchar(255) DEFAULT NULL,
  "text_dt" varchar(255) DEFAULT NULL,
  "text_st" varchar(255) DEFAULT NULL,
  "tm_line" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "typ_funkce";

CREATE TABLE "typ_funkce" (
  "id_typ_funkce" integer ,
  "id_typ_org" integer DEFAULT NULL,
  "typ_funkce_cz" varchar(255) DEFAULT NULL,
  "typ_funkce_en" varchar(255) DEFAULT NULL,
  "priorita" integer DEFAULT NULL,
  "typ_funkce_obecny" integer DEFAULT NULL,
  PRIMARY KEY ("id_typ_funkce")
) ;

DROP TABLE IF EXISTS "typ_organu";

CREATE TABLE "typ_organu" (
  "id_typ_org" integer ,
  "typ_id_typ_org" varchar(255) DEFAULT NULL,
  "nazev_typ_org_cz" varchar(255) DEFAULT NULL,
  "nazev_typ_org_en" varchar(255) DEFAULT NULL,
  "typ_org_obecny" varchar(255) DEFAULT NULL,
  "priorita" integer DEFAULT NULL,
  PRIMARY KEY ("id_typ_org")
) ;

DROP TABLE IF EXISTS "zarazeni";

CREATE TABLE "zarazeni" (
  "id_osoba" integer DEFAULT NULL,
  "id_of" integer DEFAULT NULL,
  "cl_funkce" integer DEFAULT NULL,
  "od_o" varchar(255) DEFAULT NULL,
  "do_o" varchar(255) DEFAULT NULL,
  "od_f" varchar(255) DEFAULT NULL,
  "do_f" varchar(255) DEFAULT NULL
) ;

DROP TABLE IF EXISTS "zmatecne";

CREATE TABLE "zmatecne" (
  "id_hlasovani" integer DEFAULT NULL
) ;

