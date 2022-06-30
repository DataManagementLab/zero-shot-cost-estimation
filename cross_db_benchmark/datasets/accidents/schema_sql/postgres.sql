

DROP TABLE IF EXISTS "nesreca";

CREATE TABLE "nesreca" (
  "id_nesreca" char(6) NOT NULL,
  "klas_nesreca" char(1) NOT NULL,
  "upravna_enota" char(4) NOT NULL,
  "cas_nesreca" varchar(255) NOT NULL,
  "naselje_ali_izven" char(1) NOT NULL,
  "kategorija_cesta" char(1) DEFAULT NULL,
  "oznaka_cesta_ali_naselje" char(5) NOT NULL,
  "tekst_cesta_ali_naselje" varchar(25) NOT NULL,
  "oznaka_odsek_ali_ulica" char(5) NOT NULL,
  "tekst_odsek_ali_ulica" varchar(25) NOT NULL,
  "stacionazna_ali_hisna_st" varchar(9) DEFAULT NULL,
  "opis_prizorisce" char(1) NOT NULL,
  "vzrok_nesreca" char(2) NOT NULL,
  "tip_nesreca" char(2) NOT NULL,
  "vreme_nesreca" char(1) NOT NULL,
  "stanje_promet" char(1) NOT NULL,
  "stanje_vozisce" char(2) NOT NULL,
  "stanje_povrsina_vozisce" char(2) NOT NULL,
  "x" integer DEFAULT NULL,
  "y" integer DEFAULT NULL,
  "x_wgs84" double precision DEFAULT NULL,
  "y_wgs84" double precision DEFAULT NULL,
  PRIMARY KEY ("id_nesreca")
) ;

DROP TABLE IF EXISTS "oseba";

CREATE TABLE "oseba" (
  "id_nesreca" char(6) NOT NULL,
  "povzrocitelj_ali_udelezenec" char(1) DEFAULT NULL,
  "starost" integer DEFAULT NULL,
  "spol" char(1) NOT NULL,
  "upravna_enota" char(4) NOT NULL,
  "drzavljanstvo" char(3) NOT NULL,
  "poskodba" char(1) DEFAULT NULL,
  "vrsta_udelezenca" char(2) DEFAULT NULL,
  "varnostni_pas_ali_celada" char(1) DEFAULT NULL,
  "vozniski_staz_LL" integer DEFAULT NULL,
  "vozniski_staz_MM" integer DEFAULT NULL,
  "alkotest" decimal(3,2) DEFAULT NULL,
  "strokovni_pregled" decimal(3,2) DEFAULT NULL,
  "starost_d" char(1) DEFAULT NULL,
  "vozniski_staz_d" char(1) NOT NULL,
  "alkotest_d" char(1) NOT NULL,
  "strokovni_pregled_d" char(1) NOT NULL
) ;

DROP TABLE IF EXISTS "upravna_enota";

CREATE TABLE "upravna_enota" (
  "id_upravna_enota" char(4) NOT NULL,
  "ime_upravna_enota" varchar(255) NOT NULL,
  "st_prebivalcev" integer DEFAULT NULL,
  "povrsina" integer DEFAULT NULL,
  PRIMARY KEY ("id_upravna_enota")
) ;

