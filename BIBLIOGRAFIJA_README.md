# Bibliografske Reference - Uputstvo

## Sistem bibliografskih referenci

Svi dokumenti u vektorskoj bazi sada imaju bibliografske reference u formatu:
**Autor (godina). Naslov. Izdavač. strane**

## Fajlovi

1. **bibliografija.json** - Katalog sa bibliografskim podacima za sve dokumente
2. **add_txt_to_faiss.py** - Skripta za dodavanje tekstualnih dokumenata (koristi bibliografiju)
3. **add_pdf_to_faiss.py** - Skripta za dodavanje PDF dokumenata (koristi bibliografiju)
4. **update_bibliography.py** - Skripta za ažuriranje postojećih dokumenata u bazi
5. **app.py** - Aplikacija koja prikazuje bibliografske reference u rezultatima

## Koraci za kompletno postavljanje

### 1. Popunite bibliografiju

Uredite `bibliografija.json` i dodajte tačne bibliografske podatke za svaki dokument:

```json
{
  "naziv_fajla.txt": {
    "autor": "Ime Autora",
    "godina": "2020",
    "naslov": "Pun Naslov Knjige",
    "izdavac": "Naziv Izdavača",
    "strane": "1-350"
  }
}
```

**Napomena:** Već sam kreirao template sa osnovnim podacima izvučenim iz imena fajlova. Trebate samo da dopunite detalje (godina, izdavač, strane).

### 2. Ažurirajte postojeće dokumente (ako ih imate)

Ako već imate dokumente u bazi, pokrenite:

```powershell
python update_bibliography.py
```

Ova skripta će:
- Učitati bibliografiju iz `bibliografija.json`
- Ažurirati sve postojeće dokumente sa bibliografskim podacima
- Kreirati backup `metadata.pkl.backup` pre izmene

### 3. Dodajte nove tekstualne dokumente

Za dodavanje svih tekstualnih dokumenata odjednom:

```powershell
python add_txt_to_faiss.py --all
```

Za dodavanje jednog fajla:

```powershell
python add_txt_to_faiss.py --file docs/tekstualni_fajlovi/primer.txt
```

Za ponovno indeksiranje (ako je već u bazi):

```powershell
python add_txt_to_faiss.py --all --force
```

### 4. Dodajte PDF dokumente

```powershell
python add_pdf_to_faiss.py putanja/do/fajla.pdf --period novi_vek
```

Periodi: `novi_vek`, `rani_vek`, `srednji_vek`, `ostalo`

## Format referenci

### Primer kompletne reference:
```
Branko Petranović (1988). Istorija Jugoslavije, Knjiga II. Nolit. str. 1-450
```

### Primer parcijalne reference (bez izdavača):
```
Mavro Orbin (1601). Kraljevstvo Slovena
```

### Prikaz u aplikaciji

Kada korisnik postavi pitanje, rezultati će izgledati ovako:

```
1. [tekst iz dokumenta...]
   Izvor: Branko Petranović (1988). Istorija Jugoslavije, Knjiga II. Nolit. str. 1-450

2. [tekst iz dokumenta...]
   Izvor: Čedomir Popov. Od Versaja do Dancinga
```

## Trenutno stanje

### Tekst fajlovi sa osnovnim podacima:
- ✅ 29 tekstualnih fajlova u `docs/tekstualni_fajlovi/`
- ✅ Template kreiran u `bibliografija.json`
- ⚠️ Potrebno dopuniti: godina, izdavač, strane

### Najvažniji dokumenti koji čekaju indeksiranje:
1. Branko Petranović - Istorija Jugoslavije (3 knjige)
2. Mavro Orbin - Kraljevstvo Slovena
3. Veselin Čajkanović - Mit i Religija u Srba
4. Nikola Tesla - Autobiografija
5. I još 24 drugih...

## Sledeći koraci

1. **Sada:** Popunite bibliografiju sa detaljima
2. **Zatim:** Pokrenite `update_bibliography.py` ako imate postojeće dokumente
3. **Na kraju:** Pokrenite `add_txt_to_faiss.py --all` da indeksirate sve tekstualne dokumente

Da li želite da prvo popunimo bibliografiju ili da odmah pokrenemo indeksiranje sa trenutnim podacima?
