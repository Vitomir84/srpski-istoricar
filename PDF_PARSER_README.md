# PDF OCR Parser - Dokumentacija

## Pregled

`pdf_parser.py` je script koji automatski parsuje PDF dokumente iz organizovanih foldera i dodaje ih u Qdrant vektorsku bazu sa očuvanom vremenskom periodizacijom.

## Funkcionalnosti

✅ **Automatsko skeniranje foldera** po istorijskim periodima  
✅ **Direktna ekstrakcija teksta** iz PDF-ova  
✅ **OCR podrška** za skenirane dokumente (koristi Tesseract)  
✅ **Pametno deljenje teksta** na chunke sa preklapanjem  
✅ **Metadata preservation** - čuva informacije o periodu i izvoru  
✅ **Rate limiting** - zaštita od prekoračenja API limita  
✅ **Progress tracking** - detaljan izveštaj o napretku  

## Struktura foldera

```
docs/
├── novi_vek/          # Novi vek (1804-1918)
├── rani_vek/          # Rani srednji vek (6-11 vek)
├── srednji_vek/       # Srednji vek (12-15 vek)
└── ostalo/            # Ostali dokumenti
```

## Preduslov - Tesseract OCR

Za OCR funkcionalnost potreban je **Tesseract OCR** sa srpskim jezičkim paketom.

### Windows Instalacija:

1. **Preuzmi Tesseract:**
   - Link: https://github.com/UB-Mannheim/tesseract/wiki
   - Preuzmi najnoviju verziju (tesseract-ocr-w64-setup-*.exe)

2. **Instaliraj:**
   - Pokreni installer
   - **VAŽNO:** Tokom instalacije označi "Additional Language Data"
   - Obavezno izaberi **Serbian** (srp) jezik

3. **Dodaj u PATH:**
   ```powershell
   # Dodaj Tesseract u sistemski PATH (obično):
   C:\Program Files\Tesseract-OCR
   ```

4. **Verifikuj instalaciju:**
   ```powershell
   tesseract --version
   tesseract --list-langs  # Treba da vidiš 'srp' u listi
   ```

### Alternative - bez OCR-a:

Ako ne želiš OCR, script će i dalje raditi, ali samo za PDF-ove koji već imaju tekstualni layer.

## Instalacija dependencies

```powershell
# Aktiviraj Poetry environment
poetry shell

# Instaliraj nove biblioteke
poetry install
```

## Korišćenje

### Osnovno pokretanje:

```powershell
poetry run python pdf_parser.py
```

Script će automatski:
1. Skenirati sve foldere (novi_vek, rani_vek, srednji_vek, ostalo)
2. Procesovati svaki PDF fajl
3. Ekstraktovati tekst (sa OCR ako je potrebno)
4. Podeliti tekst na chunke (~1000 karaktera)
5. Kreirati embeddings preko OpenAI API
6. Dodati u Qdrant bazu sa metadata

### Što čuva u bazi:

Svaki chunk dobija sledeće metadata:
- `period`: Folder/period kategorija (npr. "srednji_vek")
- `period_name`: Ljudski naziv (npr. "Средњи век")
- `source_file`: Ime PDF fajla
- `source_path`: Relativna putanja
- `chunk_index`: Redni broj chunka
- `total_chunks`: Ukupan broj chunkova iz dokumenta
- `text`: Sam tekst chunka

## Primer output-a

```
🇷🇸 SRPSKI ISTORIČAR - PDF OCR Parser
==================================================================
✓ Kolekcija 'serbian_history' postoji

==================================================================
📁 Period: Средњи век
📄 Pronađeno 6 PDF fajlova
==================================================================

📖 Procesovanje: Srpska zemlja u doba Nemanjica, Djordje Bubalo.pdf
   Period: Средњи век
    ✓ Ekstraktovano 245678 karaktera
    📄 Kreirano 246 chunkova
    💾 Dodavanje u bazu...
    ⏳ 5/246 chunkova dodato
    ⏳ 10/246 chunkova dodato
    ...
    ✅ Završeno - 246/246 chunkova uspešno

==================================================================
📊 FINALNI IZVEŠTAJ
==================================================================
✅ Ukupno procesovanih fajlova:  24
📝 Ukupno kreiranih chunkova:   3456
❌ Neuspešnih fajlova:          2

📂 Po periodima:
   Нови век                  - 8 fajlova, 1234 chunkova
   Рани средњи век           - 5 fajlova, 567 chunkova
   Средњи век               - 9 fajlova, 1456 chunkova
   Остало                   - 2 fajla, 199 chunkova

💾 Ukupno dokumenata u bazi:    3456
==================================================================
```

## Parametri (u script-u)

Možeš prilagoditi na vrhu `pdf_parser.py`:

```python
CHUNK_SIZE = 1000        # Veličina jednog chunka (karakteri)
CHUNK_OVERLAP = 200      # Preklapanje između chunkova
```

Za OCR:
```python
max_pages = 50           # Maksimalan broj stranica za OCR po dokumentu
```

## Napredne opcije

### Procesuj samo jedan period:

Izmeni `main()` funkciju da poziva:
```python
await process_directory("srednji_vek")  # Samo srednji vek
```

### Prilagodi chunking strategiju:

U `split_text_into_chunks()` funkciji možeš prilagoditi:
- Veličinu chunka
- Preklapanje
- Logiku za "pametno" deljenje (na kraju rečenice, paragrafa)

## Troubleshooting

### "Tesseract nije pronađen"
```powershell
# Dodaj u PATH ili navedi eksplicitno:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### "Jezik 'srp' nije dostupan"
Prilikom instalacije Tesseract-a označi "Serbian" jezik ili naknadno preuzmi:
- https://github.com/tesseract-ocr/tessdata

### "Rate limit exceeded" (OpenAI)
Povećaj `await asyncio.sleep()` delay u `process_pdf_file()` funkciji.

### "Qdrant connection refused"
Pokreni Qdrant:
```powershell
docker run -p 6333:6333 qdrant/qdrant
```

## Performance

- **Bez OCR**: ~5-10 stranica/sekund
- **Sa OCR**: ~1-2 stranice/sekund
- **API calls**: ~0.3 sekundi između embeddings (rate limiting)

Za 100 stranica (sa OCR): očekuj ~10-15 minuta

## Napomene

⚠️ **API troškovi**: OpenAI embeddings koštaju - svaki chunk = 1 API call  
⚠️ **RAM**: OCR može koristiti dosta memorije za velike PDF-ove  
⚠️ **Duplikati**: Ponavljanje script-a će dodavati nove chunke (neće proveravati duplikate)

## Autor

Vitomir Jovanović
