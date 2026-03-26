"""
Automatski popuni bibliografiju analiziranjem tekstualnih dokumenata
Koristi AI (OpenAI) da izvuče bibliografske podatke iz sadržaja fajlova
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
TEXT_DOCS_FOLDER = Path("docs/tekstualni_fajlovi")
BIBLIOGRAPHY_FILE = Path("bibliografija.json")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


def load_text_file(file_path: Path, max_chars: int = 3000) -> str:
    """Load text from file with multiple encoding attempts"""
    encodings = ['utf-8', 'utf-8-sig', 'windows-1252', 'cp1252', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read(max_chars)  # Read first 3000 chars
            return text
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"    ⚠️  Greška: {e}")
    
    return ""


def load_bibliography() -> Dict:
    """Load existing bibliography"""
    if BIBLIOGRAPHY_FILE.exists():
        try:
            with open(BIBLIOGRAPHY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Greška pri učitavanju bibliografije: {e}")
            return {}
    return {}


def save_bibliography(bibliography: Dict):
    """Save bibliography to JSON file"""
    try:
        # Create backup first
        if BIBLIOGRAPHY_FILE.exists():
            backup_path = BIBLIOGRAPHY_FILE.with_suffix('.json.backup')
            with open(BIBLIOGRAPHY_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Backup kreiran: {backup_path}")
        
        # Save updated bibliography
        with open(BIBLIOGRAPHY_FILE, 'w', encoding='utf-8') as f:
            json.dump(bibliography, f, ensure_ascii=False, indent=2)
        print(f"✓ Bibliografija sačuvana: {BIBLIOGRAPHY_FILE}")
    except Exception as e:
        print(f"❌ Greška pri čuvanju: {e}")


async def extract_bibliography_info(filename: str, text_content: str) -> Dict:
    """Use AI to extract bibliographic information from text"""
    
    prompt = f"""Analiziraj početak ovog dokumenta i izvuci bibliografske informacije.

NAZIV FAJLA: {filename}

POČETAK DOKUMENTA:
{text_content}

Zadatak: Izvuci sledeće bibliografske podatke iz teksta:
- autor: Puno ime autora ili autora (ako ih ima više, navedi sve)
- godina: Godina objavljivanja
- naslov: Tačan naslov knjige/dokumenta
- izdavac: Naziv izdavača
- strane: Broj strana ili raspon strana (ako je navedeno)

PRAVILA:
1. Ako neka informacija nije jasno navedena, ostavi prazno polje ""
2. Za autora: Koristi puno ime (npr. "Branko Petranović" umesto "B. Petranović")
3. Za godinu: Samo brojevi (npr. "1988")
4. Za naslov: Koristi tačan naslov ako je naveden na početku, inače iz imena fajla
5. Ako je ovo scan ili prepis bez metapodataka, pokušaj da zaključiš iz konteksta

Odgovori ISKLJUČIVO u ovom JSON formatu:
{{
  "autor": "...",
  "godina": "...",
  "naslov": "...",
  "izdavac": "...",
  "strane": "..."
}}"""

    try:
        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Ti si bibliografski ekspert koji analizira dokumente i izvlači bibliografske podatke. Uvek odgovaraš u JSON formatu."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        # Remove markdown code blocks if present
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        data = json.loads(result.strip())
        
        # Validate structure
        required_keys = ['autor', 'godina', 'naslov', 'izdavac', 'strane']
        for key in required_keys:
            if key not in data:
                data[key] = ""
        
        return data
        
    except Exception as e:
        print(f"    ⚠️  Greška pri analizi: {e}")
        return {
            "autor": "",
            "godina": "",
            "naslov": "",
            "izdavac": "",
            "strane": ""
        }


async def process_all_documents():
    """Process all text documents and populate bibliography"""
    
    print("\n" + "="*70)
    print("🤖 AUTOMATSKO POPUNJAVANJE BIBLIOGRAFIJE")
    print("="*70 + "\n")
    
    if not TEXT_DOCS_FOLDER.exists():
        print(f"❌ Folder ne postoji: {TEXT_DOCS_FOLDER}")
        return
    
    # Load existing bibliography
    bibliography = load_bibliography()
    print(f"✓ Učitana postojeća bibliografija: {len(bibliography)} unosa\n")
    
    # Find all text files
    txt_files = sorted(TEXT_DOCS_FOLDER.glob("*.txt"))
    
    if not txt_files:
        print(f"❌ Nema .txt fajlova u {TEXT_DOCS_FOLDER}")
        return
    
    print(f"📚 Pronađeno {len(txt_files)} tekstualnih fajlova\n")
    print("Analiziram dokumente...\n")
    
    updated_count = 0
    skipped_count = 0
    
    for i, file_path in enumerate(txt_files, 1):
        filename = file_path.name
        
        print(f"[{i}/{len(txt_files)}] {filename}")
        
        # Check if already has complete info
        if filename in bibliography:
            bib = bibliography[filename]
            has_autor = bib.get('autor', '').strip()
            has_godina = bib.get('godina', '').strip()
            has_izdavac = bib.get('izdavac', '').strip()
            
            if has_autor and has_godina and has_izdavac:
                print(f"    ✓ Već ima kompletne podatke, preskačem\n")
                skipped_count += 1
                continue
        
        # Load document content
        print(f"    📖 Čitam dokument...")
        text_content = load_text_file(file_path, max_chars=3000)
        
        if not text_content:
            print(f"    ⚠️  Ne mogu da učitam sadržaj\n")
            continue
        
        # Extract info using AI
        print(f"    🤖 Analiziram sa AI...")
        extracted_info = await extract_bibliography_info(filename, text_content)
        
        # Merge with existing data (keep existing if not empty)
        if filename not in bibliography:
            bibliography[filename] = {}
        
        for key in ['autor', 'godina', 'naslov', 'izdavac', 'strane']:
            existing = bibliography[filename].get(key, '').strip()
            extracted = extracted_info.get(key, '').strip()
            
            # Keep existing if it's not empty, otherwise use extracted
            if not existing and extracted:
                bibliography[filename][key] = extracted
            elif existing:
                bibliography[filename][key] = existing
            else:
                bibliography[filename][key] = ""
        
        # Show what was found
        autor = bibliography[filename]['autor']
        godina = bibliography[filename]['godina']
        naslov = bibliography[filename]['naslov']
        
        if autor or godina:
            citation_parts = []
            if autor:
                citation_parts.append(autor)
            if godina:
                if citation_parts:
                    citation_parts[-1] += f" ({godina})"
                else:
                    citation_parts.append(f"({godina})")
            if naslov:
                citation_parts.append(naslov)
            
            print(f"    ✓ {'. '.join(citation_parts)}")
        else:
            print(f"    ⚠️  Nisam našao autor/godinu")
        
        updated_count += 1
        print()
        
        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)
        
        # Save after every 5 files (incremental save)
        if i % 5 == 0:
            save_bibliography(bibliography)
    
    # Final save
    save_bibliography(bibliography)
    
    print("="*70)
    print("✅ ZAVRŠENO!")
    print(f"   📝 Analizirano fajlova: {updated_count}")
    print(f"   ⏭️  Preskočeno (već ima podatke): {skipped_count}")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(process_all_documents())
