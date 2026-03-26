"""
Ažuriraj postojeće dokumente u FAISS bazi sa bibliografskim referencama
"""
import os
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./faiss_data/metadata.pkl")
BIBLIOGRAPHY_FILE = Path("bibliografija.json")


def load_bibliography():
    """Load bibliography data from JSON file"""
    if BIBLIOGRAPHY_FILE.exists():
        try:
            with open(BIBLIOGRAPHY_FILE, 'r', encoding='utf-8') as f:
                bibliography = json.load(f)
            print(f"✓ Učitana bibliografija: {len(bibliography)} unosa")
            return bibliography
        except Exception as e:
            print(f"❌ Greška pri učitavanju bibliografije: {e}")
            return {}
    else:
        print(f"❌ Bibliografija nije pronađena: {BIBLIOGRAPHY_FILE}")
        return {}


def format_citation(filename: str, bibliography: dict) -> str:
    """Format citation in the required format: Autor (godina). Naslov. Izdavač. strane"""
    if filename not in bibliography:
        return filename
    
    bib = bibliography[filename]
    parts = []
    
    # Autor
    if bib.get('autor'):
        parts.append(bib['autor'])
    
    # Godina
    if bib.get('godina'):
        if parts:
            parts[-1] += f" ({bib['godina']})"
        else:
            parts.append(f"({bib['godina']})")
    
    # Naslov
    if bib.get('naslov'):
        parts.append(bib['naslov'])
    
    # Izdavač
    if bib.get('izdavac'):
        parts.append(bib['izdavac'])
    
    # Strane
    if bib.get('strane'):
        parts.append(f"str. {bib['strane']}")
    
    return '. '.join(parts) if parts else filename


def update_metadata_with_citations():
    """Update existing metadata with bibliographic citations"""
    
    print("\n" + "="*70)
    print("🇷🇸 AŽURIRANJE METAPODATAKA SA BIBLIOGRAFSKIM REFERENCAMA")
    print("="*70 + "\n")
    
    # Check if metadata file exists
    if not os.path.exists(FAISS_METADATA_PATH):
        print(f"❌ Metadata fajl ne postoji: {FAISS_METADATA_PATH}")
        return
    
    # Load bibliography
    bibliography = load_bibliography()
    if not bibliography:
        print("❌ Nema bibliografskih podataka za ažuriranje")
        return
    
    # Load existing metadata
    try:
        with open(FAISS_METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✓ Učitano {len(metadata)} dokumenata iz baze\n")
    except Exception as e:
        print(f"❌ Greška pri učitavanju metadata: {e}")
        return
    
    # Create backup
    backup_path = FAISS_METADATA_PATH + ".backup"
    try:
        with open(backup_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Kreiran backup: {backup_path}\n")
    except Exception as e:
        print(f"⚠️  Greška pri kreiranju backupa: {e}\n")
    
    # Update metadata with citations
    updated_count = 0
    files_updated = set()
    
    print("📝 Ažuriranje metapodataka...\n")
    
    for doc in metadata:
        filename = doc.get('source_file', '')
        
        if filename and filename in bibliography:
            # Get citation
            citation = format_citation(filename, bibliography)
            bib_data = bibliography[filename]
            
            # Update document metadata
            doc['citation'] = citation
            doc['autor'] = bib_data.get('autor', '')
            doc['godina'] = bib_data.get('godina', '')
            doc['naslov'] = bib_data.get('naslov', '')
            doc['izdavac'] = bib_data.get('izdavac', '')
            doc['strane'] = bib_data.get('strane', '')
            
            updated_count += 1
            files_updated.add(filename)
    
    # Save updated metadata
    try:
        with open(FAISS_METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Sačuvan ažurirani metadata fajl\n")
    except Exception as e:
        print(f"❌ Greška pri čuvanju metadata: {e}")
        return
    
    # Summary
    print("="*70)
    print("✅ AŽURIRANJE ZAVRŠENO!")
    print(f"   📝 Ažurirano dokumenata: {updated_count}")
    print(f"   📚 Ažurirano fajlova: {len(files_updated)}")
    print("="*70 + "\n")
    
    if files_updated:
        print("📚 Ažurirani fajlovi:\n")
        for filename in sorted(files_updated):
            citation = format_citation(filename, bibliography)
            print(f"   • {filename}")
            print(f"     {citation}\n")


if __name__ == "__main__":
    update_metadata_with_citations()
