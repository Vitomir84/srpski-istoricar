"""
Test script - provera sistema za PDF parsing
"""
import sys

def check_imports():
    """Proveri da li su sve potrebne biblioteke instalirane"""
    print("🔍 Proveravam Python biblioteke...\n")
    
    required = {
        'pypdf': 'PyPDF',
        'pytesseract': 'pytesseract',
        'pdf2image': 'pdf2image',
        'PIL': 'Pillow',
        'qdrant_client': 'qdrant-client',
        'openai': 'openai',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NEDOSTAJE!")
            all_ok = False
    
    return all_ok


def check_tesseract():
    """Proveri da li je Tesseract OCR instaliran"""
    print("\n🔍 Proveravam Tesseract OCR...\n")
    
    try:
        import pytesseract
        from PIL import Image
        
        # Pokušaj da dobiješ verziju
        version = pytesseract.get_tesseract_version()
        print(f"  ✅ Tesseract {version} pronađen")
        
        # Proveri srpski jezik
        try:
            langs = pytesseract.get_languages()
            if 'srp' in langs:
                print(f"  ✅ Srpski jezik (srp) dostupan")
            else:
                print(f"  ⚠️  Srpski jezik (srp) NIJE dostupan")
                print(f"      Dostupni jezici: {', '.join(langs[:5])}...")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Greška pri proveri jezika: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Tesseract NIJE pronađen!")
        print(f"      Greška: {e}")
        print(f"\n  📥 Instaliraj Tesseract:")
        print(f"      https://github.com/UB-Mannheim/tesseract/wiki")
        return False


def check_qdrant():
    """Proveri konekciju sa Qdrant bazom"""
    print("\n🔍 Proveravam Qdrant bazu...\n")
    
    try:
        from qdrant_client import QdrantClient
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections()
        
        print(f"  ✅ Qdrant server dostupan na {qdrant_url}")
        print(f"  📊 Broj kolekcija: {len(collections.collections)}")
        
        # Proveri serbian_history kolekciju
        try:
            info = client.get_collection("serbian_history")
            print(f"  ✅ Kolekcija 'serbian_history' postoji")
            print(f"      Dokumenata: {info.points_count}")
            print(f"      Vektor dimenzija: {info.config.params.vectors.size}")
        except Exception:
            print(f"  ℹ️  Kolekcija 'serbian_history' ne postoji (biće kreirana)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Qdrant NIJE dostupan!")
        print(f"      Greška: {e}")
        print(f"\n  🐳 Pokreni Qdrant sa Docker:")
        print(f"      docker run -p 6333:6333 qdrant/qdrant")
        return False


def check_openai():
    """Proveri OpenAI API key"""
    print("\n🔍 Proveravam OpenAI API...\n")
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print(f"  ❌ OPENAI_API_KEY nije setovan!")
        print(f"      Dodaj u .env fajl")
        return False
    
    # Sakrij većinu API key-a
    masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"  ✅ API Key pronađen: {masked}")
    
    return True


def check_pdf_files():
    """Proveri da li postoje PDF fajlovi za procesovanje"""
    print("\n🔍 Proveravam PDF fajlove...\n")
    
    from pathlib import Path
    
    docs_dir = Path("docs")
    periods = ["novi_vek", "rani_vek", "srednji_vek", "ostalo"]
    
    if not docs_dir.exists():
        print(f"  ❌ Folder 'docs' ne postoji!")
        return False
    
    total_pdfs = 0
    for period in periods:
        period_dir = docs_dir / period
        if period_dir.exists():
            pdfs = list(period_dir.glob("*.pdf"))
            print(f"  📁 {period:15} - {len(pdfs):3} PDF fajlova")
            total_pdfs += len(pdfs)
        else:
            print(f"  ⚠️  {period:15} - folder ne postoji")
    
    print(f"\n  📄 Ukupno PDF fajlova: {total_pdfs}")
    
    if total_pdfs == 0:
        print(f"  ⚠️  Nema PDF fajlova za procesovanje!")
        return False
    
    return True


def main():
    """Glavna test funkcija"""
    print("\n" + "="*70)
    print("🇷🇸 SRPSKI ISTORIČAR - Sistem Test")
    print("="*70 + "\n")
    
    checks = [
        ("Python biblioteke", check_imports),
        ("Tesseract OCR", check_tesseract),
        ("Qdrant baza", check_qdrant),
        ("OpenAI API", check_openai),
        ("PDF fajlovi", check_pdf_files)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ Greška tokom provere: {e}")
            results.append((name, False))
    
    # Finalni izveštaj
    print("\n" + "="*70)
    print("📊 FINALNI IZVEŠTAJ")
    print("="*70 + "\n")
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n  🎉 Sistem je spreman!")
        print("  ▶️  Pokreni: poetry run python pdf_parser.py")
    else:
        print("\n  ⚠️  Sistem nije potpuno spreman")
        print("      Reši gornje probleme pre pokretanja parsera")
    
    print("\n" + "="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
