"""
fotocroismo_razbidtse_orb_ransac.py
====================================
Misurazione della deviazione rifrattiva apparente nelle immagini
prodotte con prisma dicroico X-cube K9 (fotocroismo razbidtse).

Autore: Dmitrij Musella
Contatto: dmanatolievic@gmail.com
Portfolio: https://www.instagram.com/dmitrijanatolievic/
Versione: 1.0 (2026)

Descrizione
-----------
Questo script misura la deviazione geometrica (scala apparente) tra
un'immagine di riferimento senza prisma e una o più immagini prodotte
attraverso il prisma K9 X-cube (fotocroismo razbidtse).

Il metodo si articola in quattro fasi:
1. Preprocessamento: conversione in scala di grigi + CLAHE
2. Rilevamento keypoint: algoritmo ORB (Oriented FAST and Rotated BRIEF)
3. Matching e filtraggio: Brute Force Matcher + test rapporto di Lowe
4. Stima omografia: RANSAC per calcolo matrice di trasformazione H (3x3)
   da cui si estrae il fattore di scala apparente

Risultati ottenuti nella misurazione esplorativa (2025):
- Media robusta (4 orientazioni valide su 5): +1.86%
- Range: da +0.61% a +3.15%
- Deviazione standard: ±0.15
- L'ipotesi iniziale del fotografo (+2%) è compatibile con i dati

Nota metodologica
-----------------
Questa è una prima verifica esplorativa su campione limitato.
Per una validazione rigorosa si raccomanda:
- Target di calibrazione certificato (scacchiera con dimensioni note)
- Test su almeno 3 soggetti distinti e 3 lunghezze focali diverse
- Calcolo degli intervalli di confidenza al 95%
- Immagini prodotte con treppiede per comparazione geometrica controllata

Dipendenze
----------
pip install opencv-python numpy

Utilizzo
--------
python fotocroismo_orb_ransac.py --ref immagine_riferimento.jpg
                                  --test img1.jpg img2.jpg img3.jpg
                                  [--output risultati.csv]
                                  [--show]

Oppure come modulo:
    from fotocroismo_orb_ransac import misura_deviazione
    scala, inliers = misura_deviazione("riferimento.jpg", "test.jpg")

Citazione
---------
Se usi questo codice in una pubblicazione, cita:
Musella, D. (2026). Fotocroismo razbidtse: Verso una teoria della
frammentazione cromatica nella fotografia contemporanea (v9).
DOI: 10.5281/zenodo.19319995
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np


# ─────────────────────────────────────────────
# Parametri configurabili
# ─────────────────────────────────────────────

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

# ORB
ORB_MAX_FEATURES = 2000

# Lowe ratio test
LOWE_RATIO = 0.75

# RANSAC
RANSAC_REPROJ_THRESHOLD = 5.0

# Soglia minima di inliers per considerare il matching affidabile
MIN_INLIERS = 30


# ─────────────────────────────────────────────
# Funzioni core
# ─────────────────────────────────────────────

def preprocessa(immagine_bgr):
    """
    Converte in scala di grigi e applica equalizzazione adattiva
    dell'istogramma (CLAHE) per compensare le dominanti cromatiche
    introdotte dal prisma K9.

    Args:
        immagine_bgr: immagine BGR caricata con cv2.imread

    Returns:
        immagine in scala di grigi equalizzata
    """
    grigio = cv2.cvtColor(immagine_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    return clahe.apply(grigio)


def rileva_keypoint(immagine_grigio):
    """
    Rileva fino a ORB_MAX_FEATURES keypoint e calcola i descrittori
    con l'algoritmo ORB (Oriented FAST and Rotated BRIEF).

    ORB è stato scelto per la sua robustezza con immagini fortemente
    alterate cromaticamente, condizione tipica delle immagini razbidtse.

    Args:
        immagine_grigio: immagine in scala di grigi preprocessata

    Returns:
        tuple (keypoint, descrittori)
    """
    orb = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
    keypoint, descrittori = orb.detectAndCompute(immagine_grigio, None)
    return keypoint, descrittori


def filtra_match(desc_ref, desc_test):
    """
    Confronta i descrittori con Brute Force Matcher (distanza Hamming)
    e applica il test del rapporto di Lowe per eliminare match ambigui.

    Per ogni punto nell'immagine di riferimento, il match è accettato
    solo se la distanza con il punto corrispondente è significativamente
    inferiore al secondo candidato (soglia: LOWE_RATIO).

    Args:
        desc_ref: descrittori immagine di riferimento
        desc_test: descrittori immagine di test

    Returns:
        lista di match accettati
    """
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    coppie = matcher.knnMatch(desc_ref, desc_test, k=2)

    match_buoni = []
    for coppia in coppie:
        if len(coppia) == 2:
            m, n = coppia
            if m.distance < LOWE_RATIO * n.distance:
                match_buoni.append(m)

    return match_buoni


def stima_scala(kp_ref, kp_test, match):
    """
    Calcola la matrice di omografia H (3x3) tramite RANSAC e ne estrae
    il fattore di scala apparente.

    La scala viene calcolata come media dei fattori di scala lungo i
    due assi della matrice H:
        sx = sqrt(H[0,0]^2 + H[1,0]^2)
        sy = sqrt(H[0,1]^2 + H[1,1]^2)
        scala = (sx + sy) / 2

    Un valore > 1.0 indica che la scena appare leggermente ingrandita
    attraverso il prisma rispetto all'immagine di riferimento.

    Args:
        kp_ref: keypoint immagine di riferimento
        kp_test: keypoint immagine di test
        match: lista di match filtrati

    Returns:
        tuple (scala, num_inliers) oppure (None, 0) se il matching
        non è sufficiente
    """
    if len(match) < MIN_INLIERS:
        return None, len(match)

    punti_ref = np.float32(
        [kp_ref[m.queryIdx].pt for m in match]
    ).reshape(-1, 1, 2)

    punti_test = np.float32(
        [kp_test[m.trainIdx].pt for m in match]
    ).reshape(-1, 1, 2)

    H, maschera = cv2.findHomography(
        punti_ref,
        punti_test,
        cv2.RANSAC,
        RANSAC_REPROJ_THRESHOLD
    )

    if H is None:
        return None, 0

    num_inliers = int(maschera.sum()) if maschera is not None else 0

    sx = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
    sy = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
    scala = (sx + sy) / 2.0

    return scala, num_inliers


def misura_deviazione(percorso_ref, percorso_test, verbose=True):
    """
    Funzione principale: misura la deviazione rifrattiva apparente
    tra un'immagine di riferimento (senza prisma) e un'immagine di test
    (con prisma K9 X-cube).

    Args:
        percorso_ref: percorso immagine di riferimento
        percorso_test: percorso immagine di test
        verbose: se True stampa i risultati a schermo

    Returns:
        dict con chiavi:
            'file': nome file di test
            'scala': fattore di scala (es. 1.0186)
            'deviazione_pct': deviazione percentuale (es. +1.86)
            'inliers': numero di inliers RANSAC
            'affidabile': True se inliers >= MIN_INLIERS
    """
    # Caricamento
    img_ref = cv2.imread(percorso_ref)
    img_test = cv2.imread(percorso_test)

    if img_ref is None:
        raise FileNotFoundError(
            f"Impossibile caricare l'immagine di riferimento: {percorso_ref}"
        )
    if img_test is None:
        raise FileNotFoundError(
            f"Impossibile caricare l'immagine di test: {percorso_test}"
        )

    # Pipeline
    ref_gray = preprocessa(img_ref)
    test_gray = preprocessa(img_test)

    kp_ref, desc_ref = rileva_keypoint(ref_gray)
    kp_test, desc_test = rileva_keypoint(test_gray)

    if desc_ref is None or desc_test is None:
        if verbose:
            print(f"  {os.path.basename(percorso_test)}: nessun keypoint rilevato")
        return {
            'file': os.path.basename(percorso_test),
            'scala': None,
            'deviazione_pct': None,
            'inliers': 0,
            'affidabile': False
        }

    match = filtra_match(desc_ref, desc_test)
    scala, inliers = stima_scala(kp_ref, kp_test, match)

    if scala is None:
        if verbose:
            print(
                f"  {os.path.basename(percorso_test)}: "
                f"matching insufficiente ({inliers} inliers < {MIN_INLIERS})"
            )
        return {
            'file': os.path.basename(percorso_test),
            'scala': None,
            'deviazione_pct': None,
            'inliers': inliers,
            'affidabile': False
        }

    deviazione_pct = (scala - 1.0) * 100.0
    segno = "+" if deviazione_pct >= 0 else ""

    if verbose:
        print(
            f"  {os.path.basename(percorso_test)}: "
            f"scala {scala:.4f}  |  deviazione {segno}{deviazione_pct:.2f}%  |  "
            f"{inliers} inliers"
        )

    return {
        'file': os.path.basename(percorso_test),
        'scala': scala,
        'deviazione_pct': deviazione_pct,
        'inliers': inliers,
        'affidabile': True
    }


def calcola_statistiche(risultati):
    """
    Calcola media robusta e deviazione standard sui risultati affidabili.

    Args:
        risultati: lista di dict prodotti da misura_deviazione

    Returns:
        dict con statistiche o None se nessun risultato affidabile
    """
    validi = [r for r in risultati if r['affidabile']]
    if not validi:
        return None

    scale = [r['scala'] for r in validi]
    deviazioni = [r['deviazione_pct'] for r in validi]
    inliers = [r['inliers'] for r in validi]

    media_scala = np.mean(scale)
    media_dev = np.mean(deviazioni)
    std_dev = np.std(deviazioni)
    segno = "+" if media_dev >= 0 else ""

    return {
        'n_validi': len(validi),
        'n_totali': len(risultati),
        'media_scala': media_scala,
        'media_deviazione_pct': media_dev,
        'std_deviazione_pct': std_dev,
        'min_deviazione_pct': min(deviazioni),
        'max_deviazione_pct': max(deviazioni),
        'media_inliers': np.mean(inliers)
    }


# ─────────────────────────────────────────────
# Interfaccia a riga di comando
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Misura la deviazione rifrattiva apparente nelle immagini "
            "di fotocroismo razbidtse (prisma K9 X-cube)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python fotocroismo_orb_ransac.py --ref ref.jpg --test p1.jpg p2.jpg p3.jpg
  python fotocroismo_orb_ransac.py --ref ref.jpg --test *.jpg --output risultati.csv
        """
    )

    parser.add_argument(
        "--ref",
        required=True,
        metavar="FILE",
        help="Immagine di riferimento senza prisma"
    )
    parser.add_argument(
        "--test",
        required=True,
        nargs="+",
        metavar="FILE",
        help="Una o più immagini prodotte con il prisma K9"
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Salva i risultati in formato CSV (opzionale)"
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=MIN_INLIERS,
        metavar="N",
        help=f"Soglia minima inliers per matching affidabile (default: {MIN_INLIERS})"
    )

    args = parser.parse_args()

    # Aggiornamento soglia se specificata dall'utente
    if args.min_inliers != 30:
        import fotocroismo_orb_ransac as _self
        _self.MIN_INLIERS = args.min_inliers

    print("=" * 60)
    print("Fotocroismo razbidtse — Misurazione deviazione rifrattiva")
    print("Dmitrij Musella | DOI: 10.5281/zenodo.19319995")
    print("=" * 60)
    print(f"\nRiferimento: {args.ref}")
    print(f"Immagini di test: {len(args.test)}")
    print(f"Soglia inliers: {MIN_INLIERS}\n")

    risultati = []
    for percorso_test in args.test:
        risultato = misura_deviazione(args.ref, percorso_test, verbose=True)
        risultati.append(risultato)

    # Statistiche
    stats = calcola_statistiche(risultati)
    if stats:
        segno = "+" if stats['media_deviazione_pct'] >= 0 else ""
        print("\n" + "-" * 60)
        print("STATISTICHE")
        print("-" * 60)
        print(f"Misurazioni valide:    {stats['n_validi']} / {stats['n_totali']}")
        print(f"Media scala:           {stats['media_scala']:.4f}")
        print(
            f"Media deviazione:      "
            f"{segno}{stats['media_deviazione_pct']:.2f}%"
        )
        print(
            f"Dev. standard:         "
            f"\u00b1{stats['std_deviazione_pct']:.2f}%"
        )
        print(
            f"Range:                 "
            f"{stats['min_deviazione_pct']:+.2f}% / "
            f"{stats['max_deviazione_pct']:+.2f}%"
        )
        print(f"Media inliers:         {stats['media_inliers']:.0f}")
    else:
        print("\nNessun risultato valido ottenuto.")

    # Export CSV
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "scala", "deviazione_pct", "inliers", "affidabile"]
            )
            writer.writeheader()
            writer.writerows(risultati)
        print(f"\nRisultati salvati in: {args.output}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
