"""
Extracteur de profil utilisateur depuis les messages
Utilise des expressions régulières et du NLP pour extraire les informations
"""

import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ProfileExtractor:
    """
    Extrait les informations du profil utilisateur depuis les messages textuels
    
    Extrait:
    - Série BAC (S1, S2, S3)
    - Notes (MATH, SCPH, FR, AN, PHILO, SVT, HG, EPS)
    - Âge
    - Sexe
    - Résidence
    - Académie
    - Formation souhaitée
    """
    
    def __init__(self):
        """Initialise l'extracteur avec les patterns de reconnaissance"""
        # Patterns pour la série
        self.serie_patterns = [
            (r'\b(s[123]|série\s*[123])\b', lambda m: f"S{m.group(1)[-1]}"),
            (r'\b(s1|série\s*1)\b', 'S1'),
            (r'\b(s2|série\s*2)\b', 'S2'),
            (r'\b(s3|série\s*3)\b', 'S3'),
        ]
        
        # Patterns pour les notes
        self.note_patterns = {
            "MATH": [
                r'math(?:ématiques)?[:\s]*(\d+(?:[.,]\d+)?)',
                r'note\s*(?:en\s*)?math[:\s]*(\d+(?:[.,]\d+)?)',
                r'maths?\s+(\d+(?:[.,]\d+)?)',  # "Maths 16"
            ],
            "SCPH": [
                r'scph[:\s]*(\d+(?:[.,]\d+)?)',
                r'sciences?\s*physiques?[:\s]*(\d+(?:[.,]\d+)?)',
                r'physique[:\s]*(\d+(?:[.,]\d+)?)',
                r'physique\s+(\d+(?:[.,]\d+)?)',  # "Physique 15"
                r'\bpc[:\s]*(\d+(?:[.,]\d+)?)',  # "PC: 13" ou "PC 13"
                r'physique[:\s]*chimie[:\s]*(\d+(?:[.,]\d+)?)',  # "Physique-Chimie 13"
            ],
            "FR": [
                r'français[:\s]*(\d+(?:[.,]\d+)?)',
                r'fr[:\s]*(\d+(?:[.,]\d+)?)',
                r'français\s+(\d+(?:[.,]\d+)?)',  # "Français 13"
            ],
            "AN": [
                r'anglais[:\s]*(\d+(?:[.,]\d+)?)',
                r'an[:\s]*(\d+(?:[.,]\d+)?)',
                r'anglais\s+(\d+(?:[.,]\d+)?)',  # "Anglais 12"
            ],
            "PHILO": [
                r'philo(?:sophie)?[:\s]*(\d+(?:[.,]\d+)?)',
                r'philo\s+(\d+(?:[.,]\d+)?)',  # "Philo 11"
                r'note\s+(?:de\s+|en\s+)?philo(?:sophie)?[:\s]*(\d+(?:[.,]\d+)?)',  # "note de philosophie 12"
                r'philosophie\s+(\d+(?:[.,]\d+)?)',  # "philosophie 12"
                r'philo(?:sophie)?\s+est\s+(\d+(?:[.,]\d+)?)',  # "philosophie est 12"
            ],
            "SVT": [
                r'svt[:\s]*(\d+(?:[.,]\d+)?)',
                r'sciences?\s*(?:de\s*la\s*vie\s*et\s*de\s*la\s*terre|vie)[:\s]*(\d+(?:[.,]\d+)?)',
                r'svt\s+(\d+(?:[.,]\d+)?)',  # "SVT 14"
            ],
            "HG": [
                r'hist(?:oire)?[:\s]*(\d+(?:[.,]\d+)?)',
                r'hg[:\s]*(\d+(?:[.,]\d+)?)',
                r'hist(?:oire)?\s+(\d+(?:[.,]\d+)?)',  # "Histoire 11"
            ],
            "EPS": [
                r'eps[:\s]*(\d+(?:[.,]\d+)?)',
                r'sport[:\s]*(\d+(?:[.,]\d+)?)',
            ],
        }
        
        # Patterns pour l'âge
        self.age_patterns = [
            r'\b(\d{2})\s*ans?\b',
            r'âge[:\s]*(\d{2})',
            r'j\'?ai\s*(\d{2})\s*ans?',
        ]
        
        # Patterns pour le sexe
        self.sexe_patterns = [
            (r'\b(homme|masculin|garçon|m)\b', 'M'),
            (r'\b(femme|féminin|fille|f)\b', 'F'),
        ]
        
        # Patterns pour la formation
        self.formation_patterns = [
            (r'\b(mpi|math.*phys.*info)\b', 'L1MPI'),
            (r'\b(bcgs|bio.*chim.*géol)\b', 'L1BCGS'),
            (r'\b(pcsm|phys.*chim.*math)\b', 'L1PCSM'),
        ]
    
    def extract(
        self,
        message: str,
        existing_profile: Optional[Dict] = None,
        context: Optional[str] = None
    ) -> Dict:
        """
        Extrait les informations du profil depuis un message
        
        Args:
            message: Message de l'utilisateur
            existing_profile: Profil existant à mettre à jour
            context: Contexte de la conversation (pour aider à interpréter les réponses courtes)
        
        Returns:
            Dictionnaire avec les informations extraites
        """
        message_lower = message.lower()
        extracted = {}
        
        # Extraire la série
        serie = self._extract_serie(message_lower)
        if serie:
            extracted["serie"] = serie
        
        # Extraire les notes (avec contexte pour les réponses courtes)
        notes = self._extract_notes(message_lower, context=context)
        if notes:
            if "notes" not in extracted:
                extracted["notes"] = {}
            extracted["notes"].update(notes)
        
        # Extraire l'âge
        age = self._extract_age(message_lower)
        if age:
            extracted["age"] = age
        
        # Extraire le sexe
        sexe = self._extract_sexe(message_lower)
        if sexe:
            extracted["sexe"] = sexe
        
        # Extraire la formation
        formation = self._extract_formation(message_lower)
        if formation:
            extracted["formation"] = formation
        
        # Extraire la résidence et l'académie (basique)
        residence = self._extract_residence(message_lower)
        if residence:
            extracted["residence"] = residence
        
        academie = self._extract_academie(message_lower)
        if academie:
            extracted["academie"] = academie
        
        # Fusionner avec le profil existant
        if existing_profile:
            merged = existing_profile.copy()
            if "notes" in extracted and "notes" in merged:
                merged["notes"].update(extracted["notes"])
                extracted["notes"] = merged["notes"]
            merged.update(extracted)
            return merged
        
        return extracted
    
    def _extract_serie(self, message: str) -> Optional[str]:
        """Extrait la série BAC"""
        for pattern, replacement in self.serie_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                if callable(replacement):
                    return replacement(match)
                return replacement
        return None
    
    def _extract_notes(self, message: str, context: Optional[str] = None) -> Dict[str, float]:
        """
        Extrait les notes depuis le message
        
        Args:
            message: Message de l'utilisateur
            context: Contexte de la conversation (pour détecter quelle note est demandée)
        """
        notes = {}
        message_lower = message.lower()
        
        # Si le message est juste un nombre, essayer de le mapper au contexte
        if context and re.match(r'^\s*(\d+(?:[.,]\d+)?)\s*$', message.strip()):
            # Message est juste un nombre, utiliser le contexte
            context_lower = context.lower()
            for subject, patterns in self.note_patterns.items():
                # Chercher le nom de la matière dans le contexte
                subject_keywords = {
                    "MATH": ["math", "mathématiques"],
                    "SCPH": ["physique", "sciences physiques", "scph"],
                    "FR": ["français", "fr"],
                    "AN": ["anglais", "an"],
                    "PHILO": ["philo", "philosophie"],
                    "SVT": ["svt", "sciences de la vie", "sciences vie"],
                    "HG": ["histoire", "géographie", "hg"]
                }
                
                if subject in subject_keywords:
                    for keyword in subject_keywords[subject]:
                        if keyword in context_lower:
                            try:
                                note_str = message.strip().replace(',', '.')
                                note = float(note_str)
                                if 0 <= note <= 20:
                                    notes[subject] = note
                                    return notes
                            except (ValueError):
                                continue
        
        # Extraction normale avec patterns
        for subject, patterns in self.note_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, message_lower, re.IGNORECASE)
                if match:
                    try:
                        # Vérifier que le groupe existe
                        if match.lastindex is None or match.lastindex < 1:
                            continue
                        note_str = match.group(1)
                        if note_str is None:
                            continue
                        note_str = note_str.replace(',', '.')
                        note = float(note_str)
                        # Valider que la note est entre 0 et 20
                        if 0 <= note <= 20:
                            notes[subject] = note
                            break
                    except (ValueError, IndexError, AttributeError):
                        continue
        
        return notes
    
    def _extract_age(self, message: str) -> Optional[int]:
        """Extrait l'âge"""
        for pattern in self.age_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    age = int(match.group(1))
                    # Valider que l'âge est raisonnable (15-30 ans)
                    if 15 <= age <= 30:
                        return age
                except (ValueError, IndexError):
                    continue
        return None
    
    def _extract_sexe(self, message: str) -> Optional[str]:
        """Extrait le sexe"""
        for pattern, sexe in self.sexe_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return sexe
        return None
    
    def _extract_formation(self, message: str) -> Optional[str]:
        """Extrait la formation souhaitée"""
        for pattern, formation in self.formation_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return formation
        return None
    
    def _extract_residence(self, message: str) -> Optional[str]:
        """Extrait la résidence (basique)"""
        # Liste des villes principales du Sénégal
        villes = [
            "dakar", "thies", "saint-louis", "kaolack", "ziguinchor",
            "touba", "mbour", "louga", "tambacounda", "kolda", "fatick"
        ]
        
        for ville in villes:
            if ville in message:
                return ville.capitalize()
        
        return None
    
    def _extract_academie(self, message: str) -> Optional[str]:
        """Extrait l'académie"""
        academies = [
            "dakar", "thies", "saint-louis", "kaolack", "ziguinchor",
            "louga", "tambacounda", "kolda", "fatick", "diourbel"
        ]
        
        for academie in academies:
            if f"académie {academie}" in message or academie in message:
                return f"Académie de {academie.capitalize()}"
        
        return None

