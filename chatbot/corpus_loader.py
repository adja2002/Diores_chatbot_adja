"""
Module pour charger et traiter le corpus de formations depuis les documents
"""

import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx non disponible. Installation: pip install python-docx")

try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDFLoader non disponible. Installation: pip install pypdf")


class CorpusLoader:
    """
    Charge et traite les documents du corpus pour le système RAG
    """
    
    def __init__(self, corpus_dir: str = "chatbot/corpus"):
        """
        Initialise le chargeur de corpus
        
        Args:
            corpus_dir: Chemin vers le dossier contenant les documents
        """
        self.corpus_dir = corpus_dir
        self.base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            corpus_dir
        ) if not os.path.isabs(corpus_dir) else corpus_dir
    
    def load_documents(self) -> List[Dict]:
        """
        Charge tous les documents du corpus (formations FST)
        
        Returns:
            Liste de dictionnaires avec les données des formations
        """
        formations_data = []
        
        # Chercher le fichier Word
        docx_file = os.path.join(self.base_path, "fst_formations.docx")
        
        if os.path.exists(docx_file):
            logger.info(f"Chargement du document: {docx_file}")
            doc_data = self._load_docx(docx_file)
            if doc_data:
                formations_data.extend(doc_data)
        else:
            logger.warning(f"Fichier {docx_file} introuvable")
            # Fallback vers les données par défaut
            formations_data = self._get_default_data()
        
        return formations_data
    
    def load_universities_guide(self) -> List[Dict]:
        """
        Charge le guide complet des universités du Sénégal depuis le PDF
        
        Returns:
            Liste de dictionnaires avec les données sur les universités
        """
        universities_data = []
        
        # Chercher le fichier PDF
        pdf_file = os.path.join(self.base_path, "guide_complet.pdf")
        
        if os.path.exists(pdf_file):
            logger.info(f"Chargement du guide des universités: {pdf_file}")
            guide_data = self._load_pdf(pdf_file)
            if guide_data:
                universities_data.extend(guide_data)
        else:
            logger.warning(f"Fichier {pdf_file} introuvable")
        
        return universities_data
    
    def _load_docx(self, file_path: str) -> List[Dict]:
        """
        Charge et parse un fichier Word (.docx)
        
        Args:
            file_path: Chemin vers le fichier .docx
        
        Returns:
            Liste de dictionnaires avec les formations extraites
        """
        if not DOCX_AVAILABLE:
            logger.error("python-docx non disponible. Utilisation des données par défaut.")
            return self._get_default_data()
        
        try:
            doc = Document(file_path)
            full_text = []
            
            # Extraire tout le texte du document
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            
            # Nettoyer le texte
            cleaned_text = self._clean_text("\n".join(full_text), is_pdf=False)
            
            # Traiter le texte pour extraire les formations
            formations = self._parse_formations_text(cleaned_text)
            
            logger.info(f"Document chargé: {len(full_text)} paragraphes, {len(formations)} formations extraites")
            return formations
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du document Word: {str(e)}", exc_info=True)
            logger.info("Utilisation des données par défaut")
            return self._get_default_data()
    
    def _load_pdf(self, file_path: str) -> List[Dict]:
        """
        Charge et parse un fichier PDF (.pdf)
        
        Args:
            file_path: Chemin vers le fichier .pdf
        
        Returns:
            Liste de dictionnaires avec les universités extraites
        """
        if not PDF_AVAILABLE:
            logger.error("PyPDFLoader non disponible. Installation: pip install pypdf")
            return []
        
        try:
            # Utiliser PyPDFLoader de LangChain
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Extraire tout le texte
            full_text = "\n".join([doc.page_content for doc in documents])
            
            # Nettoyer le texte
            full_text = self._clean_text(full_text, is_pdf=True)
            
            # Traiter le texte pour extraire les informations sur les universités
            universities = self._parse_universities_text(full_text)
            
            logger.info(f"PDF chargé: {len(documents)} pages, {len(universities)} universités extraites")
            return universities
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du PDF: {str(e)}", exc_info=True)
            return []
    
    def _parse_universities_text(self, text: str) -> List[Dict]:
        """
        Parse le texte pour extraire les informations sur les universités
        
        Args:
            text: Texte complet du document PDF
        
        Returns:
            Liste de dictionnaires avec les universités
        """
        import re
        
        universities = []
        seen_universities = {}  # Pour éviter les doublons
        
        # Identifier les sections principales
        sections = self._identify_university_sections(text)
        
        # Traiter chaque section d'université
        for section in sections:
            if section["university_id"]:
                university_id = section["university_id"]
                
                # Si on a déjà vu cette université, ignorer les doublons
                # (le parser détecte parfois plusieurs fois la même section)
                if university_id in seen_universities:
                    # Vérifier si le nouveau contenu est significativement différent
                    existing_university = seen_universities[university_id]
                    existing_content_length = sum(len(chunk["content"]) for chunk in existing_university["chunks"])
                    new_content_length = len(section["content"])
                    
                    # Si le nouveau contenu est beaucoup plus long, remplacer par le nouveau contenu
                    if new_content_length > existing_content_length * 1.5:
                        existing_university["chunks"] = self._create_chunks_from_sub_sections(
                            section["content"], 
                            section["university_id"],
                            is_university=True
                        )
                        if existing_university["chunks"]:
                            existing_university["chunks"] = self._merge_short_chunks(existing_university["chunks"], min_length=100)
                    # Sinon, ignorer le doublon
                    continue
                else:
                    # Nouvelle université
                    university = {
                        "id": section["university_id"],
                        "name": section["university_name"],
                        "chunks": self._create_chunks_from_sub_sections(
                            section["content"],
                            section["university_id"],
                            is_university=True
                        )
                    }
                    
                    # Fusionner les chunks trop courts avec le précédent
                    if university["chunks"]:
                        university["chunks"] = self._merge_short_chunks(university["chunks"], min_length=100)
                        if university["chunks"]:
                            seen_universities[university_id] = university
        
        # Extraire les sections générales (universités privées, conseils, vie étudiante)
        general_sections = self._extract_general_sections(text)
        
        # Convertir le dictionnaire en liste et ajouter les sections générales
        universities = list(seen_universities.values())
        
        # Éliminer les doublons dans les sections générales avant de les ajouter
        seen_general_ids = set()
        for section in general_sections:
            section_id = section["id"]
            # Vérifier si on a déjà cette section
            if section_id not in seen_general_ids:
                seen_general_ids.add(section_id)
                universities.append(section)
            else:
                # Fusionner avec la section existante si nécessaire
                existing = next((u for u in universities if u["id"] == section_id), None)
                if existing:
                    # Comparer les longueurs pour garder la plus complète
                    existing_length = sum(len(c["content"]) for c in existing["chunks"])
                    new_length = sum(len(c["content"]) for c in section["chunks"])
                    if new_length > existing_length * 1.2:
                        # Remplacer par la nouvelle si elle est significativement plus longue
                        universities.remove(existing)
                        universities.append(section)
        
        # Éliminer les doublons de contenu entre différentes sections
        # (par exemple, si STUDENT_LIFE contient du contenu UCAD)
        universities = self._remove_duplicate_content_across_sections(universities)
        
        return universities
    
    def _remove_duplicate_content_across_sections(self, universities: List[Dict]) -> List[Dict]:
        """
        Élimine les chunks dupliqués entre différentes sections
        Par exemple, si STUDENT_LIFE contient du contenu qui devrait être dans UCAD
        
        Args:
            universities: Liste des universités/sections
        
        Returns:
            Liste nettoyée sans doublons inter-sections
        """
        if not universities or len(universities) <= 1:
            return universities
        
        # Créer un index des contenus par section importante (UCAD, etc.)
        important_sections = ["UCAD", "UGB", "UIDT", "UADB", "UASZ", "USSEIN", "UN_CHK", "UAM"]
        important_contents = {}
        important_keywords = {}  # Mots-clés spécifiques à chaque section
        
        for university in universities:
            if university["id"] in important_sections:
                for chunk in university["chunks"]:
                    content_normalized = chunk["content"].strip().lower()
                    # Prendre les premiers 200 caractères pour la signature
                    content_signature = content_normalized[:200] if len(content_normalized) > 200 else content_normalized
                    if content_signature not in important_contents:
                        important_contents[content_signature] = university["id"]
                    
                    # Extraire les mots-clés spécifiques (noms d'instituts, facultés, etc.)
                    if university["id"] == "UCAD":
                        # Mots-clés spécifiques à UCAD
                        ucad_keywords = [
                            "faculté de sciences et techniques", "fst", "clad", "cerer", 
                            "ipdsr", "ifee", "ips", "irempt", "ised", "ist", "ise", 
                            "ifan", "inseps", "iupa", "cesti", "esp", "ebad", "ensetp",
                            "fmpos", "flsh", "fsjp", "faseg", "fastef"
                        ]
                        for keyword in ucad_keywords:
                            if keyword in content_normalized:
                                if keyword not in important_keywords:
                                    important_keywords[keyword] = []
                                important_keywords[keyword].append(university["id"])
        
        # Nettoyer les sections générales pour enlever le contenu qui appartient aux sections importantes
        cleaned_universities = []
        for university in universities:
            if university["id"] in important_sections:
                # Garder les sections importantes telles quelles
                cleaned_universities.append(university)
            else:
                # Pour les sections générales, enlever les chunks qui sont dans les sections importantes
                cleaned_chunks = []
                for chunk in university["chunks"]:
                    content_normalized = chunk["content"].strip().lower()
                    content_signature = content_normalized[:200] if len(content_normalized) > 200 else content_normalized
                    
                    # Vérifier si c'est un doublon exact
                    is_duplicate = content_signature in important_contents
                    
                    # Vérifier si le chunk contient des mots-clés spécifiques à UCAD
                    contains_ucad_keywords = False
                    if university["id"] == "STUDENT_LIFE":
                        ucad_keywords = [
                            "faculté de sciences et techniques", "fst", "clad", "cerer", 
                            "ipdsr", "ifee", "ips", "irempt", "ised", "ist", "ise", 
                            "ifan", "inseps", "iupa", "douze instituts de l'ucad",
                            "centre de linguistique", "centre d'études et de recherches"
                        ]
                        for keyword in ucad_keywords:
                            if keyword in content_normalized:
                                contains_ucad_keywords = True
                                break
                    
                    # Garder le chunk seulement s'il n'est pas un doublon et ne contient pas de mots-clés UCAD
                    if not is_duplicate and not contains_ucad_keywords:
                        cleaned_chunks.append(chunk)
                
                if cleaned_chunks:
                    university["chunks"] = cleaned_chunks
                    cleaned_universities.append(university)
                elif university["id"] not in ["STUDENT_LIFE", "ORIENTATION_ADVICE"]:
                    # Garder les sections même si vides (sauf STUDENT_LIFE et ORIENTATION_ADVICE)
                    cleaned_universities.append(university)
        
        return cleaned_universities
    
    def _identify_university_sections(self, text: str) -> List[Dict]:
        """
        Identifie les sections principales du document (une par université)
        
        Returns:
            Liste de dictionnaires avec university_id, university_name et content
        """
        import re
        
        sections = []
        lines = text.split("\n")
        
        # Patterns pour identifier les universités publiques
        university_patterns = {
            "UCAD": {
                "patterns": [
                    r"Université Cheikh Anta Diop",
                    r"UCAD",
                    r"Cheikh Anta Diop de Dakar"
                ],
                "name": "Université Cheikh Anta Diop de Dakar (UCAD)"
            },
            "UGB": {
                "patterns": [
                    r"Université Gaston Berger",
                    r"UGB",
                    r"Gaston Berger de Saint-Louis"
                ],
                "name": "Université Gaston Berger de Saint-Louis (UGB)"
            },
            "UIDT": {
                "patterns": [
                    r"Université Iba Der Thiam",
                    r"UIDT",
                    r"Université de Thiès"
                ],
                "name": "Université Iba Der Thiam de Thiès (UIDT)"
            },
            "UADB": {
                "patterns": [
                    r"Université Alioune Diop de Bambey",
                    r"UADB",
                    r"Alioune Diop"
                ],
                "name": "Université Alioune Diop de Bambey (UADB)"
            },
            "UASZ": {
                "patterns": [
                    r"Université Assane Seck de Ziguinchor",
                    r"UASZ",
                    r"Assane Seck"
                ],
                "name": "Université Assane Seck de Ziguinchor (UASZ)"
            },
            "USSEIN": {
                "patterns": [
                    r"Université Sine Saloum",
                    r"USSEIN",
                    r"El Hadj Ibrahima Niasse"
                ],
                "name": "Université Sine Saloum El Hadj Ibrahima Niasse (USSEIN)"
            },
            "UN_CHK": {
                "patterns": [
                    r"Université Numérique Cheikh Hamidou Kane",
                    r"UN-CHK",
                    r"Université Virtuelle du Sénégal",
                    r"UVS"
                ],
                "name": "Université Numérique Cheikh Hamidou Kane (UN-CHK)"
            },
            "UAM": {
                "patterns": [
                    r"Université Amadou Mahtar Mbow",
                    r"UAM",
                    r"Amadou Mahtar Mbow de Diamniadio"
                ],
                "name": "Université Amadou Mahtar Mbow de Diamniadio (UAM)"
            }
        }
        
        current_section = None
        current_content = []
        seen_universities = set()
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_content.append("")
                continue
            
            # Vérifier si c'est le début d'une nouvelle université
            university_found = None
            university_name = None
            
            for university_id, info in university_patterns.items():
                if university_id in seen_universities and current_section == university_id:
                    continue
                
                for pattern in info["patterns"]:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        # Vérifier que c'est bien un titre de section
                        is_title = (
                            line_stripped.isupper() or 
                            "Université" in line_stripped or
                            i < 10  # Les premières lignes sont souvent des titres
                        )
                        
                        if is_title:
                            # Sauvegarder la section précédente
                            if current_section and current_section in seen_universities:
                                sections.append({
                                    "university_id": current_section,
                                    "university_name": university_name or current_section,
                                    "content": "\n".join(current_content)
                                })
                            
                            # Nouvelle section
                            university_found = university_id
                            university_name = info["name"]
                            if university_id not in seen_universities:
                                seen_universities.add(university_id)
                                current_section = university_id
                                current_content = [line_stripped]
                            else:
                                university_found = None
                            break
                
                if university_found:
                    break
            
            if university_found:
                continue
            
            # Ajouter à la section courante
            if current_section:
                current_content.append(line_stripped)
        
        # Sauvegarder la dernière section
        if current_section and current_section in seen_universities:
            sections.append({
                "university_id": current_section,
                "university_name": university_name or current_section,
                "content": "\n".join(current_content)
            })
        
        return sections
    
    def _extract_general_sections(self, text: str) -> List[Dict]:
        """
        Extrait les sections générales du guide (universités privées, conseils, vie étudiante)
        
        Returns:
            Liste de dictionnaires avec les sections générales
        """
        import re
        
        sections = []
        lines = text.split("\n")
        
        # Patterns pour identifier les sections générales
        general_section_patterns = [
            {
                "id": "PRIVATE_UNIVERSITIES",
                "name": "Universités Privées du Sénégal",
                "patterns": [
                    r"^Les Universités Privées",
                    r"^Universités Privées du Sénégal",
                    r"Université Amadou Hampaté Bâ",
                    r"Université Dakar-Bourguiba",
                    r"Université Catholique de l'Afrique de l'Ouest",
                    r"UCAO"
                ]
            },
            {
                "id": "OTHER_INSTITUTIONS",
                "name": "Autres Établissements d'Enseignement Supérieur",
                "patterns": [
                    r"^Autres Établissements",
                    r"Autres Établissements d'Enseignement",
                    r"École Polytechnique de Thiès",
                    r"instituts publics",
                    r"établissements spécialisés"
                ]
            },
            {
                "id": "ORIENTATION_ADVICE",
                "name": "Conseils d'orientation et Choix d'Établissement",
                "patterns": [
                    r"^Conseils d'orientation",
                    r"Conseils d'orientation et Choix",
                    r"^Choix d'Établissement",
                    r"Pour les étudiants souhaitant étudier la médecine",
                    r"Pour les étudiants souhaitant étudier l'agriculture",
                    r"Pour les sciences et technologies",
                    r"Les critères de choix"
                ]
            },
            {
                "id": "STUDENT_LIFE",
                "name": "Vie Étudiante et Recommandations Pratiques",
                "patterns": [
                    r"^Vie Étudiante",
                    r"^Recommandations Pratiques",
                    r"quartier Point E",
                    r"quartier Ouakam",
                    r"quartier Sacré-Cœur",
                    r"logement",
                    r"étudiants internationaux"
                ]
            }
        ]
        
        current_section = None
        current_content = []
        seen_sections = set()
        section_info_map = {}
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_content.append("")
                continue
            
            # Vérifier si c'est le début d'une nouvelle section générale
            section_found = None
            section_info = None
            
            for section_def in general_section_patterns:
                if section_def["id"] in seen_sections and current_section == section_def["id"]:
                    continue
                
                for pattern in section_def["patterns"]:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        # Sauvegarder la section précédente
                        if current_section and current_section in seen_sections:
                            sections.append({
                                "id": current_section,
                                "name": section_info_map.get(current_section, current_section),
                                "content": "\n".join(current_content)
                            })
                        
                        # Nouvelle section
                        section_found = section_def["id"]
                        section_info = section_def
                        section_info_map[section_def["id"]] = section_def["name"]
                        if section_def["id"] not in seen_sections:
                            seen_sections.add(section_def["id"])
                            current_section = section_def["id"]
                            current_content = [line_stripped]
                        else:
                            section_found = None
                        break
                
                if section_found:
                    break
            
            if section_found:
                continue
            
            # Ajouter à la section courante
            if current_section:
                current_content.append(line_stripped)
        
        # Sauvegarder la dernière section
        if current_section and current_section in seen_sections:
            sections.append({
                "id": current_section,
                "name": section_info_map.get(current_section, current_section),
                "content": "\n".join(current_content)
            })
        
        # Convertir en format université avec chunks et éliminer les doublons
        result = []
        seen_section_ids = set()
        
        for section in sections:
            section_id = section["id"]
            
            # Éviter les doublons
            if section_id in seen_section_ids:
                continue
            
            seen_section_ids.add(section_id)
            
            university = {
                "id": section_id,
                "name": section["name"],
                "chunks": self._create_chunks_from_sub_sections(
                    section["content"],
                    section_id,
                    is_university=True
                )
            }
            
            # Fusionner les chunks trop courts avec le précédent
            if university["chunks"]:
                university["chunks"] = self._merge_short_chunks(university["chunks"], min_length=100)
                # Nettoyer le texte de chaque chunk pour corriger les problèmes de formatage
                for chunk in university["chunks"]:
                    chunk["content"] = self._clean_text(chunk["content"], is_pdf=True)
                # Éliminer les doublons de contenu dans cette section
                university["chunks"] = self._remove_duplicate_chunks(university["chunks"])
                if university["chunks"]:
                    result.append(university)
        
        return result
    
    def _extract_university_sub_sections(self, content: str, university_id: str) -> List[Dict]:
        """
        Extrait les sous-sections d'une université (description, facultés, programmes, etc.)
        Détecte les structures avec ou sans articles (La Faculté, L'École, Faculté, etc.)
        
        Args:
            content: Contenu à analyser
            university_id: ID de l'université (pour déterminer si c'est une vraie université)
        
        Returns:
            Liste de dictionnaires avec type et content
        """
        import re
        
        # Déterminer si c'est une vraie université (pas une section générale)
        is_real_university = university_id not in [
            "STUDENT_LIFE", "ORIENTATION_ADVICE", "PRIVATE_UNIVERSITIES", 
            "OTHER_INSTITUTIONS", "OVERVIEW", "ADMISSION"
        ]
        
        sub_sections = []
        
        # Patterns améliorés pour détecter les structures avec articles
        # Ordre de priorité : les patterns plus spécifiques en premier
        section_patterns = [
            (r"^Vue d'ensemble|Vue d'ensemble du système", "overview"),
            (r"^Reconnaissance des diplômes|formalités d'entrée", "admission"),
            # Détecter les structures avec articles (La/L'/Le/Les Faculté/École/Institut)
            # Mais seulement si c'est une vraie université
            (r"^(La|L'|Le|Les)\s+(Faculté|École|Institut|UFR|Pôle)", "structure" if is_real_university else "description"),
            # Détecter les structures sans articles en début de ligne
            (r"^(Faculté|UFR|École|Institut|Pôle)\s+", "structure" if is_real_university else "description"),
            # Détecter les structures au milieu d'une phrase (pour les cas où elles commencent une nouvelle section)
            (r"\.\s+(La|L'|Le|Les)\s+(Faculté|École|Institut|UFR|Pôle)", "structure" if is_real_university else "description"),
            (r"^Programme|^Formation|^Spécialité", "programs"),
            (r"^Conseils|^Choix|^Critères|^Recommandations", "advice"),
            (r"^Vie Étudiante|^Vie étudiante|^Recommandations Pratiques", "student_life")
        ]
        
        lines = content.split("\n")
        current_type = "description"
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_content:
                    current_content.append("")
                continue
            
            # Vérifier si c'est le début d'une nouvelle sous-section
            section_found = None
            for pattern, section_type in section_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    # Sauvegarder la sous-section précédente
                    if current_content:
                        content_text = "\n".join(current_content).strip()
                        if len(content_text) > 50:
                            sub_sections.append({
                                "type": current_type,
                                "content": content_text
                            })
                    
                    # Nouvelle sous-section
                    section_found = section_type
                    current_type = section_type
                    current_content = [line_stripped]
                    break
            
            if section_found:
                continue
            
            # Ajouter à la sous-section courante
            current_content.append(line_stripped)
        
        # Sauvegarder la dernière sous-section
        if current_content:
            content_text = "\n".join(current_content).strip()
            if len(content_text) > 50:
                sub_sections.append({
                    "type": current_type,
                    "content": content_text
                })
        
        # Si on n'a trouvé qu'une seule section "description" et qu'elle est très longue,
        # essayer de la découper par structures (Faculté, École, Institut)
        # Mais SEULEMENT si c'est une vraie université (pas une section générale)
        if is_real_university and len(sub_sections) == 1 and sub_sections[0]["type"] == "description":
            long_content = sub_sections[0]["content"]
            if len(long_content) > 2000:
                # Vérifier qu'il y a au moins 2 structures distinctes
                structures_found = re.findall(
                    r"(La|L'|Le|Les)\s+(Faculté|École|Institut|UFR|Pôle)\s+[^\.]+",
                    long_content,
                    re.IGNORECASE
                )
                # Découper seulement s'il y a au moins 2 structures distinctes
                if len(structures_found) >= 2:
                    structure_chunks = self._split_by_structures(long_content)
                    if len(structure_chunks) > 1:
                        sub_sections = structure_chunks
        
        return sub_sections
    
    def _split_by_structures(self, content: str) -> List[Dict]:
        """
        Découpe un contenu long par structures (Faculté, École, Institut)
        Chaque structure est découpée individuellement pour un meilleur découpage
        
        Args:
            content: Contenu à découper
        
        Returns:
            Liste de sous-sections découpées par structure
        """
        import re
        
        sub_sections = []
        
        # Pattern pour détecter le début d'une nouvelle structure
        # Chercher "La Faculté", "L'École", "Les Instituts", etc.
        # Pattern amélioré pour mieux capturer chaque structure individuelle
        structure_pattern = r"(\.\s+)?(La|L'|Le|Les)\s+(Faculté|École|Institut|UFR|Pôle)\s+[^\.]+?(?=\.\s+(?:La|L'|Le|Les)\s+(?:Faculté|École|Institut|UFR|Pôle)|\.\s+L'|\.\s+Le|\.\s+Les|$)"
        
        # Trouver toutes les positions des structures
        matches = list(re.finditer(
            r"(\.\s+)?(La|L'|Le|Les)\s+(Faculté|École|Institut|UFR|Pôle)\s+",
            content,
            re.IGNORECASE
        ))
        
        if len(matches) < 2:
            # Pas assez de structures pour découper
            return [{"type": "description", "content": content}]
        
        # Découper le contenu en sections basées sur les structures
        start_pos = 0
        
        for i, match in enumerate(matches):
            # Contenu avant cette structure (sauf pour la première)
            if i > 0 and match.start() > start_pos:
                before_text = content[start_pos:match.start()].strip()
                # Nettoyer le texte
                before_text = re.sub(r'\s+', ' ', before_text)
                before_text = re.sub(r'\s+([,\.])', r'\1', before_text)
                if len(before_text) > 100:
                    sub_sections.append({
                        "type": "description" if i == 1 else "structure",
                        "content": before_text
                    })
            
            # Trouver la fin de cette section (début de la prochaine structure ou fin du texte)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            # Contenu de cette structure (incluant le nom de la structure)
            structure_text = content[match.start():end_pos].strip()
            
            # Nettoyer les espaces multiples et les problèmes de formatage
            structure_text = re.sub(r'\s+', ' ', structure_text)  # Espaces multiples
            structure_text = re.sub(r'\s+([,\.])', r'\1', structure_text)  # Espace avant ponctuation
            structure_text = re.sub(r'([,\.])\s*([,\.])', r'\1 \2', structure_text)  # Ponctuation double
            
            if len(structure_text) > 100:
                sub_sections.append({
                    "type": "structure",
                    "content": structure_text
                })
            
            start_pos = end_pos
        
        # Contenu après la dernière structure (s'il y en a)
        if start_pos < len(content):
            after_text = content[start_pos:].strip()
            if len(after_text) > 100:
                # Nettoyer le texte final aussi
                after_text = re.sub(r'\s+', ' ', after_text)
                after_text = re.sub(r'\s+([,\.])', r'\1', after_text)
                sub_sections.append({
                    "type": "structure",
                    "content": after_text
                })
        
        # Si le découpage n'a pas fonctionné, retourner le contenu original
        if not sub_sections or len(sub_sections) == 1:
            return [{"type": "description", "content": content}]
        
        return sub_sections
    
    def _parse_formations_text(self, text: str) -> List[Dict]:
        """
        Parse le texte pour extraire les informations sur les formations
        
        Args:
            text: Texte complet du document
        
        Returns:
            Liste de dictionnaires avec les formations
        """
        formations = []
        
        # D'abord, identifier les sections principales du document
        sections = self._identify_main_sections(text)
        
        # Traiter chaque section de formation
        for section in sections:
            if section["formation_id"]:
                # Extraire les sous-sections pour cette formation
                sub_sections = self._extract_sub_sections(section["content"], section["formation_id"])
                
                # Créer la formation avec ses chunks
                formation = {
                    "id": section["formation_id"],
                    "name": self._get_formation_name(section["formation_id"]),
                    "chunks": self._create_chunks_from_sub_sections(
                        section["content"],
                        section["formation_id"],
                        is_university=False
                    )
                }
                
                # Fusionner les chunks trop courts avec le précédent
                if formation["chunks"]:
                    formation["chunks"] = self._merge_short_chunks(formation["chunks"], min_length=100)
                    # Nettoyer le texte de chaque chunk pour corriger les problèmes de formatage
                    for chunk in formation["chunks"]:
                        chunk["content"] = self._clean_text(chunk["content"], is_pdf=False)
                    if formation["chunks"]:
                        formations.append(formation)
        
        # Si aucune formation n'a été trouvée, découper le texte en chunks génériques
        if not formations:
            logger.warning("Aucune formation spécifique détectée, découpage générique du texte")
            formations = self._split_text_generic(text)
        
        return formations
    
    def _identify_main_sections(self, text: str) -> List[Dict]:
        """
        Identifie les sections principales du document (une par formation)
        
        Returns:
            Liste de dictionnaires avec formation_id et content
        """
        import re
        
        sections = []
        lines = text.split("\n")
        
        formation_patterns = {
            "L1MPI": [
                r"^LICENCE MPI",
                r"^LICENCE.*MPI",
                r"MATHÉMATIQUES-PHYSIQUE-INFORMATIQUE",
                r"Mathématiques-Physique-Informatique"
            ],
            "L1BCGS": [
                r"^LICENCE BCGS",
                r"^LICENCE.*BCGS",
                r"BIOLOGIE-CHIMIE-GÉOSCIENCES",
                r"Biologie-Chimie-GéoSciences"
            ],
            "L1PCSM": [
                r"^LICENCE PCSM",
                r"^LICENCE.*PCSM",
                r"PHYSIQUE-CHIMIE-SCIENCES DE LA MATIÈRE",
                r"Physique-Chimie-Sciences de la Matière"
            ]
        }
        
        current_section = None
        current_content = []
        seen_formations = set()  # Pour éviter les doublons
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_section:
                    current_content.append("")
                continue
            
            # Vérifier si c'est le début d'une nouvelle formation
            # On cherche des patterns plus spécifiques pour éviter les faux positifs
            formation_found = None
            for formation_id, patterns in formation_patterns.items():
                # Ignorer si on a déjà vu cette formation (sauf si c'est vraiment une nouvelle section)
                if formation_id in seen_formations and current_section == formation_id:
                    continue
                
                for pattern in patterns:
                    # Vérifier que c'est bien un titre de section (en majuscules ou avec des mots-clés)
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        # Vérifier que ce n'est pas juste une mention dans le texte
                        # Les titres de sections sont généralement en majuscules ou suivis de "("
                        is_title = (
                            line_stripped.isupper() or 
                            line_stripped.startswith("LICENCE") or
                            "(" in line_stripped or
                            i < 5  # Les premières lignes sont souvent des titres
                        )
                        
                        if is_title:
                            # Sauvegarder la section précédente
                            if current_section and current_section in seen_formations:
                                sections.append({
                                    "formation_id": current_section,
                                    "content": "\n".join(current_content)
                                })
                            
                            # Nouvelle section
                            formation_found = formation_id
                            if formation_id not in seen_formations:
                                seen_formations.add(formation_id)
                                current_section = formation_id
                                current_content = [line_stripped]
                            else:
                                # Si on a déjà vu cette formation, on continue avec la section actuelle
                                formation_found = None
                            break
                
                if formation_found:
                    break
            
            if formation_found:
                continue
            
            # Ajouter à la section courante
            if current_section:
                current_content.append(line_stripped)
        
        # Sauvegarder la dernière section
        if current_section and current_section in seen_formations:
            sections.append({
                "formation_id": current_section,
                "content": "\n".join(current_content)
            })
        
        return sections
    
    def _extract_sub_sections(self, content: str, formation_id: str) -> List[Dict]:
        """
        Extrait les sous-sections d'une formation (présentation, admission, etc.)
        
        Returns:
            Liste de dictionnaires avec type et content
        """
        import re
        
        sub_sections = []
        
        # Patterns pour identifier les sous-sections
        section_patterns = [
            (r"Présentation\s+Générale", "presentation"),
            (r"Profil\s+du\s+Candidat", "admission"),
            (r"Conditions\s+d['']?admission", "admission"),
            (r"Structure\s+Pédagogique", "programme"),
            (r"Contenu\s+des\s+Enseignements", "programme"),
            (r"Poursuites\s+d['']?Études", "poursuites"),
            (r"Poursuites\s+d['']?études", "poursuites"),
            (r"Débouchés\s+Professionnels", "debouches"),
            (r"Débouchés\s+professionnels", "debouches"),
            (r"Taux\s+de\s+Réussite", "statistiques"),
            (r"Statistiques", "statistiques"),
        ]
        
        lines = content.split("\n")
        current_type = "presentation"
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_content:
                    current_content.append("")
                continue
            
            # Vérifier si c'est le début d'une nouvelle sous-section
            section_found = None
            for pattern, section_type in section_patterns:
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    # Sauvegarder la sous-section précédente
                    if current_content:
                        content_text = "\n".join(current_content).strip()
                        if len(content_text) > 50:
                            sub_sections.append({
                                "type": current_type,
                                "content": content_text
                            })
                    
                    # Nouvelle sous-section
                    section_found = section_type
                    current_type = section_type
                    current_content = [line_stripped]
                    break
            
            if section_found:
                continue
            
            # Ajouter à la sous-section courante
            current_content.append(line_stripped)
        
        # Sauvegarder la dernière sous-section
        if current_content:
            content_text = "\n".join(current_content).strip()
            if len(content_text) > 50:
                sub_sections.append({
                    "type": current_type,
                    "content": content_text
                })
        
        return sub_sections
    
    def _split_into_chunks(self, text: str, max_size: int = 1000) -> List[str]:
        """
        Découpe un texte en chunks de taille maximale
        Gère intelligemment les très longs textes en découpant par phrases et paragraphes
        
        Args:
            text: Texte à découper
            max_size: Taille maximale d'un chunk (recommandé: 1000-1500)
        
        Returns:
            Liste de chunks
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        
        # Pour les très longs textes (> 5000 chars), découper d'abord par paragraphes doubles
        if len(text) > 5000:
            paragraphs = text.split("\n\n")
            if len(paragraphs) > 1:
                current_chunk = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    # Si le paragraphe seul dépasse max_size, le découper par phrases
                    if len(para) > max_size:
                        # Sauvegarder le chunk actuel
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        
                        # Découper le paragraphe par phrases
                        para_chunks = self._split_long_paragraph(para, max_size)
                        chunks.extend(para_chunks)
                    else:
                        # Vérifier si on peut ajouter ce paragraphe
                        if len(current_chunk) + len(para) + 2 > max_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = para
                        else:
                            current_chunk += "\n\n" + para if current_chunk else para
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # Pour les textes moyens, utiliser la méthode standard
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Si le paragraphe seul dépasse max_size, le découper
            if len(para) > max_size:
                # Sauvegarder le chunk actuel
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Découper le long paragraphe par phrases
                para_chunks = self._split_long_paragraph(para, max_size)
                chunks.extend(para_chunks)
            else:
                # Vérifier si on peut ajouter ce paragraphe
                if len(current_chunk) + len(para) + 2 > max_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_long_paragraph(self, para: str, max_size: int) -> List[str]:
        """
        Découpe un long paragraphe en chunks par phrases
        Gère intelligemment les phrases pour éviter les chunks trop courts
        
        Args:
            para: Paragraphe à découper
            max_size: Taille maximale d'un chunk
        
        Returns:
            Liste de chunks
        """
        import re
        chunks = []
        
        # Découper par phrases (point suivi d'un espace ou saut de ligne)
        # Utiliser un pattern plus robuste qui capture le point et l'espace
        sentences = re.split(r'(\.\s+)', para)
        
        current_chunk = ""
        i = 0
        
        while i < len(sentences):
            sentence = sentences[i]
            # Si la phrase suivante existe (le séparateur), l'ajouter
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
                i += 2
            else:
                i += 1
            
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Si la phrase seule dépasse max_size, la découper par virgules ou points-virgules
            if len(sentence) > max_size:
                # Sauvegarder le chunk actuel
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Découper la phrase trop longue par sous-phrases
                sub_sentences = re.split(r'([,;]\s+)', sentence)
                for j in range(0, len(sub_sentences), 2):
                    sub_sentence = sub_sentences[j]
                    if j + 1 < len(sub_sentences):
                        sub_sentence += sub_sentences[j + 1]
                    
                    sub_sentence = sub_sentence.strip()
                    if not sub_sentence:
                        continue
                    
                    if len(current_chunk) + len(sub_sentence) + 2 > max_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sub_sentence
                    else:
                        current_chunk += " " + sub_sentence if current_chunk else sub_sentence
            else:
                # Vérifier si on peut ajouter cette phrase
                if len(current_chunk) + len(sentence) + 2 > max_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str, is_pdf: bool = False) -> str:
        """
        Nettoie le texte en supprimant les espaces multiples et sauts de ligne excessifs
        Corrige aussi les problèmes de formatage (espaces avant ponctuation, URLs, etc.)
        
        Args:
            text: Texte à nettoyer
            is_pdf: True si le texte vient d'un PDF (nécessite un nettoyage plus agressif)
        
        Returns:
            Texte nettoyé
        """
        import re
        
        if is_pdf:
            # Pour les PDFs : remplacer tous les sauts de ligne par un espace
            # (les PDFs ont souvent un saut de ligne par mot)
            text = re.sub(r'\n+', ' ', text)
            # Remplacer les espaces multiples par un seul espace
            text = re.sub(r' +', ' ', text)
            # Corriger les espaces avant la ponctuation (mais pas dans les URLs)
            text = re.sub(r'\s+([,\.;:!?])', r'\1', text)
            # Corriger les espaces après la ponctuation (sauf si suivi d'une majuscule ou dans une URL)
            text = re.sub(r'([,\.;:!?])([^\s])', r'\1 \2', text)
            # Restaurer les sauts de ligne après les points finaux
            text = re.sub(r'\. ', '.\n', text)
            # Nettoyer les espaces en début/fin de ligne
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
        else:
            # Pour les DOCX : remplacer les sauts de ligne multiples par un seul saut de ligne
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Nettoyer les espaces multiples dans chaque ligne
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                cleaned_line = re.sub(r' +', ' ', line.strip())
                # Corriger les espaces avant la ponctuation (mais pas dans les URLs)
                cleaned_line = re.sub(r'\s+([,\.;:!?])', r'\1', cleaned_line)
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
            text = '\n'.join(cleaned_lines)
        
        # Nettoyage final : corriger les espaces avant ponctuation restants
        text = re.sub(r'\s+([,\.;:!?])', r'\1', text)
        # Corriger les doubles points ou virgules (". ." -> ".")
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r',\s*,', ',', text)
        
        # Corriger les URLs mal formatées EN FIN (après tout le traitement)
        # Corriger "https: //" ou "https:// " ou "https: //www" etc.
        text = re.sub(r'https:\s*//', 'https://', text)
        text = re.sub(r'http:\s*//', 'http://', text)
        
        # Corriger les sauts de ligne et espaces dans les URLs (plus agressif)
        # Pattern pour détecter les URLs avec sauts de ligne ou espaces
        # Exemple: https://www.univ-thies.sn/ -> https://www.univ-thies.sn/
        url_pattern = r'(https?://[^\s\)]+?)(\s*\n\s*|\.\s*\n\s*)([^\s\)]+?)(?=\s|\)|,|\.|$)'
        def fix_url(match):
            url_start = match.group(1)
            separator = match.group(2)
            url_end = match.group(3)
            # Si url_end ressemble à une continuation d'URL (contient .sn, .com, etc.)
            if re.search(r'\.(sn|com|org|net|edu|fr|gov)', url_end):
                return url_start + url_end
            else:
                return match.group(0)
        
        text = re.sub(url_pattern, fix_url, text)
        
        # Corriger les URLs avec des points suivis de sauts de ligne
        # Exemple: https://www.univ-thies.sn/ -> https://www.univ-thies.sn/
        text = re.sub(r'(https?://[^\s]+?)\.\s*\n\s*([a-z0-9\-]+)', r'\1.\2', text, flags=re.IGNORECASE)
        
        # Corriger les espaces dans les URLs (mais pas les espaces après les URLs)
        text = re.sub(r'(https?://[^\s]+?)\s+([a-z0-9\-\.]+)', r'\1\2', text, flags=re.IGNORECASE)
        
        return text
    
    def _create_chunks_from_sub_sections(
        self, 
        content: str, 
        item_id: str, 
        is_university: bool = False
    ) -> List[Dict]:
        """
        Crée des chunks à partir des sous-sections d'une formation ou université
        
        Args:
            content: Contenu à traiter
            item_id: ID de la formation ou université
            is_university: True si c'est une université, False si c'est une formation
        
        Returns:
            Liste de chunks
        """
        if is_university:
            sub_sections = self._extract_university_sub_sections(content, item_id)
        else:
            sub_sections = self._extract_sub_sections(content, item_id)
        
        chunks = []
        for sub_section in sub_sections:
            split_chunks = self._split_into_chunks(sub_section["content"], max_size=1000)
            for chunk_text in split_chunks:
                cleaned_text = chunk_text.strip()
                if len(cleaned_text) > 50:
                    chunks.append({
                        "type": sub_section["type"],
                        "content": cleaned_text
                    })
        
        return chunks
    
    def _merge_short_chunks(self, chunks: List[Dict], min_length: int = 100) -> List[Dict]:
        """
        Fusionne les chunks trop courts avec le chunk précédent ou suivant
        Gère aussi les cas spéciaux comme les titres seuls
        
        Args:
            chunks: Liste de chunks à traiter
            min_length: Longueur minimale pour qu'un chunk soit considéré comme valide
        
        Returns:
            Liste de chunks avec les chunks courts fusionnés
        """
        if not chunks:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            current_length = len(chunk["content"])
            current_content = chunk["content"].strip()
            
            # Vérifier si c'est juste un titre (tout en majuscules ou très court)
            is_title_only = (
                current_length < 100 and 
                (current_content.isupper() or 
                 len(current_content.split()) <= 5 or
                 # Vérifier si c'est un titre de section
                 any(keyword in current_content.upper() for keyword in [
                     "LICENCE", "PRÉSENTATION", "PROFIL", "CONDITIONS", 
                     "STRUCTURE", "CONTENU", "POURSUITES", "DÉBOUCHÉS", "TAUX"
                 ]))
            )
            
            # Si le chunk est trop court (ou entre 100-200 chars pour optimisation)
            # Fusionner les chunks entre 100-200 chars s'ils sont adjacents à des chunks courts
            should_merge = (
                current_length < min_length or 
                is_title_only or
                (current_length < 200 and merged_chunks and len(merged_chunks[-1]["content"]) < 300)
            )
            
            if should_merge:
                # Essayer de fusionner avec le chunk précédent
                if merged_chunks:
                    previous_chunk = merged_chunks[-1]
                    merged_type = previous_chunk["type"] if len(previous_chunk["content"]) >= current_length else chunk["type"]
                    separator = "\n" if is_title_only else " "
                    merged_content = previous_chunk["content"] + separator + current_content
                    merged_chunks[-1] = {
                        "type": merged_type,
                        "content": merged_content.strip()
                    }
                # Sinon, essayer de fusionner avec le chunk suivant (si c'est le premier chunk)
                elif i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    merged_type = chunk["type"] if current_length >= len(next_chunk["content"]) else next_chunk["type"]
                    separator = "\n" if is_title_only else " "
                    merged_content = current_content + separator + next_chunk["content"]
                    merged_chunks.append({
                        "type": merged_type,
                        "content": merged_content.strip()
                    })
                    i += 1  # Skip le chunk suivant car il est fusionné
                else:
                    # Pas de chunk précédent ni suivant, garder tel quel
                    merged_chunks.append(chunk)
            else:
                # Ajouter le chunk tel quel
                merged_chunks.append(chunk)
            
            i += 1
        
        return merged_chunks
    
    def _remove_duplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Élimine les chunks dupliqués ou très similaires dans une liste
        
        Args:
            chunks: Liste de chunks à nettoyer
        
        Returns:
            Liste de chunks sans doublons
        """
        if not chunks or len(chunks) <= 1:
            return chunks
        
        unique_chunks = []
        seen_contents = set()
        
        for chunk in chunks:
            content_normalized = chunk["content"].strip().lower()
            # Prendre les premiers 200 caractères pour détecter les doublons
            content_signature = content_normalized[:200] if len(content_normalized) > 200 else content_normalized
            
            # Vérifier si ce contenu a déjà été vu
            if content_signature not in seen_contents:
                seen_contents.add(content_signature)
                unique_chunks.append(chunk)
            # Si c'est un doublon mais que le chunk actuel est plus long, remplacer
            else:
                # Trouver le chunk existant avec ce contenu
                for i, existing_chunk in enumerate(unique_chunks):
                    existing_normalized = existing_chunk["content"].strip().lower()
                    existing_signature = existing_normalized[:200] if len(existing_normalized) > 200 else existing_normalized
                    if existing_signature == content_signature:
                        # Remplacer si le nouveau est plus long et complet
                        if len(chunk["content"]) > len(existing_chunk["content"]) * 1.1:
                            unique_chunks[i] = chunk
                        break
        
        return unique_chunks
    
    def _split_text_generic(self, text: str, chunk_size: int = 500) -> List[Dict]:
        """
        Découpe le texte en chunks génériques si aucune structure n'est détectée
        
        Args:
            text: Texte à découper
            chunk_size: Taille approximative des chunks
        
        Returns:
            Liste de chunks
        """
        chunks = []
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    "type": "description",
                    "content": current_chunk.strip()
                })
                current_chunk = para
                chunk_idx += 1
            else:
                current_chunk += "\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append({
                "type": "description",
                "content": current_chunk.strip()
            })
        
        # Créer une formation générique avec tous les chunks
        return [{
            "id": "GENERAL",
            "name": "Informations générales sur les formations FST",
            "chunks": chunks
        }]
    
    def _get_formation_name(self, formation_id: str) -> str:
        """Retourne le nom complet d'une formation"""
        names = {
            "L1MPI": "Mathématiques, Physique, Informatique",
            "L1BCGS": "Biologie, Chimie, Géologie, Sciences",
            "L1PCSM": "Physique, Chimie, Sciences Mathématiques"
        }
        return names.get(formation_id, formation_id)
    
    def _get_default_data(self) -> List[Dict]:
        """
        Retourne les données par défaut si le document ne peut pas être chargé
        
        Returns:
            Liste de dictionnaires avec les formations par défaut
        """
        return [
            {
                "id": "L1MPI",
                "name": "Mathématiques, Physique, Informatique",
                "chunks": [
                    {
                        "type": "description",
                        "content": "La formation L1MPI (Mathématiques, Physique, Informatique) est une filière pluridisciplinaire qui combine les mathématiques, la physique et l'informatique. Elle s'adresse aux étudiants ayant une solide base en mathématiques et en sciences physiques."
                    },
                    {
                        "type": "prerequis",
                        "content": "Prérequis pour L1MPI: série S1 ou S2 du BAC, avec de bonnes notes en Mathématiques (MATH) et Sciences Physiques (SCPH). Les mathématiques sont essentielles pour cette formation."
                    },
                    {
                        "type": "debouches",
                        "content": "Débouchés L1MPI: ingénierie informatique, mathématiques appliquées, physique théorique, recherche, enseignement, data science, intelligence artificielle."
                    },
                    {
                        "type": "programme",
                        "content": "Programme L1MPI: algèbre, analyse, probabilités, mécanique, électricité, programmation, algorithmique, structures de données, bases de données."
                    }
                ]
            },
            {
                "id": "L1BCGS",
                "name": "Biologie, Chimie, Géologie, Sciences",
                "chunks": [
                    {
                        "type": "description",
                        "content": "La formation L1BCGS (Biologie, Chimie, Géologie, Sciences) est orientée vers les sciences de la vie et de la terre. Elle convient aux étudiants intéressés par la biologie, la chimie et les sciences naturelles."
                    },
                    {
                        "type": "prerequis",
                        "content": "Prérequis pour L1BCGS: série S1, S2 ou S3 du BAC, avec de bonnes notes en Sciences de la Vie et de la Terre (SVT) et Sciences Physiques (SCPH). La biologie est importante."
                    },
                    {
                        "type": "debouches",
                        "content": "Débouchés L1BCGS: médecine, pharmacie, biologie, chimie, géologie, environnement, recherche en sciences de la vie, enseignement."
                    },
                    {
                        "type": "programme",
                        "content": "Programme L1BCGS: biologie cellulaire, génétique, biochimie, chimie organique, géologie, écologie, physiologie, microbiologie."
                    }
                ]
            },
            {
                "id": "L1PCSM",
                "name": "Physique, Chimie, Sciences Mathématiques",
                "chunks": [
                    {
                        "type": "description",
                        "content": "La formation L1PCSM (Physique, Chimie, Sciences Mathématiques) est une filière équilibrée entre physique, chimie et mathématiques. Elle prépare aux études d'ingénierie et de sciences."
                    },
                    {
                        "type": "prerequis",
                        "content": "Prérequis pour L1PCSM: série S1 ou S2 du BAC, avec de bonnes notes en Mathématiques (MATH), Sciences Physiques (SCPH) et Chimie. Un équilibre entre ces matières est important."
                    },
                    {
                        "type": "debouches",
                        "content": "Débouchés L1PCSM: ingénierie chimique, physique appliquée, recherche, enseignement, industries chimiques et pharmaceutiques, énergie."
                    },
                    {
                        "type": "programme",
                        "content": "Programme L1PCSM: mécanique, thermodynamique, électromagnétisme, chimie générale, chimie organique, mathématiques appliquées, statistiques."
                    }
                ]
            }
        ]

