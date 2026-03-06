"""
Système RAG pour les formations FST et le guide des universités
Gère deux collections ChromaDB séparées pour une recherche optimisée
Utilise le fichier JSON formations_chunks.json comme source de données
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging

from chatbot.json_loader import JSONLoader

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Système RAG qui gère deux collections :
    - Formations FST (MPI, BCGS, PCSM)
    - Guide des universités du Sénégal
    
    Les données sont chargées depuis formations_chunks.json
    """
    
    def __init__(
        self,
        persist_directory: str = "chatbot/data/chroma_db"
    ):
        """
        Initialise le système RAG avec deux collections
        
        Args:
            persist_directory: Dossier pour persister les données
        """
        self.persist_directory = persist_directory
        
        # Créer le dossier si nécessaire
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialiser ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialiser les deux collections
        self._init_formations_collection()
        self._init_universities_collection()
    
    def _init_formations_collection(self):
        """Initialise la collection des formations FST"""
        collection_name = "diores_formations"
        
        try:
            self.formations_collection = self.client.get_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' chargée")
        except:
            self.formations_collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Formations DIORES FST UCAD"}
            )
            logger.info(f"Collection '{collection_name}' créée")
            self._initialize_formations_collection()
    
    def _init_universities_collection(self):
        """Initialise la collection du guide des universités"""
        collection_name = "universities_guide"
        
        try:
            self.universities_collection = self.client.get_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' chargée")
        except:
            self.universities_collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Guide complet des universités du Sénégal"}
            )
            logger.info(f"Collection '{collection_name}' créée")
            self._initialize_universities_collection()
    
    def _initialize_formations_collection(self):
        """Initialise la collection avec les données des formations FST depuis le JSON"""
        logger.info("Initialisation de la collection des formations FST...")
        
        json_loader = JSONLoader()
        chunks = json_loader.get_fst_chunks()
        
        if not chunks:
            logger.warning("Aucune donnée de formation FST à charger")
            return
        
        # Préparer les données pour ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            chunk_id = chunk.get("id", f"chunk_{len(ids)}")
            text = chunk.get("text", "")
            metadata_chunk = chunk.get("metadata", {})
            
            if not text:
                logger.warning(f"Chunk {chunk_id} ignoré : texte vide")
                continue
            
            documents.append(text)
            metadatas.append({
                "source": "fst",
                "formation": metadata_chunk.get("formation", "unknown"),
                "section": metadata_chunk.get("section", "unknown"),
                "universite": metadata_chunk.get("universite", "UCAD_FST")
            })
            ids.append(chunk_id)
        
        # Ajouter les documents à la collection
        if documents:
            self.formations_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"{len(documents)} chunks de formations FST ajoutés à la collection")
    
    def _initialize_universities_collection(self):
        """Initialise la collection avec les données du guide des universités depuis le JSON"""
        logger.info("Initialisation de la collection du guide des universités...")
        
        json_loader = JSONLoader()
        chunks = json_loader.get_general_chunks()
        
        if not chunks:
            logger.warning("Aucune donnée d'université à charger")
            return
        
        # Préparer les données pour ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            chunk_id = chunk.get("id", f"chunk_{len(ids)}")
            text = chunk.get("text", "")
            metadata_chunk = chunk.get("metadata", {})
            
            if not text:
                logger.warning(f"Chunk {chunk_id} ignoré : texte vide")
                continue
            
            documents.append(text)
            metadatas.append({
                "source": "universities",
                "section": metadata_chunk.get("section", "unknown"),
                "universite": metadata_chunk.get("universite", "unknown")
            })
            ids.append(chunk_id)
        
        # Ajouter les documents à la collection
        if documents:
            self.universities_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"{len(documents)} chunks d'universités ajoutés à la collection")
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        source: Optional[str] = None,
        formation_filter: Optional[str] = None,
        university_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Effectue une recherche sémantique dans les collections
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner par collection
            source: "fst", "universities", ou None pour rechercher dans les deux
            formation_filter: Filtrer par formation spécifique (optionnel)
            university_filter: Filtrer par université spécifique (optionnel)
        
        Returns:
            Liste de dictionnaires avec les résultats (content, metadata, distance, source)
        """
        results = []
        
        # Recherche dans les formations FST
        if source is None or source == "fst":
            fst_results = self._search_formations(query, top_k, formation_filter)
            results.extend(fst_results)
        
        # Recherche dans le guide des universités
        if source is None or source == "universities":
            univ_results = self._search_universities(query, top_k, university_filter)
            results.extend(univ_results)
        
        # Trier par distance (pertinence) si disponible
        if results and results[0].get("distance") is not None:
            results.sort(key=lambda x: x.get("distance", float('inf')))
        
        # Limiter au top_k total si on a cherché dans les deux collections
        if source is None and len(results) > top_k:
            results = results[:top_k]
        
        return results
    
    def _search_formations(
        self,
        query: str,
        top_k: int,
        formation_filter: Optional[str] = None
    ) -> List[Dict]:
        """Recherche dans la collection des formations FST"""
        try:
            # Détecter la formation mentionnée dans la requête pour améliorer la pertinence
            query_lower = query.lower()
            detected_formation = None
            
            if "mpi" in query_lower or "mathématiques physique informatique" in query_lower:
                detected_formation = "L1MPI"
            elif "bcgs" in query_lower or "biologie chimie" in query_lower:
                detected_formation = "L1BCGS"
            elif "pcsm" in query_lower or "physique chimie sciences" in query_lower:
                detected_formation = "L1PCSM"
            
            # Utiliser le filtre détecté ou celui fourni
            where = {}
            formation_to_filter = formation_filter or detected_formation
            if formation_to_filter:
                where["formation"] = formation_to_filter
            
            # Augmenter le nombre de résultats pour avoir plus de choix
            n_results = top_k * 2 if not where else top_k
            
            results = self.formations_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )
            
            formatted_results = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": metadata,
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "source": "fst"
                    })
            
            # Si on a détecté une formation, prioriser les résultats de cette formation
            if detected_formation and not formation_filter:
                # Séparer les résultats par formation
                results_by_formation = {detected_formation: [], "other": []}
                for result in formatted_results:
                    result_formation = result.get("metadata", {}).get("formation", "")
                    if result_formation == detected_formation:
                        results_by_formation[detected_formation].append(result)
                    else:
                        results_by_formation["other"].append(result)
                
                # Réorganiser : d'abord les résultats de la formation détectée, puis les autres
                formatted_results = results_by_formation[detected_formation] + results_by_formation["other"]
                # Limiter au top_k
                formatted_results = formatted_results[:top_k]
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche dans les formations: {str(e)}", exc_info=True)
            return []
    
    def _search_universities(
        self,
        query: str,
        top_k: int,
        university_filter: Optional[str] = None
    ) -> List[Dict]:
        """Recherche dans la collection du guide des universités"""
        try:
            where = {}
            if university_filter:
                where["university"] = university_filter
            
            results = self.universities_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where if where else None
            )
            
            formatted_results = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": metadata,
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "source": "universities"
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche dans les universités: {str(e)}", exc_info=True)
            return []
    
    def detect_query_type(self, query: str) -> str:
        """
        Détecte le type de requête pour déterminer dans quelle collection chercher
        
        Args:
            query: Requête de l'utilisateur
        
        Returns:
            "fst", "universities", ou "both"
        """
        query_lower = query.lower()
        
        # Mots-clés pour les formations FST
        fst_keywords = [
            "mpi", "bcgs", "pcsm", "l1mpi", "l1bcgs", "l1pcsm",
            "mathématiques physique informatique",
            "biologie chimie géologie",
            "physique chimie sciences",
            "fst", "faculté des sciences et techniques",
            "formation", "licence", "admission", "prérequis",
            "débouchés", "programme", "mention"
        ]
        
        # Mots-clés pour les universités
        universities_keywords = [
            "ucad", "ugb", "uidt", "uadb", "uasz", "ussein", "uam",
            "université", "universités", "sénégal",
            "cheikh anta diop", "gaston berger", "iba der thiam",
            "alioune diop", "assane seck", "sine saloum",
            "amadou mahtar mbow", "virtuelle", "numérique",
            "saint-louis", "thiès", "ziguinchor", "bambey", "kaolack",
            "diamniadio", "faculté", "école", "institut", "ufr"
        ]
        
        fst_score = sum(1 for keyword in fst_keywords if keyword in query_lower)
        univ_score = sum(1 for keyword in universities_keywords if keyword in query_lower)
        
        if fst_score > 0 and univ_score == 0:
            return "fst"
        elif univ_score > 0 and fst_score == 0:
            return "universities"
        else:
            return "both"
