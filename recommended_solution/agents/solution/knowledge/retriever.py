import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from agents.solution.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_MODEL

class VectorKnowledgeRetriever:
    """
    Real semantic vector search retriever using Qdrant and SentenceTransformers.
    """
    def __init__(self):
        # Establish a lightweight connection to Qdrant Cloud
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = QDRANT_COLLECTION
        
        # Load the embedding model
        # This will download the model to cache on the first run, then load from cache
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, domain: str, component: str, failure_type: str, complaint_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Concatenate relevant fields into a single search query
        # This helps the semantic search find both contextual and specific matches
        query_text = f"{domain} {component} {failure_type}. {complaint_text}"
        
        # Convert the query into a 384-dimensional vector
        query_vector = self.model.encode(query_text).tolist()
        
        # Search Qdrant - Retrieve top 10 candidates
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=10
        ).points
        
        # Extract the payloads and their scores
        candidates = []
        for res in results:
            payload = dict(res.payload)
            payload["_score"] = res.score
            candidates.append(payload)
            
        print("\n[DEBUG] Original Qdrant candidates:")
        for idx, cand in enumerate(candidates):
            c_id = cand.get("id") or cand.get("knowledge_id")
            c_domain = cand.get("domain")
            c_score = cand.get("_score")
            print(f"  {idx+1}. {c_id} (domain: {c_domain}, score: {c_score:.4f})")
            
        # 1. Deduplicate by KB ID (highest score first)
        seen_ids = set()
        unique_candidates = []
        for cand in candidates:
            c_id = cand.get("id") or cand.get("knowledge_id")
            if not c_id:
                continue
            if c_id in seen_ids:
                continue
            seen_ids.add(c_id)
            unique_candidates.append(cand)
            
        # Helper to normalize equivalent domains
        def normalize_domain(domain_str: str) -> str:
            if not domain_str:
                return ""
            d_clean = str(domain_str).strip().lower()
            if d_clean in ("account", "account access", "portal_auth"):
                return "account"
            if d_clean in ("network_connectivity", "internet / connectivity", "internet performance", "infrastructure", "network_infrastructure"):
                return "network_connectivity"
            if d_clean in ("billing_payment", "billing / payment", "billing"):
                return "billing_payment"
            return d_clean
            
        mapped_domain = normalize_domain(domain)
        print(f"[DEBUG] Mapping input category/domain '{domain}' -> mapped target domain '{mapped_domain}'")
        print(f"[DEBUG] Query component: '{component}', failure_type: '{failure_type}'")
        
        # Helper to check case-insensitive match/contains or token-level overlap
        def matches_field(query_val, doc_val):
            if not query_val or not doc_val:
                return False
            q_clean = str(query_val).strip().lower()
            d_clean = str(doc_val).strip().lower()
            if q_clean == d_clean or q_clean in d_clean or d_clean in q_clean:
                return True
            q_words = set(q_clean.replace('_', ' ').replace('-', ' ').split())
            d_words = set(d_clean.replace('_', ' ').replace('-', ' ').split())
            common = q_words.intersection(d_words)
            common = {w for w in common if w not in ('and', 'or', 'the', 'for', 'to', 'in', 'of')}
            return len(common) > 0

        # Helper to provide semantic keyword boost based on complaint context
        def get_keyword_overlap_boost(cand, complaint_txt):
            if not complaint_txt:
                return 0.0
            
            txt_lower = str(complaint_txt).lower()
            cand_failure = str(cand.get("failure_type", "")).lower()
            cand_title = str(cand.get("title", "")).lower()
            cand_content = str(cand.get("content", "")).lower()
            
            boost = 0.0
            
            # Outage boost: if complaint mentions outage/down and candidate is outage-related
            if any(word in txt_lower for word in ["outage", "down", "offline", "no internet"]):
                if "outage" in cand_failure or "outage" in cand_title:
                    boost += 0.3
                    
            # Password/auth boost: if complaint mentions credential/login/reset and candidate is password/auth-related
            if any(word in txt_lower for word in ["password", "login", "reset", "credential", "auth"]):
                if any(k in cand_failure or k in cand_title for k in ["password", "login", "reset", "auth", "verification"]):
                    boost += 0.2
                    
            # Slow/speed boost: if complaint mentions slow/speed/latency and candidate is speed-related
            if any(word in txt_lower for word in ["slow", "speed", "lag", "latency", "buffering"]):
                if any(k in cand_failure or k in cand_title for k in ["slow", "speed", "latency", "performance"]):
                    boost += 0.2
                    
            # Billing boost: if complaint mentions charge/billing/refund and candidate is billing-related
            if any(word in txt_lower for word in ["charge", "bill", "refund", "pay", "fee"]):
                if any(k in cand_failure or k in cand_title for k in ["charge", "bill", "payment", "fee"]):
                    boost += 0.2
                    
            # Wifi/router connection boost: if complaint mentions router/wifi/disconnect/intermittent
            if any(word in txt_lower for word in ["router", "wifi", "wi-fi", "disconnect", "intermittent"]):
                if any(k in cand_failure or k in cand_title or k in cand_content for k in ["wifi", "wi-fi", "router", "disconnect", "intermittent"]):
                    boost += 0.1

            return boost
            
        # 2. Re-rank unique candidates using a composite score based on:
        # - normalized domain match (weight 0.5)
        # - component match (weight 0.4)
        # - failure type match (weight 0.6)
        # - semantic keyword overlap boost (up to 0.3)
        # - raw vector cosine similarity score (weight 1.0)
        for cand in unique_candidates:
            cand_domain = cand.get("domain")
            cand_component = cand.get("component")
            cand_failure_type = cand.get("failure_type")
            score = cand.get("_score", 0.0)
            
            is_domain = (mapped_domain != "") and (normalize_domain(cand_domain) == mapped_domain)
            is_component = matches_field(component, cand_component)
            is_failure = matches_field(failure_type, cand_failure_type)
            
            domain_weight = 0.5 if is_domain else 0.0
            component_weight = 0.4 if is_component else 0.0
            failure_weight = 0.6 if is_failure else 0.0
            keyword_boost = get_keyword_overlap_boost(cand, complaint_text)
            
            cand["_composite_score"] = score + domain_weight + component_weight + failure_weight + keyword_boost
            
        # Sort in descending order of composite score
        unique_candidates.sort(key=lambda x: x["_composite_score"], reverse=True)
        
        # Take the top_k results
        final_docs = unique_candidates[:top_k]
        
        print("[DEBUG] Re-ranked candidates output:")
        for idx, doc in enumerate(final_docs):
            c_id = doc.get("id") or doc.get("knowledge_id")
            c_domain = doc.get("domain")
            c_component = doc.get("component")
            c_failure = doc.get("failure_type")
            c_score = doc.get("_composite_score", 0.0)
            print(f"  {idx+1}. {c_id} (domain: {c_domain}, component: {c_component}, failure_type: {c_failure}, score: {c_score:.4f})")
            
        # Remove helper score metadata to keep returned schemas unchanged
        for doc in final_docs:
            doc.pop("_score", None)
            doc.pop("_composite_score", None)
            
        return final_docs

    def print_debug_info(self, complaint_text: str, top_k: int = 3):
        """
        Startup/debug method that prints collection details and retrieved KB articles.
        """
        print("\nQdrant collection:")
        print(self.collection_name)
        
        docs = self.search(
            domain="general",
            component="general",
            failure_type="general",
            complaint_text=complaint_text,
            top_k=top_k
        )
        
        print("\nRetrieved:")
        for idx, doc in enumerate(docs):
            kb_id = doc.get("knowledge_id") or doc.get("id") or doc.get("doc_id") or "UNKNOWN_ID"
            title = doc.get("title") or "No Title"
            print(f"{idx+1}. {kb_id} - {title}")
