"""
A/B Testing Framework for Cost vs Accuracy Trade-offs

Enables controlled experiments to compare different configurations:
- Prompt versions (e.g., v1.0.0 vs v1.1.0)
- Model selection (gpt-4o-mini vs gpt-3.5-turbo)
- Caching strategies
- Rate limiting thresholds
"""

import json
import os
import random
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List
from threading import Lock
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger


# Database for experiment tracking
DB_PATH = os.path.join("data", "ab_testing.db")
_db_lock = Lock()


class VariantType(Enum):
    """Types of variants for A/B testing"""
    PROMPT_VERSION = "prompt_version"
    MODEL_SELECTION = "model_selection"
    CACHING_STRATEGY = "caching_strategy"
    RATE_LIMIT = "rate_limit"
    HYBRID_CONFIG = "hybrid_config"


@dataclass
class ExperimentVariant:
    """Configuration for an A/B test variant"""
    variant_id: str
    variant_type: VariantType
    config: Dict[str, Any]
    traffic_percentage: float  # 0.0 to 1.0
    description: str


@dataclass
class ExperimentResult:
    """Result of a single request in an experiment"""
    experiment_id: str
    variant_id: str
    request_id: str
    timestamp: str
    user_input: str
    
    # Performance metrics
    latency_ms: float
    token_count: int
    cost_usd: float
    
    # Accuracy metrics
    predicted_action: str
    confidence: float
    is_correct: Optional[bool] = None  # Requires ground truth or feedback
    
    # Additional context
    metadata: Dict[str, Any] = None


class ABTestManager:
    """Manages A/B experiments for cost and accuracy optimization"""
    
    def __init__(self):
        self._ensure_db()
        self.active_experiments: Dict[str, List[ExperimentVariant]] = {}
        self._load_active_experiments()
    
    def _ensure_db(self):
        """Initialize SQLite database for experiment tracking"""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Variants table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS variants (
                    variant_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    variant_type TEXT,
                    config TEXT,
                    traffic_percentage REAL,
                    description TEXT,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            
            # Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    variant_id TEXT,
                    request_id TEXT,
                    timestamp TEXT,
                    user_input TEXT,
                    latency_ms REAL,
                    token_count INTEGER,
                    cost_usd REAL,
                    predicted_action TEXT,
                    confidence REAL,
                    is_correct INTEGER,
                    metadata TEXT,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
                    FOREIGN KEY(variant_id) REFERENCES variants(variant_id)
                )
            """)
            
            conn.commit()
            conn.close()
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        variants: List[ExperimentVariant],
        description: str = "",
        start_date: Optional[str] = None
    ) -> bool:
        """
        Create a new A/B test experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            name: Human-readable name
            variants: List of variant configurations
            description: Detailed description of the experiment
            start_date: ISO format start date (default: now)
        
        Returns:
            True if experiment created successfully
        """
        # Validate traffic percentages sum to 1.0
        total_traffic = sum(v.traffic_percentage for v in variants)
        if not (0.99 <= total_traffic <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Traffic percentages must sum to 1.0 (got {total_traffic})")
        
        start_date = start_date or datetime.utcnow().isoformat()
        
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            try:
                # Insert experiment
                cursor.execute("""
                    INSERT INTO experiments (experiment_id, name, description, start_date, status)
                    VALUES (?, ?, ?, ?, 'active')
                """, (experiment_id, name, description, start_date))
                
                # Insert variants
                for variant in variants:
                    cursor.execute("""
                        INSERT INTO variants (variant_id, experiment_id, variant_type, config, traffic_percentage, description)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        variant.variant_id,
                        experiment_id,
                        variant.variant_type.value,
                        json.dumps(variant.config),
                        variant.traffic_percentage,
                        variant.description
                    ))
                
                conn.commit()
                
                # Load into memory
                self.active_experiments[experiment_id] = variants
                
                logger.info(f"Experiment '{name}' created with {len(variants)} variants")
                return True
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to create experiment: {e}")
                return False
            finally:
                conn.close()
    
    def assign_variant(self, experiment_id: str, request_id: str) -> Optional[ExperimentVariant]:
        """
        Assign a variant to a request based on traffic split.
        
        Uses consistent hashing for stable assignment.
        
        Args:
            experiment_id: ID of the active experiment
            request_id: Unique request identifier
        
        Returns:
            Assigned variant or None if experiment not found
        """
        if experiment_id not in self.active_experiments:
            return None
        
        variants = self.active_experiments[experiment_id]
        
        # Use hash of request_id for consistent assignment
        hash_val = hash(request_id) % 10000 / 10000.0  # 0.0 to 1.0
        
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_percentage
            if hash_val < cumulative:
                return variant
        
        # Fallback to last variant (shouldn't happen if percentages sum to 1.0)
        return variants[-1]
    
    def record_result(self, result: ExperimentResult):
        """
        Record the result of an A/B test request.
        
        Args:
            result: ExperimentResult with performance and accuracy metrics
        """
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO results 
                    (experiment_id, variant_id, request_id, timestamp, user_input,
                     latency_ms, token_count, cost_usd, predicted_action, confidence,
                     is_correct, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.experiment_id,
                    result.variant_id,
                    result.request_id,
                    result.timestamp,
                    result.user_input,
                    result.latency_ms,
                    result.token_count,
                    result.cost_usd,
                    result.predicted_action,
                    result.confidence,
                    1 if result.is_correct else 0 if result.is_correct is not None else None,
                    json.dumps(result.metadata) if result.metadata else None
                ))
                
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to record result: {e}")
            finally:
                conn.close()
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for an experiment.
        
        Returns metrics for each variant:
        - Request count
        - Average latency
        - Average cost
        - Total cost
        - Average token count
        - Average confidence
        - Accuracy (if is_correct is available)
        """
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get experiment info
            cursor.execute("""
                SELECT name, description, start_date, status
                FROM experiments
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            exp_row = cursor.fetchone()
            if not exp_row:
                conn.close()
                return {"error": "Experiment not found"}
            
            exp_name, exp_desc, start_date, status = exp_row
            
            # Get variant summaries
            cursor.execute("""
                SELECT 
                    v.variant_id,
                    v.description,
                    v.traffic_percentage,
                    COUNT(r.id) as request_count,
                    AVG(r.latency_ms) as avg_latency,
                    AVG(r.cost_usd) as avg_cost,
                    SUM(r.cost_usd) as total_cost,
                    AVG(r.token_count) as avg_tokens,
                    AVG(r.confidence) as avg_confidence,
                    AVG(CASE WHEN r.is_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy
                FROM variants v
                LEFT JOIN results r ON v.variant_id = r.variant_id
                WHERE v.experiment_id = ?
                GROUP BY v.variant_id
            """, (experiment_id,))
            
            variants = []
            for row in cursor.fetchall():
                variants.append({
                    "variant_id": row[0],
                    "description": row[1],
                    "traffic_percentage": row[2],
                    "request_count": row[3] or 0,
                    "avg_latency_ms": round(row[4], 2) if row[4] else 0,
                    "avg_cost_usd": round(row[5], 6) if row[5] else 0,
                    "total_cost_usd": round(row[6], 4) if row[6] else 0,
                    "avg_tokens": round(row[7], 0) if row[7] else 0,
                    "avg_confidence": round(row[8], 3) if row[8] else 0,
                    "accuracy": round(row[9], 3) if row[9] else None
                })
            
            conn.close()
            
            return {
                "experiment_id": experiment_id,
                "name": exp_name,
                "description": exp_desc,
                "start_date": start_date,
                "status": status,
                "variants": variants
            }
    
    def stop_experiment(self, experiment_id: str):
        """Stop an active experiment"""
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE experiments
                SET status = 'stopped', end_date = ?
                WHERE experiment_id = ?
            """, (datetime.utcnow().isoformat(), experiment_id))
            
            conn.commit()
            conn.close()
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            logger.info(f"Experiment '{experiment_id}' stopped")
    
    def _load_active_experiments(self):
        """Load active experiments from database into memory"""
        with _db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT experiment_id
                FROM experiments
                WHERE status = 'active'
            """)
            
            for (exp_id,) in cursor.fetchall():
                cursor.execute("""
                    SELECT variant_id, variant_type, config, traffic_percentage, description
                    FROM variants
                    WHERE experiment_id = ?
                """, (exp_id,))
                
                variants = []
                for row in cursor.fetchall():
                    variants.append(ExperimentVariant(
                        variant_id=row[0],
                        variant_type=VariantType(row[1]),
                        config=json.loads(row[2]),
                        traffic_percentage=row[3],
                        description=row[4]
                    ))
                
                self.active_experiments[exp_id] = variants
            
            conn.close()


# Singleton instance
_ab_test_manager = None


def get_ab_test_manager() -> ABTestManager:
    """Get singleton instance of ABTestManager"""
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n🧪 A/B Testing Framework Demo\n")
    
    manager = get_ab_test_manager()
    
    # Create an experiment: v1.0.0 vs v1.1.0 prompts
    variants = [
        ExperimentVariant(
            variant_id="control_v1.0.0",
            variant_type=VariantType.PROMPT_VERSION,
            config={"prompt_version": "1.0.0"},
            traffic_percentage=0.5,
            description="Original verbose prompt"
        ),
        ExperimentVariant(
            variant_id="treatment_v1.1.0",
            variant_type=VariantType.PROMPT_VERSION,
            config={"prompt_version": "1.1.0"},
            traffic_percentage=0.5,
            description="Optimized token-efficient prompt"
        )
    ]
    
    manager.create_experiment(
        experiment_id="exp_prompt_optimization_001",
        name="Prompt Optimization v1.0.0 vs v1.1.0",
        variants=variants,
        description="Test token reduction impact on cost and accuracy"
    )
    
    # Simulate some requests
    for i in range(10):
        request_id = f"req_{i}"
        variant = manager.assign_variant("exp_prompt_optimization_001", request_id)
        
        # Simulate result (in production, these come from actual requests)
        result = ExperimentResult(
            experiment_id="exp_prompt_optimization_001",
            variant_id=variant.variant_id,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            user_input=f"test query {i}",
            latency_ms=random.uniform(500, 2000),
            token_count=random.randint(200, 900),
            cost_usd=random.uniform(0.0005, 0.004),
            predicted_action="SEARCH_PRODUCT",
            confidence=random.uniform(0.7, 0.95),
            is_correct=random.choice([True, True, True, False]),  # 75% accuracy
            metadata={"version": variant.config["prompt_version"]}
        )
        
        manager.record_result(result)
        print(f"✓ Request {request_id} → {variant.variant_id}")
    
    # Get summary
    print("\n📊 Experiment Summary:\n")
    summary = manager.get_experiment_summary("exp_prompt_optimization_001")
    
    print(f"Experiment: {summary['name']}")
    print(f"Status: {summary['status']}")
    print(f"Started: {summary['start_date']}\n")
    
    for variant in summary['variants']:
        print(f"Variant: {variant['variant_id']}")
        print(f"  Description: {variant['description']}")
        print(f"  Traffic: {variant['traffic_percentage']*100:.0f}%")
        print(f"  Requests: {variant['request_count']}")
        print(f"  Avg Latency: {variant['avg_latency_ms']}ms")
        print(f"  Avg Cost: ${variant['avg_cost_usd']:.6f}")
        print(f"  Total Cost: ${variant['total_cost_usd']:.4f}")
        print(f"  Avg Tokens: {variant['avg_tokens']:.0f}")
        print(f"  Avg Confidence: {variant['avg_confidence']:.2f}")
        print(f"  Accuracy: {variant['accuracy']*100:.1f}%\n")
    
    print("✅ A/B Testing Demo Complete\n")

