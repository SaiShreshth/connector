"""
Integration Layer for A/B Testing Systems

Bridges two complementary A/B testing systems:
1. Cost Monitoring A/B (TASK-22): app/ai/cost_monitor/ab_testing.py
   - Database-backed (SQLite)
   - API-driven
   - Focus: Cost optimization, budget tracking
   
2. Testing Framework (TASK-23): app/core/ab_testing/
   - File-based (CSV/JSONL)
   - CLI-driven
   - Focus: Model/prompt validation, accuracy testing

This integration layer provides:
- Unified experiment interface
- Cross-system metric sharing
- Experiment synchronization
- Consolidated reporting
"""

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from loguru import logger

# Import both systems
try:
    from app.ai.cost_monitor.ab_testing import get_ab_test_manager, ExperimentResult as CostExperimentResult
    COST_SYSTEM_AVAILABLE = True
except Exception as e:
    logger.warning(f"Cost monitoring A/B system not available: {e}")
    COST_SYSTEM_AVAILABLE = False

try:
    from app.core.ab_testing.ab_framework import run_llm_inference, choose_variant, log_event
    from app.core.ab_testing.bandit import BanditController
    from app.core.ab_testing.analysis import load_events, compute_variant_stats
    TESTING_SYSTEM_AVAILABLE = True
except Exception as e:
    logger.warning(f"Testing framework not available: {e}")
    TESTING_SYSTEM_AVAILABLE = False


class UnifiedABTestManager:
    """
    Unified interface for both A/B testing systems.
    
    Use this class to:
    - Create experiments that log to both systems
    - Track metrics consistently
    - Generate unified reports
    - Manage experiments across both frameworks
    """
    
    def __init__(self):
        """Initialize both A/B testing systems"""
        self.cost_manager = None
        self.bandit_controller = None
        
        if COST_SYSTEM_AVAILABLE:
            try:
                self.cost_manager = get_ab_test_manager()
                logger.info("Cost monitoring A/B system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize cost manager: {e}")
        
        if TESTING_SYSTEM_AVAILABLE:
            try:
                self.bandit_controller = BanditController(experiment_id="unified_experiment")
                logger.info("Testing framework initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize bandit controller: {e}")
        
        self.active_experiment_mapping = {}  # Map experiment IDs between systems
    
    def create_unified_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        experiment_type: str = "both"  # "cost", "testing", or "both"
    ) -> bool:
        """
        Create experiment in one or both systems.
        
        Args:
            experiment_id: Unique identifier
            name: Human-readable name
            description: Detailed description
            variants: List of variant configurations
            experiment_type: Which system(s) to use
        
        Returns:
            True if experiment created successfully
        """
        success = True
        
        # Create in cost monitoring system
        if experiment_type in ["cost", "both"] and self.cost_manager:
            try:
                from app.ai.cost_monitor.ab_testing import ExperimentVariant, VariantType
                
                cost_variants = []
                for v in variants:
                    cost_variants.append(ExperimentVariant(
                        variant_id=v["variant_id"],
                        variant_type=VariantType.HYBRID_CONFIG,
                        config=v.get("config", {}),
                        traffic_percentage=v.get("traffic_percentage", 1.0 / len(variants)),
                        description=v.get("description", "")
                    ))
                
                success &= self.cost_manager.create_experiment(
                    experiment_id=f"cost_{experiment_id}",
                    name=name,
                    variants=cost_variants,
                    description=description
                )
                
                logger.info(f"Experiment created in cost monitoring system: cost_{experiment_id}")
                
            except Exception as e:
                logger.error(f"Failed to create experiment in cost system: {e}")
                success = False
        
        # Create in testing framework
        if experiment_type in ["testing", "both"] and TESTING_SYSTEM_AVAILABLE:
            try:
                config_path = Path("app/core/ab_testing/experiments") / f"{experiment_id}.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                config = {
                    "experiment_id": experiment_id,
                    "description": description,
                    "use_bandit": True,
                    "variants": variants,
                    "logging": {
                        "events_csv": "app/core/ab_testing/ab_events.csv",
                        "report_md": f"app/core/ab_testing/reports/{experiment_id}_report.md"
                    }
                }
                
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"Experiment created in testing framework: {experiment_id}")
                
            except Exception as e:
                logger.error(f"Failed to create experiment in testing system: {e}")
                success = False
        
        # Store mapping
        self.active_experiment_mapping[experiment_id] = {
            "cost_id": f"cost_{experiment_id}",
            "testing_id": experiment_id,
            "type": experiment_type
        }
        
        return success
    
    def record_unified_result(
        self,
        experiment_id: str,
        variant_id: str,
        request_id: str,
        user_input: str,
        predicted_action: str,
        confidence: float,
        latency_ms: float,
        token_count: int,
        cost_usd: float,
        is_correct: Optional[bool] = None
    ):
        """
        Record result to both systems simultaneously.
        
        Args:
            experiment_id: Experiment identifier
            variant_id: Variant that was used
            request_id: Unique request ID
            user_input: User query
            predicted_action: Predicted intent/action
            confidence: Model confidence score
            latency_ms: Response time
            token_count: Total tokens used
            cost_usd: Cost in USD
            is_correct: Whether prediction was correct (optional)
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Record in cost monitoring system
        if self.cost_manager and experiment_id in self.active_experiment_mapping:
            try:
                result = CostExperimentResult(
                    experiment_id=self.active_experiment_mapping[experiment_id]["cost_id"],
                    variant_id=variant_id,
                    request_id=request_id,
                    timestamp=timestamp,
                    user_input=user_input,
                    latency_ms=latency_ms,
                    token_count=token_count,
                    cost_usd=cost_usd,
                    predicted_action=predicted_action,
                    confidence=confidence,
                    is_correct=is_correct,
                    metadata={"source": "unified_ab_test"}
                )
                
                self.cost_manager.record_result(result)
                
            except Exception as e:
                logger.warning(f"Failed to record result in cost system: {e}")
        
        # Record in testing framework
        if TESTING_SYSTEM_AVAILABLE:
            try:
                # Log to CSV/JSONL
                event = {
                    "timestamp": timestamp,
                    "experiment_id": experiment_id,
                    "variant": variant_id,
                    "request_id": request_id,
                    "user_input": user_input,
                    "predicted_action": predicted_action,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "tokens": token_count,
                    "cost_usd": cost_usd,
                    "result_label": "success" if is_correct else "failure" if is_correct is not None else "unknown"
                }
                
                # Append to JSONL
                jsonl_path = "app/core/ab_testing/ab_events.jsonl"
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(event) + "\n")
                
                # Append to CSV
                import csv
                csv_path = "app/core/ab_testing/ab_events.csv"
                file_exists = os.path.exists(csv_path)
                
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=event.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(event)
                
            except Exception as e:
                logger.warning(f"Failed to record result in testing system: {e}")
    
    def get_unified_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get consolidated summary from both systems.
        
        Returns:
            Combined metrics from both A/B testing systems
        """
        summary = {
            "experiment_id": experiment_id,
            "cost_monitoring": None,
            "testing_framework": None,
            "combined_metrics": {}
        }
        
        # Get cost monitoring summary
        if self.cost_manager and experiment_id in self.active_experiment_mapping:
            try:
                cost_id = self.active_experiment_mapping[experiment_id]["cost_id"]
                summary["cost_monitoring"] = self.cost_manager.get_experiment_summary(cost_id)
            except Exception as e:
                logger.warning(f"Failed to get cost monitoring summary: {e}")
        
        # Get testing framework summary
        if TESTING_SYSTEM_AVAILABLE:
            try:
                import pandas as pd
                events = load_events()
                
                if not events.empty:
                    exp_events = events[events.get("experiment_id", "") == experiment_id]
                    if not exp_events.empty:
                        stats = compute_variant_stats(exp_events)
                        summary["testing_framework"] = stats.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Failed to get testing framework summary: {e}")
        
        # Combine metrics
        if summary["cost_monitoring"] or summary["testing_framework"]:
            try:
                summary["combined_metrics"] = self._combine_metrics(
                    summary["cost_monitoring"],
                    summary["testing_framework"]
                )
            except Exception as e:
                logger.warning(f"Failed to combine metrics: {e}")
        
        return summary
    
    def _combine_metrics(self, cost_data: Optional[Dict], testing_data: Optional[List]) -> Dict:
        """Combine metrics from both systems into unified view"""
        combined = {
            "variants": {},
            "totals": {
                "requests": 0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "success_rate": 0.0
            }
        }
        
        # Process cost monitoring data
        if cost_data and "variants" in cost_data:
            for variant in cost_data["variants"]:
                vid = variant["variant_id"]
                combined["variants"][vid] = {
                    "requests": variant.get("request_count", 0),
                    "avg_cost": variant.get("avg_cost_usd", 0.0),
                    "total_cost": variant.get("total_cost_usd", 0.0),
                    "avg_latency": variant.get("avg_latency_ms", 0.0),
                    "avg_confidence": variant.get("avg_confidence", 0.0),
                    "accuracy": variant.get("accuracy", None)
                }
        
        # Enhance with testing framework data
        if testing_data:
            for variant in testing_data:
                vid = variant.get("variant")
                if vid in combined["variants"]:
                    combined["variants"][vid]["success_rate"] = variant.get("success_rate", 0.0)
                    combined["variants"][vid]["p50_latency"] = variant.get("p50_latency_ms", 0.0)
                    combined["variants"][vid]["p95_latency"] = variant.get("p95_latency_ms", 0.0)
                else:
                    combined["variants"][vid] = {
                        "requests": variant.get("total_requests", 0),
                        "success_rate": variant.get("success_rate", 0.0),
                        "p50_latency": variant.get("p50_latency_ms", 0.0),
                        "p95_latency": variant.get("p95_latency_ms", 0.0),
                        "total_cost": variant.get("total_cost_usd", 0.0)
                    }
        
        # Calculate totals
        for variant_data in combined["variants"].values():
            combined["totals"]["requests"] += variant_data.get("requests", 0)
            combined["totals"]["total_cost"] += variant_data.get("total_cost", 0.0)
        
        if combined["totals"]["requests"] > 0:
            total_latency = sum(v.get("avg_latency", 0) * v.get("requests", 0) 
                              for v in combined["variants"].values())
            combined["totals"]["avg_latency"] = total_latency / combined["totals"]["requests"]
        
        return combined
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of both A/B testing systems"""
        return {
            "cost_monitoring": {
                "available": COST_SYSTEM_AVAILABLE,
                "initialized": self.cost_manager is not None,
                "active_experiments": len(self.cost_manager.active_experiments) if self.cost_manager else 0
            },
            "testing_framework": {
                "available": TESTING_SYSTEM_AVAILABLE,
                "initialized": self.bandit_controller is not None,
                "experiment_configs": len(list(Path("app/core/ab_testing/experiments").glob("*.json"))) if Path("app/core/ab_testing/experiments").exists() else 0
            },
            "integration": {
                "unified_experiments": len(self.active_experiment_mapping)
            }
        }


# Singleton instance
_unified_manager = None


def get_unified_ab_manager() -> UnifiedABTestManager:
    """Get singleton instance of unified A/B test manager"""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedABTestManager()
    return _unified_manager


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n🧪 Unified A/B Testing Integration Demo\n")
    
    manager = get_unified_ab_manager()
    
    # Check system status
    status = manager.get_system_status()
    print("📊 System Status:")
    print(f"  Cost Monitoring: {'✅' if status['cost_monitoring']['available'] else '❌'}")
    print(f"  Testing Framework: {'✅' if status['testing_framework']['available'] else '❌'}")
    print(f"  Active Unified Experiments: {status['integration']['unified_experiments']}\n")
    
    # Create a unified experiment
    variants = [
        {
            "variant_id": "gpt4_mini",
            "model": "gpt-4o-mini",
            "traffic_percentage": 0.5,
            "description": "Fast and cost-efficient",
            "config": {"model": "gpt-4o-mini", "temperature": 0.2}
        },
        {
            "variant_id": "gpt4",
            "model": "gpt-4",
            "traffic_percentage": 0.5,
            "description": "High accuracy",
            "config": {"model": "gpt-4", "temperature": 0.2}
        }
    ]
    
    success = manager.create_unified_experiment(
        experiment_id="exp_unified_001",
        name="Unified Model Comparison Test",
        description="Compare GPT-4 vs GPT-4-mini across both systems",
        variants=variants,
        experiment_type="both"
    )
    
    if success:
        print("✅ Unified experiment created successfully!\n")
    else:
        print("⚠️ Experiment creation had some issues\n")
    
    print("✅ Integration demo complete\n")

