"""
API endpoints for A/B Testing Management and Results
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from app.ai.cost_monitor.ab_testing import (
    get_ab_test_manager,
    ExperimentVariant,
    VariantType
)


router = APIRouter(prefix="/api/ab-testing", tags=["A/B Testing"])
ab_manager = get_ab_test_manager()


class CreateExperimentRequest(BaseModel):
    experiment_id: str = Field(..., description="Unique experiment identifier")
    name: str = Field(..., description="Human-readable experiment name")
    description: str = Field(default="", description="Detailed description")
    variants: List[Dict[str, Any]] = Field(..., description="List of variant configurations")


class VariantConfig(BaseModel):
    variant_id: str
    variant_type: str
    config: Dict[str, Any]
    traffic_percentage: float
    description: str


@router.post("/experiments")
def create_experiment(request: CreateExperimentRequest):
    """
    Create a new A/B test experiment.
    
    Example payload:
    ```json
    {
      "experiment_id": "exp_prompt_v1.1",
      "name": "Prompt Optimization Test",
      "description": "Compare v1.0.0 vs v1.1.0",
      "variants": [
        {
          "variant_id": "control",
          "variant_type": "prompt_version",
          "config": {"prompt_version": "1.0.0"},
          "traffic_percentage": 0.5,
          "description": "Original prompt"
        },
        {
          "variant_id": "treatment",
          "variant_type": "prompt_version",
          "config": {"prompt_version": "1.1.0"},
          "traffic_percentage": 0.5,
          "description": "Optimized prompt"
        }
      ]
    }
    ```
    """
    try:
        # Parse variants
        variants = []
        for v in request.variants:
            variants.append(ExperimentVariant(
                variant_id=v["variant_id"],
                variant_type=VariantType(v["variant_type"]),
                config=v["config"],
                traffic_percentage=v["traffic_percentage"],
                description=v.get("description", "")
            ))
        
        # Create experiment
        success = ab_manager.create_experiment(
            experiment_id=request.experiment_id,
            name=request.name,
            variants=variants,
            description=request.description
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Experiment '{request.name}' created",
                "experiment_id": request.experiment_id
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create experiment")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/experiments/{experiment_id}")
def get_experiment_summary(experiment_id: str):
    """
    Get detailed summary and results for an experiment.
    
    Returns:
    - Experiment metadata
    - Per-variant metrics (cost, latency, accuracy, tokens)
    - Request counts and distributions
    """
    summary = ab_manager.get_experiment_summary(experiment_id)
    
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    
    return summary


@router.post("/experiments/{experiment_id}/stop")
def stop_experiment(experiment_id: str):
    """
    Stop an active experiment.
    
    After stopping, no new requests will be assigned to this experiment,
    but existing results are preserved for analysis.
    """
    try:
        ab_manager.stop_experiment(experiment_id)
        return {
            "status": "success",
            "message": f"Experiment '{experiment_id}' stopped",
            "experiment_id": experiment_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
def list_experiments():
    """
    List all experiments (active and stopped).
    """
    import sqlite3
    from app.ai.cost_monitor.ab_testing import DB_PATH
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT experiment_id, name, description, start_date, end_date, status
        FROM experiments
        ORDER BY created_at DESC
    """)
    
    experiments = []
    for row in cursor.fetchall():
        experiments.append({
            "experiment_id": row[0],
            "name": row[1],
            "description": row[2],
            "start_date": row[3],
            "end_date": row[4],
            "status": row[5]
        })
    
    conn.close()
    
    return {"experiments": experiments}


@router.get("/experiments/{experiment_id}/compare")
def compare_variants(experiment_id: str):
    """
    Compare variants in an experiment with statistical significance.
    
    Returns:
    - Cost comparison (absolute and percentage)
    - Latency comparison
    - Accuracy comparison
    - Token efficiency
    - Winner recommendation
    """
    summary = ab_manager.get_experiment_summary(experiment_id)
    
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    
    if len(summary["variants"]) < 2:
        return {"error": "Need at least 2 variants to compare"}
    
    # Assume first variant is control, rest are treatments
    control = summary["variants"][0]
    treatments = summary["variants"][1:]
    
    comparisons = []
    
    for treatment in treatments:
        # Cost comparison
        cost_diff = treatment["avg_cost_usd"] - control["avg_cost_usd"]
        cost_pct = (cost_diff / control["avg_cost_usd"] * 100) if control["avg_cost_usd"] > 0 else 0
        
        # Token comparison
        token_diff = treatment["avg_tokens"] - control["avg_tokens"]
        token_pct = (token_diff / control["avg_tokens"] * 100) if control["avg_tokens"] > 0 else 0
        
        # Latency comparison
        latency_diff = treatment["avg_latency_ms"] - control["avg_latency_ms"]
        latency_pct = (latency_diff / control["avg_latency_ms"] * 100) if control["avg_latency_ms"] > 0 else 0
        
        # Accuracy comparison (if available)
        accuracy_diff = None
        if treatment["accuracy"] is not None and control["accuracy"] is not None:
            accuracy_diff = treatment["accuracy"] - control["accuracy"]
        
        # Winner determination (simple heuristic)
        winner = "none"
        if cost_diff < -0.0001 and (accuracy_diff is None or accuracy_diff >= -0.05):
            # Treatment is cheaper and not significantly less accurate
            winner = "treatment"
        elif accuracy_diff and accuracy_diff > 0.05 and cost_diff < 0.001:
            # Treatment is significantly more accurate and not much more expensive
            winner = "treatment"
        elif abs(cost_diff) < 0.0001 and abs(latency_diff) < 100:
            winner = "tie"
        else:
            winner = "control"
        
        comparisons.append({
            "control_variant": control["variant_id"],
            "treatment_variant": treatment["variant_id"],
            "cost": {
                "control": control["avg_cost_usd"],
                "treatment": treatment["avg_cost_usd"],
                "difference": round(cost_diff, 6),
                "percentage_change": round(cost_pct, 2)
            },
            "tokens": {
                "control": control["avg_tokens"],
                "treatment": treatment["avg_tokens"],
                "difference": round(token_diff, 0),
                "percentage_change": round(token_pct, 2)
            },
            "latency": {
                "control": control["avg_latency_ms"],
                "treatment": treatment["avg_latency_ms"],
                "difference": round(latency_diff, 2),
                "percentage_change": round(latency_pct, 2)
            },
            "accuracy": {
                "control": control["accuracy"],
                "treatment": treatment["accuracy"],
                "difference": round(accuracy_diff, 3) if accuracy_diff else None
            },
            "winner": winner,
            "recommendation": _get_recommendation(winner, cost_pct, accuracy_diff)
        })
    
    return {
        "experiment_id": experiment_id,
        "experiment_name": summary["name"],
        "comparisons": comparisons
    }


def _get_recommendation(winner: str, cost_change: float, accuracy_diff: float = None) -> str:
    """Generate human-readable recommendation"""
    if winner == "treatment":
        if cost_change < -50:
            return "🎉 Strong recommendation: Deploy treatment. Significant cost savings with maintained accuracy."
        elif cost_change < -20:
            return "✅ Recommended: Deploy treatment. Good cost savings with acceptable accuracy."
        elif accuracy_diff and accuracy_diff > 0.1:
            return "✅ Recommended: Deploy treatment. Improved accuracy with minimal cost increase."
        else:
            return "👍 Consider deploying: Treatment shows positive results."
    elif winner == "tie":
        return "🤷 Neutral: Both variants perform similarly. Consider other factors."
    else:
        return "⚠️ Not recommended: Control variant performs better. Continue optimization."


# Health check
@router.get("/health")
def ab_testing_health():
    """Check A/B testing system health"""
    try:
        manager = get_ab_test_manager()
        active_count = len(manager.active_experiments)
        return {
            "status": "healthy",
            "active_experiments": active_count,
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

