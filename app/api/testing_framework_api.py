"""
REST API Endpoints for Testing Framework (TASK-23)

Provides API access to:
- Experiment management
- Bandit state viewing/control
- Statistical analysis results
- Rollback operations
- Configuration management
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import json
from loguru import logger

# Import testing framework components
try:
    from app.core.ab_testing.bandit import BanditController
    from app.core.ab_testing.analysis import load_events, compute_variant_stats, omnibus_chi_square
    from app.core.ab_testing.rollback_manager import get_rollback_manager
    from app.core.ab_testing_integration import get_unified_ab_manager
    TESTING_AVAILABLE = True
except Exception as e:
    logger.warning(f"Testing framework not fully available: {e}")
    TESTING_AVAILABLE = False


router = APIRouter(prefix="/api/testing", tags=["Testing Framework"])


# Pydantic models
class CreateExperimentRequest(BaseModel):
    experiment_id: str = Field(..., description="Unique experiment identifier")
    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Detailed description")
    variants: List[Dict[str, Any]] = Field(..., description="Variant configurations")
    use_bandit: bool = Field(default=True, description="Enable bandit algorithm")


class RollbackRequest(BaseModel):
    target_variant: str = Field(..., description="Variant to route traffic to")
    traffic_percentage: float = Field(default=1.0, ge=0.0, le=1.0, description="Target traffic (0.0-1.0)")
    gradual: bool = Field(default=True, description="Gradual traffic shift")


# ------------------------------------------------------------
# Experiment Management
# ------------------------------------------------------------

@router.post("/experiments")
def create_experiment(request: CreateExperimentRequest):
    """
    Create a new A/B test experiment.
    
    Example:
    ```json
    {
      "experiment_id": "exp_model_test",
      "name": "Model Comparison",
      "description": "Compare GPT-4 vs GPT-4-mini",
      "use_bandit": true,
      "variants": [
        {
          "name": "gpt4",
          "traffic_pct": 50,
          "model": "gpt-4",
          "prompt_template": "Use GPT-4"
        },
        {
          "name": "gpt4_mini",
          "traffic_pct": 50,
          "model": "gpt-4o-mini",
          "prompt_template": "Use GPT-4-mini"
        }
      ]
    }
    ```
    """
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        # Create experiment config
        exp_dir = Path("app/core/ab_testing/experiments")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "experiment_id": request.experiment_id,
            "description": request.description,
            "use_bandit": request.use_bandit,
            "variants": request.variants,
            "logging": {
                "events_csv": "app/core/ab_testing/ab_events.csv",
                "report_md": f"app/core/ab_testing/reports/{request.experiment_id}_report.md"
            }
        }
        
        config_path = exp_dir / f"{request.experiment_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Experiment '{request.name}' created",
            "experiment_id": request.experiment_id,
            "config_path": str(config_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@router.get("/experiments")
def list_experiments():
    """List all experiment configurations"""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        exp_dir = Path("app/core/ab_testing/experiments")
        if not exp_dir.exists():
            return {"experiments": []}
        
        experiments = []
        for config_file in exp_dir.glob("*.json"):
            try:
                with open(config_file) as f:
                    config = json.load(f)
                
                experiments.append({
                    "experiment_id": config.get("experiment_id"),
                    "description": config.get("description"),
                    "variants_count": len(config.get("variants", [])),
                    "use_bandit": config.get("use_bandit", False),
                    "config_file": config_file.name
                })
            except Exception as e:
                logger.warning(f"Error reading {config_file}: {e}")
        
        return {"experiments": experiments, "count": len(experiments)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/results")
def get_experiment_results(experiment_id: str):
    """
    Get statistical analysis results for an experiment.
    
    Returns:
    - Per-variant statistics
    - Chi-square test results
    - Winner determination
    """
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        import pandas as pd
        
        # Load events
        df = load_events()
        
        if df.empty:
            return {
                "experiment_id": experiment_id,
                "status": "no_data",
                "message": "No events found"
            }
        
        # Filter by experiment
        exp_events = df[df.get("experiment_id", df.get("variant", "")) == experiment_id]
        
        if exp_events.empty:
            return {
                "experiment_id": experiment_id,
                "status": "no_data",
                "message": f"No events found for experiment {experiment_id}"
            }
        
        # Compute statistics
        stats = compute_variant_stats(exp_events)
        
        # Run chi-square test
        chi_square_result = omnibus_chi_square(exp_events)
        
        # Determine winner
        winner = None
        if not stats.empty:
            winner_row = stats.loc[stats["success_rate"].idxmax()]
            winner = {
                "variant": winner_row["variant"],
                "success_rate": float(winner_row["success_rate"]),
                "total_requests": int(winner_row["total_requests"])
            }
        
        return {
            "experiment_id": experiment_id,
            "status": "success",
            "variant_stats": stats.to_dict(orient="records"),
            "statistical_test": chi_square_result,
            "winner": winner,
            "total_events": len(exp_events)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ------------------------------------------------------------
# Bandit Controller
# ------------------------------------------------------------

@router.get("/bandit/state")
def get_bandit_state(experiment_id: str = Query(default="default", description="Experiment ID")):
    """Get current state of bandit controller"""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        state_path = Path("app/core/ab_testing/bandit_state.json")
        
        if not state_path.exists():
            return {
                "experiment_id": experiment_id,
                "status": "no_state",
                "message": "No bandit state found"
            }
        
        with open(state_path) as f:
            state = json.load(f)
        
        exp_state = state.get(experiment_id, {})
        
        return {
            "experiment_id": experiment_id,
            "state": exp_state,
            "variants": exp_state.get("variants", {}),
            "epsilon": exp_state.get("epsilon", 0.1)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bandit/reset")
def reset_bandit_state(experiment_id: str = Query(..., description="Experiment ID")):
    """Reset bandit state for an experiment"""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        state_path = Path("app/core/ab_testing/bandit_state.json")
        
        if not state_path.exists():
            return {"status": "no_state", "message": "No state to reset"}
        
        with open(state_path) as f:
            state = json.load(f)
        
        if experiment_id in state:
            del state[experiment_id]
            
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Bandit state reset for {experiment_id}"
            }
        else:
            return {
                "status": "not_found",
                "message": f"No state found for {experiment_id}"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Rollback Operations
# ------------------------------------------------------------

@router.post("/experiments/{experiment_id}/rollback")
def rollback_experiment(experiment_id: str, request: RollbackRequest):
    """
    Rollback experiment by routing traffic to a single variant.
    
    Safety checks are performed before rollback.
    """
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        rollback_mgr = get_rollback_manager()
        
        # Validate rollback
        is_safe, warnings = rollback_mgr.validate_rollback(experiment_id)
        
        # Perform rollback
        success = rollback_mgr.rollback_to_variant(
            experiment_id=experiment_id,
            target_variant=request.target_variant,
            gradual=request.gradual,
            target_traffic=request.traffic_percentage
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Rolled back to {request.target_variant}",
                "experiment_id": experiment_id,
                "target_variant": request.target_variant,
                "traffic_percentage": request.traffic_percentage * 100,
                "warnings": warnings
            }
        else:
            raise HTTPException(status_code=400, detail="Rollback failed")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/backups")
def list_backups(experiment_id: str):
    """List all backups for an experiment"""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        rollback_mgr = get_rollback_manager()
        backups = rollback_mgr.list_backups(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "backups": backups,
            "count": len(backups)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/backup")
def create_backup(experiment_id: str, reason: str = Query(default="manual", description="Backup reason")):
    """Create a backup of experiment configuration"""
    if not TESTING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Testing framework not available")
    
    try:
        rollback_mgr = get_rollback_manager()
        backup_path = rollback_mgr.create_backup(experiment_id, reason)
        
        if backup_path:
            return {
                "status": "success",
                "message": "Backup created",
                "experiment_id": experiment_id,
                "backup_path": backup_path
            }
        else:
            raise HTTPException(status_code=400, detail="Backup creation failed")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# System Status
# ------------------------------------------------------------

@router.get("/status")
def get_testing_status():
    """Get overall status of testing framework"""
    try:
        unified_mgr = get_unified_ab_manager()
        status = unified_mgr.get_system_status()
        
        # Add file system status
        exp_dir = Path("app/core/ab_testing/experiments")
        events_file = Path("app/core/ab_testing/ab_events.csv")
        
        status["filesystem"] = {
            "experiments_dir": str(exp_dir),
            "experiments_count": len(list(exp_dir.glob("*.json"))) if exp_dir.exists() else 0,
            "events_file": str(events_file),
            "events_exist": events_file.exists()
        }
        
        return {
            "status": "healthy" if TESTING_AVAILABLE else "degraded",
            "testing_framework_available": TESTING_AVAILABLE,
            "details": status
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/health")
def health_check():
    """Simple health check"""
    return {
        "status": "healthy" if TESTING_AVAILABLE else "unavailable",
        "service": "testing_framework_api"
    }

