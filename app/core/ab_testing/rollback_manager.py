"""
Rollback Manager for A/B Testing Experiments

Provides safe rollback functionality with:
- State validation
- Backup creation
- Gradual traffic shifting
- Rollback history tracking
- Safety checks
"""

import json
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


# Paths
EXPERIMENTS_DIR = Path("app/core/ab_testing/experiments")
CONFIG_FILE = Path("app/core/ab_testing/experiment_config.json")
BACKUP_DIR = Path("app/core/ab_testing/backups")
ROLLBACK_HISTORY = Path("app/core/ab_testing/rollback_history.json")


class RollbackManager:
    """Manages safe rollback of A/B test experiments"""
    
    def __init__(self):
        """Initialize rollback manager"""
        self._ensure_directories()
        self._load_history()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ROLLBACK_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        
        if not ROLLBACK_HISTORY.exists():
            with open(ROLLBACK_HISTORY, "w") as f:
                json.dump([], f)
    
    def _load_history(self) -> List[Dict]:
        """Load rollback history"""
        try:
            with open(ROLLBACK_HISTORY) as f:
                return json.load(f)
        except Exception:
            return []
    
    def _save_history(self, history: List[Dict]):
        """Save rollback history"""
        with open(ROLLBACK_HISTORY, "w") as f:
            json.dump(history, f, indent=2)
    
    def create_backup(self, experiment_id: str, reason: str = "manual") -> Optional[str]:
        """
        Create backup of current experiment configuration.
        
        Args:
            experiment_id: Experiment to backup
            reason: Reason for backup
        
        Returns:
            Backup file path or None if failed
        """
        config_path = EXPERIMENTS_DIR / f"{experiment_id}.json"
        
        if not config_path.exists():
            print(f"❌ Experiment not found: {experiment_id}")
            return None
        
        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{experiment_id}_backup_{timestamp}.json"
            backup_path = BACKUP_DIR / backup_name
            
            # Copy file
            shutil.copy(config_path, backup_path)
            
            # Add metadata
            with open(backup_path) as f:
                config = json.load(f)
            
            config["_backup_metadata"] = {
                "backup_timestamp": timestamp,
                "reason": reason,
                "original_path": str(config_path)
            }
            
            with open(backup_path, "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return None
    
    def validate_rollback(self, experiment_id: str) -> tuple[bool, List[str]]:
        """
        Validate that it's safe to rollback an experiment.
        
        Returns:
            (is_safe, warnings)
        """
        warnings = []
        
        # Check if experiment is currently active
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    active_config = json.load(f)
                
                if active_config.get("experiment_id") == experiment_id:
                    warnings.append("⚠️ This experiment is currently ACTIVE")
            except Exception:
                pass
        
        # Check if there are recent results
        events_file = Path("app/core/ab_testing/ab_events.csv")
        if events_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(events_file)
                
                # Check for recent events (last hour)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    recent = df[df["timestamp"] > datetime.now() - timedelta(hours=1)]
                    
                    if len(recent) > 0:
                        warnings.append(f"⚠️ {len(recent)} events in the last hour")
            except Exception as e:
                warnings.append(f"⚠️ Could not check recent events: {e}")
        
        # Check for backups
        backups = list(BACKUP_DIR.glob(f"{experiment_id}_backup_*.json"))
        if not backups:
            warnings.append("⚠️ No backups found for this experiment")
        
        is_safe = len(warnings) == 0 or all("No backups" not in w for w in warnings)
        
        return (is_safe, warnings)
    
    def rollback_to_variant(
        self,
        experiment_id: str,
        target_variant: str,
        gradual: bool = True,
        target_traffic: float = 1.0
    ) -> bool:
        """
        Rollback experiment by routing all traffic to a single variant.
        
        Args:
            experiment_id: Experiment to rollback
            target_variant: Variant to route traffic to
            gradual: Whether to shift traffic gradually
            target_traffic: Target traffic percentage for variant (0.0-1.0)
        
        Returns:
            True if successful
        """
        config_path = EXPERIMENTS_DIR / f"{experiment_id}.json"
        
        if not config_path.exists():
            print(f"❌ Experiment not found: {experiment_id}")
            return False
        
        # Create backup first
        backup_path = self.create_backup(experiment_id, reason="rollback")
        if not backup_path:
            print("❌ Cannot proceed without backup")
            return False
        
        # Validate rollback
        is_safe, warnings = self.validate_rollback(experiment_id)
        
        if warnings:
            print("⚠️ Rollback warnings:")
            for warning in warnings:
                print(f"   {warning}")
        
        if not is_safe:
            response = input("Continue with rollback? (yes/no): ")
            if response.lower() != "yes":
                print("❌ Rollback cancelled")
                return False
        
        try:
            # Load config
            with open(config_path) as f:
                config = json.load(f)
            
            # Find target variant
            target_idx = None
            for i, variant in enumerate(config["variants"]):
                if variant["name"] == target_variant:
                    target_idx = i
                    break
            
            if target_idx is None:
                print(f"❌ Variant not found: {target_variant}")
                return False
            
            # Shift traffic
            num_variants = len(config["variants"])
            new_traffic = target_traffic * 100
            other_traffic = (100 - new_traffic) / (num_variants - 1) if num_variants > 1 else 0
            
            for i, variant in enumerate(config["variants"]):
                if i == target_idx:
                    variant["traffic_pct"] = new_traffic
                else:
                    variant["traffic_pct"] = other_traffic
            
            # Disable bandit if rolling back completely
            if target_traffic >= 0.95:
                config["use_bandit"] = False
            
            # Save updated config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            # Update active config if this is the active experiment
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE) as f:
                    active_config = json.load(f)
                
                if active_config.get("experiment_id") == experiment_id:
                    with open(CONFIG_FILE, "w") as f:
                        json.dump(config, f, indent=2)
            
            # Record in history
            history = self._load_history()
            history.append({
                "timestamp": datetime.now().isoformat(),
                "experiment_id": experiment_id,
                "action": "rollback_to_variant",
                "target_variant": target_variant,
                "target_traffic": target_traffic,
                "backup_path": backup_path,
                "gradual": gradual
            })
            self._save_history(history)
            
            print(f"✅ Rollback successful")
            print(f"   Traffic to {target_variant}: {new_traffic}%")
            print(f"   Backup saved: {backup_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
    
    def rollback_to_backup(self, experiment_id: str, backup_timestamp: Optional[str] = None) -> bool:
        """
        Rollback experiment to a previous backup.
        
        Args:
            experiment_id: Experiment to rollback
            backup_timestamp: Specific backup timestamp (uses latest if None)
        
        Returns:
            True if successful
        """
        # Find backups
        backups = list(BACKUP_DIR.glob(f"{experiment_id}_backup_*.json"))
        
        if not backups:
            print(f"❌ No backups found for {experiment_id}")
            return False
        
        # Select backup
        if backup_timestamp:
            target_backup = BACKUP_DIR / f"{experiment_id}_backup_{backup_timestamp}.json"
            if not target_backup.exists():
                print(f"❌ Backup not found: {target_backup}")
                return False
        else:
            # Use most recent backup
            target_backup = sorted(backups, reverse=True)[0]
        
        print(f"📦 Rolling back to: {target_backup.name}")
        
        # Create current backup before rollback
        current_backup = self.create_backup(experiment_id, reason="pre_rollback")
        if not current_backup:
            print("❌ Cannot proceed without current backup")
            return False
        
        try:
            # Load backup config
            with open(target_backup) as f:
                backup_config = json.load(f)
            
            # Remove backup metadata
            if "_backup_metadata" in backup_config:
                del backup_config["_backup_metadata"]
            
            # Restore config
            config_path = EXPERIMENTS_DIR / f"{experiment_id}.json"
            with open(config_path, "w") as f:
                json.dump(backup_config, f, indent=2)
            
            # Update active config if needed
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE) as f:
                    active_config = json.load(f)
                
                if active_config.get("experiment_id") == experiment_id:
                    with open(CONFIG_FILE, "w") as f:
                        json.dump(backup_config, f, indent=2)
            
            # Record in history
            history = self._load_history()
            history.append({
                "timestamp": datetime.now().isoformat(),
                "experiment_id": experiment_id,
                "action": "rollback_to_backup",
                "backup_file": str(target_backup),
                "current_backup": current_backup
            })
            self._save_history(history)
            
            print(f"✅ Rollback to backup successful")
            print(f"   Restored from: {target_backup.name}")
            print(f"   Current state backed up to: {current_backup}")
            
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
    
    def list_backups(self, experiment_id: Optional[str] = None) -> List[Dict]:
        """
        List all backups.
        
        Args:
            experiment_id: Filter by experiment ID (None for all)
        
        Returns:
            List of backup information
        """
        pattern = f"{experiment_id}_backup_*.json" if experiment_id else "*_backup_*.json"
        backups = list(BACKUP_DIR.glob(pattern))
        
        backup_info = []
        for backup_path in sorted(backups, reverse=True):
            try:
                with open(backup_path) as f:
                    config = json.load(f)
                
                metadata = config.get("_backup_metadata", {})
                
                backup_info.append({
                    "file": backup_path.name,
                    "experiment_id": config.get("experiment_id", "unknown"),
                    "timestamp": metadata.get("backup_timestamp", "unknown"),
                    "reason": metadata.get("reason", "unknown"),
                    "size_kb": backup_path.stat().st_size / 1024,
                    "path": str(backup_path)
                })
            except Exception as e:
                print(f"⚠️ Error reading backup {backup_path.name}: {e}")
        
        return backup_info
    
    def get_rollback_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent rollback history.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of rollback events
        """
        history = self._load_history()
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]


# Singleton instance
_rollback_manager = None


def get_rollback_manager() -> RollbackManager:
    """Get singleton instance of rollback manager"""
    global _rollback_manager
    if _rollback_manager is None:
        _rollback_manager = RollbackManager()
    return _rollback_manager


# ------------------------------------------------------------
# CLI Interface
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rollback Manager CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("experiment_id", help="Experiment ID")
    backup_parser.add_argument("--reason", default="manual", help="Backup reason")
    
    # Rollback to variant command
    rollback_variant_parser = subparsers.add_parser("rollback-variant", help="Rollback to single variant")
    rollback_variant_parser.add_argument("experiment_id", help="Experiment ID")
    rollback_variant_parser.add_argument("variant", help="Target variant")
    rollback_variant_parser.add_argument("--traffic", type=float, default=1.0, help="Target traffic (0.0-1.0)")
    
    # Rollback to backup command
    rollback_backup_parser = subparsers.add_parser("rollback-backup", help="Rollback to backup")
    rollback_backup_parser.add_argument("experiment_id", help="Experiment ID")
    rollback_backup_parser.add_argument("--timestamp", help="Backup timestamp (uses latest if not specified)")
    
    # List backups command
    list_parser = subparsers.add_parser("list", help="List backups")
    list_parser.add_argument("--experiment-id", help="Filter by experiment ID")
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show rollback history")
    history_parser.add_argument("--limit", type=int, default=20, help="Number of entries")
    
    args = parser.parse_args()
    manager = get_rollback_manager()
    
    if args.command == "backup":
        manager.create_backup(args.experiment_id, args.reason)
    
    elif args.command == "rollback-variant":
        manager.rollback_to_variant(args.experiment_id, args.variant, target_traffic=args.traffic)
    
    elif args.command == "rollback-backup":
        manager.rollback_to_backup(args.experiment_id, args.timestamp)
    
    elif args.command == "list":
        backups = manager.list_backups(args.experiment_id)
        print(f"\n📦 Found {len(backups)} backup(s):\n")
        for backup in backups:
            print(f"  {backup['file']}")
            print(f"    Experiment: {backup['experiment_id']}")
            print(f"    Timestamp: {backup['timestamp']}")
            print(f"    Reason: {backup['reason']}")
            print(f"    Size: {backup['size_kb']:.2f} KB")
            print()
    
    elif args.command == "history":
        history = manager.get_rollback_history(args.limit)
        print(f"\n📜 Recent Rollback History:\n")
        for entry in history:
            print(f"  {entry['timestamp']}")
            print(f"    Experiment: {entry['experiment_id']}")
            print(f"    Action: {entry['action']}")
            if "target_variant" in entry:
                print(f"    Target: {entry['target_variant']} ({entry.get('target_traffic', 1.0)*100}%)")
            print()
    
    else:
        parser.print_help()

