"""
config_manager.py
Manages configuration files, variants (A/B), and OpenAI API setup.
Includes hot reloading, versioning, and .env integration.
"""

import json
import os
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
# loads .env from project root
load_dotenv()

# -------------------------------------------------------------------
# OpenAI API Key and Model Configuration
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import logger for structured logging
from loguru import logger

if not OPENAI_API_KEY:
    # Only warn here — some flows run in DRY_RUN and don't require the key.
    logger.warning("OPENAI_API_KEY not found in environment. If DRY_RUN=false you must set it in .env")

# Model names (fallback defaults provided)
GPT4_MODEL = os.getenv("GPT4_MODEL", os.getenv("OPENAI_MODEL", "gpt-4"))
GPT4_TURBO_MODEL = os.getenv("GPT4_TURBO_MODEL", "gpt-4-turbo")
GPT35_MODEL = os.getenv("GPT35_MODEL", "gpt-3.5-turbo")

# Dry-run flag: set DRY_RUN=true in .env to avoid real API calls
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

DEFAULT_MODEL = GPT4_MODEL  # Default model for system usage

# -------------------------------------------------------------------
# Directory Structure
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
KEYWORDS_DIR = os.path.join(BASE_DIR, "app", "ai", "intent_classification", "keywords")
VERSIONS_DIR = os.path.join(CONFIG_DIR, "versions")

os.makedirs(VERSIONS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Global Config Cache
# -------------------------------------------------------------------
CONFIG_CACHE = {}
ACTIVE_VARIANT = "A"  # Can be dynamically switched (A or B)

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def load_json_config(file_path):
    """Safely load a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        from loguru import logger
        logger.warning(f"Invalid JSON file skipped: {os.path.basename(file_path)} ({e})")
        return None
    except FileNotFoundError:
        return None


def validate_config(name: str, data: dict) -> bool:
    """Light validation to prevent bad configs from entering cache."""
    try:
        if name == "rules":
            # expect a rules.rule_sets dict and numeric weights if present
            rule_sets = data.get("rules", {}).get("rule_sets", {})
            if not isinstance(rule_sets, dict) or not rule_sets:
                from loguru import logger
                logger.warning("Invalid rules config: missing rules.rule_sets")
                return False
            for variant, cfg in rule_sets.items():
                if not isinstance(cfg, dict):
                    return False
                kw = cfg.get("kw_weight")
                emb = cfg.get("emb_weight")
                if kw is not None and emb is not None:
                    total = float(kw) + float(emb)
                    if abs(total - 1.0) > 1e-6:
                        from loguru import logger
                        logger.warning("Invalid weights: kw_weight + emb_weight must equal 1.0")
                        return False
        # allow other configs unvalidated for now
        return True
    except Exception as e:
        from loguru import logger
        logger.warning(f"Config validation error for {name}: {e}")
        return False


def save_version(file_path):
    """Backup configuration file when it changes."""
    try:
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_filename = f"{name}_{timestamp}{ext}"
        backup_path = os.path.join(VERSIONS_DIR, backup_filename)
        shutil.copy(file_path, backup_path)
        logger.info(f"💾 Backup saved: {backup_filename}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save backup for {file_path}: {e}")


def get_variant_filename(base_name, variant):
    """Return variant filename (e.g., config_A.json)."""
    name, ext = os.path.splitext(base_name)
    return f"{name}_{variant}{ext}"


# -------------------------------------------------------------------
# Config Loading Logic
# -------------------------------------------------------------------
def load_all_configs():
    """Load all configuration files for the current active variant."""
    global CONFIG_CACHE
    CONFIG_CACHE.clear()

    # Safe: if CONFIG_DIR doesn't exist, skip with a warning
    if not os.path.isdir(CONFIG_DIR):
        logger.warning(f"⚠️  Config directory not found: {CONFIG_DIR} (skipping file-based configs)")
    else:
        logger.info(f"🔄 Loading configuration variant: {ACTIVE_VARIANT}")

        # Load config files (e.g., rules.json, intents.json)
        for filename in os.listdir(CONFIG_DIR):
            if filename.endswith(".json") and "_" not in filename:
                variant_file = get_variant_filename(filename, ACTIVE_VARIANT)
                variant_path = os.path.join(CONFIG_DIR, variant_file)

                file_to_load = variant_path if os.path.exists(variant_path) else os.path.join(CONFIG_DIR, filename)
                data = load_json_config(file_to_load)
                if data:
                    CONFIG_CACHE[filename.replace(".json", "")] = data

    # Load keywords separately (optional)
    # Note: Variant loading message already logged above
    
    # Load rules and other config files from config/ directory
    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith(".json") and "_" not in filename:  # Base names only
            variant_file = get_variant_filename(filename, ACTIVE_VARIANT)
            variant_path = os.path.join(CONFIG_DIR, variant_file)

            # Fall back to base file if variant doesn't exist
            file_to_load = variant_path if os.path.exists(variant_path) else os.path.join(CONFIG_DIR, filename)

            data = load_json_config(file_to_load)
            if data and validate_config(filename.replace(".json", ""), data):
                CONFIG_CACHE[filename.replace(".json", "")] = data
    
    # Load keywords from the actual keywords directory
    if os.path.exists(KEYWORDS_DIR):
        keywords_data = {}
        for filename in os.listdir(KEYWORDS_DIR):
            if filename.endswith(".json") and filename != "loader.py":
                file_path = os.path.join(KEYWORDS_DIR, filename)
                data = load_json_config(file_path)
                if data:
                    # Merge keyword dictionaries (later files overwrite earlier keys)
                    keywords_data.update(data)
        if keywords_data:
            CONFIG_CACHE["keywords"] = keywords_data

    logger.info("✅ Configurations loaded successfully.")


# -------------------------------------------------------------------
# Hot Reload (Auto-refresh configs when modified)
# -------------------------------------------------------------------
class ConfigChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(".json"):
            return
        save_version(event.src_path)
        load_all_configs()
        logger.info(f"🔁 Config reloaded due to change: {event.src_path}")


def start_config_watcher():
    """Watch config directory for JSON changes."""
    # If CONFIG_DIR not found, return None
    if not os.path.isdir(CONFIG_DIR):
        logger.warning(f"👀 Config watcher not started: CONFIG_DIR does not exist ({CONFIG_DIR})")
        return None

    observer = Observer()
    handler = ConfigChangeHandler()
    observer.schedule(handler, CONFIG_DIR, recursive=False)
    observer.start()
    logger.info("👀 Watching config directory for changes...")
    return observer


# -------------------------------------------------------------------
# A/B Variant Management
# -------------------------------------------------------------------
def switch_variant(variant: str):
    """Switch configuration variant (A or B)."""
    global ACTIVE_VARIANT
    if variant.upper() not in ["A", "B"]:
        logger.error("❌ Invalid variant. Use 'A' or 'B'.")
        return

    ACTIVE_VARIANT = variant.upper()
    logger.info(f"🔁 Switched to configuration variant: {ACTIVE_VARIANT}")
    load_all_configs()


# -------------------------------------------------------------------
# Initialize on Import
# -------------------------------------------------------------------
# Attempt to load configs (safe if CONFIG_DIR missing)
load_all_configs()
