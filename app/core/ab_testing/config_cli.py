"""
Configuration CLI Tool for A/B Testing Experiments

Easy-to-use command-line interface for creating, validating, and managing
A/B test experiment configurations.

Usage:
    python -m app.core.ab_testing.config_cli create --name "My Test" --variants variant1,variant2
    python -m app.core.ab_testing.config_cli validate experiment_config.json
    python -m app.core.ab_testing.config_cli list
    python -m app.core.ab_testing.config_cli templates
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


# Paths
EXPERIMENTS_DIR = Path("app/core/ab_testing/experiments")
CONFIG_FILE = Path("app/core/ab_testing/experiment_config.json")
TEMPLATES_DIR = Path("app/core/ab_testing/templates")


def ensure_directories():
    """Create necessary directories if they don't exist"""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def get_experiment_templates() -> Dict[str, Dict]:
    """Return predefined experiment templates"""
    return {
        "model_comparison": {
            "description": "Compare different GPT models (GPT-4, GPT-4-mini, GPT-3.5)",
            "template": {
                "experiment_id": "exp_model_comparison",
                "description": "Compare model performance and cost",
                "use_bandit": True,
                "variants": [
                    {
                        "name": "gpt4",
                        "traffic_pct": 33,
                        "model": "gpt-4",
                        "prompt_template": "Use GPT-4 for high accuracy"
                    },
                    {
                        "name": "gpt4_mini",
                        "traffic_pct": 34,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Use GPT-4-mini for cost efficiency"
                    },
                    {
                        "name": "gpt35",
                        "traffic_pct": 33,
                        "model": "gpt-3.5-turbo",
                        "prompt_template": "Use GPT-3.5 as baseline"
                    }
                ]
            }
        },
        "prompt_optimization": {
            "description": "Compare different prompt versions",
            "template": {
                "experiment_id": "exp_prompt_optimization",
                "description": "Test optimized prompts for token reduction",
                "use_bandit": True,
                "variants": [
                    {
                        "name": "prompt_v1",
                        "traffic_pct": 50,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Original verbose prompt"
                    },
                    {
                        "name": "prompt_v2",
                        "traffic_pct": 50,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Optimized compact prompt"
                    }
                ]
            }
        },
        "traffic_split": {
            "description": "Simple 50/50 traffic split test",
            "template": {
                "experiment_id": "exp_traffic_split",
                "description": "A/B test with equal traffic distribution",
                "use_bandit": False,
                "variants": [
                    {
                        "name": "control",
                        "traffic_pct": 50,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Control variant"
                    },
                    {
                        "name": "treatment",
                        "traffic_pct": 50,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Treatment variant"
                    }
                ]
            }
        },
        "adaptive_bandit": {
            "description": "Adaptive traffic allocation with epsilon-greedy bandit",
            "template": {
                "experiment_id": "exp_adaptive_bandit",
                "description": "Use bandit algorithm for dynamic traffic allocation",
                "use_bandit": True,
                "bandit_config": {
                    "epsilon": 0.1,
                    "initial_traffic": "uniform"
                },
                "variants": [
                    {
                        "name": "variant_a",
                        "traffic_pct": 33,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Variant A"
                    },
                    {
                        "name": "variant_b",
                        "traffic_pct": 33,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Variant B"
                    },
                    {
                        "name": "variant_c",
                        "traffic_pct": 34,
                        "model": "gpt-4o-mini",
                        "prompt_template": "Variant C"
                    }
                ]
            }
        }
    }


def validate_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate experiment configuration.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Required fields
    required_fields = ["experiment_id", "description", "variants"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate variants
    if "variants" in config:
        if not isinstance(config["variants"], list):
            errors.append("'variants' must be a list")
        elif len(config["variants"]) < 2:
            errors.append("Must have at least 2 variants")
        else:
            total_traffic = 0
            for i, variant in enumerate(config["variants"]):
                # Check required variant fields
                if "name" not in variant:
                    errors.append(f"Variant {i}: missing 'name'")
                if "traffic_pct" not in variant:
                    errors.append(f"Variant {i}: missing 'traffic_pct'")
                else:
                    total_traffic += variant["traffic_pct"]
                if "model" not in variant:
                    errors.append(f"Variant {i}: missing 'model'")
            
            # Check traffic percentages sum to ~100
            if abs(total_traffic - 100) > 1:
                errors.append(f"Traffic percentages must sum to 100 (got {total_traffic})")
    
    # Validate experiment_id format
    if "experiment_id" in config:
        exp_id = config["experiment_id"]
        if not exp_id.startswith("exp_"):
            errors.append("experiment_id should start with 'exp_'")
        if " " in exp_id:
            errors.append("experiment_id cannot contain spaces")
    
    return (len(errors) == 0, errors)


def create_experiment(args):
    """Create a new experiment configuration"""
    ensure_directories()
    
    # Parse variants
    variant_names = [v.strip() for v in args.variants.split(",")]
    
    if len(variant_names) < 2:
        print("❌ Error: Must specify at least 2 variants")
        return 1
    
    # Calculate traffic split
    traffic_pct = 100.0 / len(variant_names)
    
    # Build config
    config = {
        "experiment_id": args.experiment_id or f"exp_{args.name.lower().replace(' ', '_')}",
        "description": args.description or args.name,
        "use_bandit": args.use_bandit,
        "variants": []
    }
    
    # Add variants
    model = args.model or "gpt-4o-mini"
    for name in variant_names:
        config["variants"].append({
            "name": name,
            "traffic_pct": round(traffic_pct, 2),
            "model": model,
            "prompt_template": f"Variant {name} template"
        })
    
    # Add logging config
    config["logging"] = {
        "events_csv": "app/core/ab_testing/ab_events.csv",
        "report_md": f"app/core/ab_testing/reports/{config['experiment_id']}_report.md"
    }
    
    # Validate
    is_valid, errors = validate_config(config)
    if not is_valid:
        print("❌ Validation errors:")
        for error in errors:
            print(f"   - {error}")
        return 1
    
    # Save
    output_path = EXPERIMENTS_DIR / f"{config['experiment_id']}.json"
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Experiment created: {output_path}")
    print(f"   Experiment ID: {config['experiment_id']}")
    print(f"   Variants: {len(config['variants'])}")
    print(f"   Traffic split: {traffic_pct:.1f}% each")
    
    if args.activate:
        activate_experiment(config['experiment_id'])
    
    return 0


def validate_experiment(args):
    """Validate an experiment configuration file"""
    config_path = Path(args.config_file)
    
    if not config_path.exists():
        print(f"❌ Error: File not found: {config_path}")
        return 1
    
    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return 1
    
    is_valid, errors = validate_config(config)
    
    if is_valid:
        print(f"✅ Configuration is valid: {config_path}")
        print(f"   Experiment ID: {config.get('experiment_id')}")
        print(f"   Variants: {len(config.get('variants', []))}")
        return 0
    else:
        print(f"❌ Validation failed: {config_path}")
        for error in errors:
            print(f"   - {error}")
        return 1


def list_experiments(args):
    """List all experiment configurations"""
    ensure_directories()
    
    experiments = list(EXPERIMENTS_DIR.glob("*.json"))
    
    if not experiments:
        print("No experiments found")
        return 0
    
    print(f"\n📋 Found {len(experiments)} experiment(s):\n")
    
    for exp_path in sorted(experiments):
        try:
            with open(exp_path) as f:
                config = json.load(f)
            
            exp_id = config.get("experiment_id", "unknown")
            desc = config.get("description", "No description")
            num_variants = len(config.get("variants", []))
            use_bandit = config.get("use_bandit", False)
            
            # Check if this is the active experiment
            is_active = ""
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE) as f:
                    active_config = json.load(f)
                if active_config.get("experiment_id") == exp_id:
                    is_active = " [ACTIVE]"
            
            print(f"  {exp_id}{is_active}")
            print(f"    File: {exp_path.name}")
            print(f"    Description: {desc}")
            print(f"    Variants: {num_variants}")
            print(f"    Bandit: {'Yes' if use_bandit else 'No'}")
            print()
            
        except Exception as e:
            print(f"  ⚠️ Error reading {exp_path.name}: {e}\n")
    
    return 0


def show_templates(args):
    """Show available experiment templates"""
    templates = get_experiment_templates()
    
    print(f"\n📝 Available Templates:\n")
    
    for template_name, template_info in templates.items():
        print(f"  {template_name}")
        print(f"    Description: {template_info['description']}")
        print(f"    Variants: {len(template_info['template']['variants'])}")
        print(f"    Use: python -m app.core.ab_testing.config_cli from-template {template_name}")
        print()
    
    return 0


def from_template(args):
    """Create experiment from template"""
    ensure_directories()
    
    templates = get_experiment_templates()
    
    if args.template_name not in templates:
        print(f"❌ Error: Template '{args.template_name}' not found")
        print(f"   Available templates: {', '.join(templates.keys())}")
        return 1
    
    template = templates[args.template_name]["template"].copy()
    
    # Customize with user inputs
    if args.experiment_id:
        template["experiment_id"] = args.experiment_id
    if args.description:
        template["description"] = args.description
    
    # Add logging
    template["logging"] = {
        "events_csv": "app/core/ab_testing/ab_events.csv",
        "report_md": f"app/core/ab_testing/reports/{template['experiment_id']}_report.md"
    }
    
    # Save
    output_path = EXPERIMENTS_DIR / f"{template['experiment_id']}.json"
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)
    
    print(f"✅ Experiment created from template: {output_path}")
    print(f"   Template: {args.template_name}")
    print(f"   Experiment ID: {template['experiment_id']}")
    
    if args.activate:
        activate_experiment(template['experiment_id'])
    
    return 0


def activate_experiment(experiment_id: str) -> bool:
    """Activate an experiment by copying it to experiment_config.json"""
    exp_path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    
    if not exp_path.exists():
        print(f"❌ Error: Experiment not found: {experiment_id}")
        return False
    
    try:
        with open(exp_path) as f:
            config = json.load(f)
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Experiment activated: {experiment_id}")
        print(f"   Config file: {CONFIG_FILE}")
        return True
        
    except Exception as e:
        print(f"❌ Error activating experiment: {e}")
        return False


def activate_command(args):
    """Activate an experiment"""
    return 0 if activate_experiment(args.experiment_id) else 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="A/B Testing Configuration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create new experiment")
    create_parser.add_argument("--name", required=True, help="Experiment name")
    create_parser.add_argument("--variants", required=True, help="Comma-separated variant names")
    create_parser.add_argument("--experiment-id", help="Experiment ID (auto-generated if not provided)")
    create_parser.add_argument("--description", help="Experiment description")
    create_parser.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    create_parser.add_argument("--use-bandit", action="store_true", help="Enable bandit algorithm")
    create_parser.add_argument("--activate", action="store_true", help="Activate immediately after creation")
    create_parser.set_defaults(func=create_experiment)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate experiment config")
    validate_parser.add_argument("config_file", help="Path to config file")
    validate_parser.set_defaults(func=validate_experiment)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments")
    list_parser.set_defaults(func=list_experiments)
    
    # Templates command
    templates_parser = subparsers.add_parser("templates", help="Show available templates")
    templates_parser.set_defaults(func=show_templates)
    
    # From-template command
    from_template_parser = subparsers.add_parser("from-template", help="Create from template")
    from_template_parser.add_argument("template_name", help="Template name")
    from_template_parser.add_argument("--experiment-id", help="Custom experiment ID")
    from_template_parser.add_argument("--description", help="Custom description")
    from_template_parser.add_argument("--activate", action="store_true", help="Activate immediately")
    from_template_parser.set_defaults(func=from_template)
    
    # Activate command
    activate_parser = subparsers.add_parser("activate", help="Activate an experiment")
    activate_parser.add_argument("experiment_id", help="Experiment ID to activate")
    activate_parser.set_defaults(func=activate_command)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

