# Configuration Versioning & Rollback

Locations
- Configs: `config/`
- Versioned backups: `config/versions/` (auto-saved on change)

Workflow
1. Edit base config (e.g., `rules.json`) or variant (e.g., `rules_A.json`, `rules_B.json`).
2. On save, watcher writes a timestamped backup to `config/versions/` and reloads configs.
3. Use `switch_variant(A|B)` for A/B rollout. `load_all_configs()` reflects the active variant.

Validation
- `kw_weight + emb_weight == 1.0` enforced for rules.
- Invalid files are skipped with a warning.

Rollback
- Copy a backup from `config/versions/` over the active file and save; watcher reloads it.
- Or manually call `load_all_configs()` after replacing the file.

Testing
- Unit tests: `tests/test_config_manager.py` cover variant switching and validation.
- Hot-reload demo: save a config file while the app runs to see auto-reload logs.
