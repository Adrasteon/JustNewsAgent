---
title: Documentation Management Tools
description: Auto-generated description for Documentation Management Tools
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Documentation Management Tools

This directory contains automated tools for managing the JustNewsAgent documentation catalogue and maintaining documentation quality.

## Available Tools

### Core Management Scripts

#### `docs_navigator.py`
**Purpose**: Interactive documentation navigator and status checker
**Usage**:
```bash
python docs_navigator.py status          # Show catalogue status
python docs_navigator.py list            # List all categories
python docs_navigator.py search <term>   # Search documentation
python docs_navigator.py validate        # Validate catalogue integrity
```

#### `catalogue_expansion.py`
**Purpose**: Automated catalogue expansion and metadata generation
**Usage**:
```bash
python catalogue_expansion.py --phase all                    # Expand all phases
python catalogue_expansion.py --phase development_reports    # Expand development reports
python catalogue_expansion.py --validate                    # Validate catalogue
```

#### `maintenance_action_plan.py`
**Purpose**: Generate maintenance reports and action plans
**Usage**:
```bash
python maintenance_action_plan.py    # Generate maintenance report
```

### Quality Assurance Scripts

#### `doc_linter.py`
**Purpose**: Validate markdown docs for frontmatter, location, and fence formatting; optionally auto-fix.
**Usage**:
```bash
python doc_linter.py --report           # JSON report of issues
python doc_linter.py --report --fix     # Apply safe fixes and report
# Optional flags
python doc_linter.py --strict-location   # Treat out-of-place files as errors (default: warn)
python doc_linter.py --add-seealso --fix # Inject default See also section when missing
```

#### `automation_tools.py`
**Purpose**: Automated quality monitoring and maintenance scheduling
**Usage**:
```bash
python automation_tools.py    # Run complete automation suite
```

#### `quality_monitor.py`
**Purpose**: Quality monitoring dashboard and metrics
**Usage**:
```bash
python quality_monitor.py --catalogue ../docs_catalogue_v2.json
```

#### `quality_enhancement.py`
**Purpose**: Quality enhancement and improvement suggestions
**Usage**:
```bash
python quality_enhancement.py
```

### Cross-Reference Management

#### `cross_reference_repair.py`
**Purpose**: Cross-reference validation and repair
**Usage**:
```bash
python cross_reference_repair.py
```

#### `find_broken_refs.py`
**Purpose**: Find broken cross-references in documentation
**Usage**:
```bash
python find_broken_refs.py
```

#### `fix_cross_references.py`
**Purpose**: Automated cross-reference fixing
**Usage**:
```bash
python fix_cross_references.py
```

#### `fix_final_refs.py` / `fix_last_ref.py`
**Purpose**: Final cross-reference fixes and cleanup
**Usage**:
```bash
python fix_final_refs.py
python fix_last_ref.py
```

### Catalogue Organization

#### `generate_docs_index.py`
**Purpose**: Build `docs_index.json` from `markdown_docs/` frontmatter grouped by category.
**Usage**:
```bash
python generate_docs_index.py --write    # Write docs_index.json at repo root
python generate_docs_index.py --dry-run  # Preview output (first categories)
```

#### `catalogue_reorganization.py`
**Purpose**: Reorganize large categories into subcategories
**Usage**:
```bash
python catalogue_reorganization.py
```

#### `agent_docs_reorganization.py`
**Purpose**: Specialized reorganization for agent documentation
**Usage**:
```bash
python agent_docs_reorganization.py
```

#### `catalogue_maintenance.py`
**Purpose**: General catalogue maintenance operations
**Usage**:
```bash
python catalogue_maintenance.py
```

### Utility Scripts

#### `automation_tools.py`
**Purpose**: General automation and maintenance tools
**Usage**:
```bash
python automation_tools.py
```

#### `version_control.py`
**Purpose**: Version control and change tracking
**Usage**:
```bash
python version_control.py
```

#### `setup_quality_system.py`
**Purpose**: Set up quality monitoring system
**Usage**:
```bash
python setup_quality_system.py
```

## Directory Structure

```
doc_management_tools/
├── README.md                          # This file
├── docs_navigator.py                  # Main navigation tool
├── catalogue_expansion.py             # Catalogue expansion
├── maintenance_action_plan.py         # Maintenance reporting
├── automation_tools.py                # Quality automation
├── quality_monitor.py                 # Quality monitoring
├── quality_enhancement.py             # Quality enhancement
├── cross_reference_repair.py          # Cross-reference repair
├── find_broken_refs.py                # Find broken references
├── fix_cross_references.py            # Fix cross-references
├── fix_final_refs.py                  # Final reference fixes
├── fix_last_ref.py                    # Last reference fixes
├── catalogue_reorganization.py        # Catalogue reorganization
├── agent_docs_reorganization.py       # Agent docs reorganization
├── catalogue_maintenance.py           # Catalogue maintenance
├── version_control.py                 # Version control
└── setup_quality_system.py            # Quality system setup
```

## Usage Guidelines

1. **Always run from the `doc_management_tools/` directory**:
   ```bash
   cd docs/doc_management_tools
   python <script_name>.py
   ```

2. **Catalogue paths are automatically adjusted** for the new location

3. **Regular maintenance schedule**:
   - Daily: Run `docs_navigator.py validate`
   - Weekly: Run `automation_tools.py`
   - Monthly: Run `maintenance_action_plan.py`

4. **Before major changes**: Always backup `../docs_catalogue_v2.json`

## Dependencies

All scripts use standard Python libraries:
- `json` - JSON processing
- `pathlib` - Path handling
- `datetime` - Date/time operations
- `typing` - Type hints
- `collections` - Data structures

## Output Files

Scripts generate output in the parent `docs/` directory:
- `MAINTENANCE_ACTION_PLAN.md` - Maintenance reports
- `DOCUMENTATION_CATALOGUE.md` - Human-readable catalogue
- `docs_catalogue_v2.json` - Machine-readable catalogue (updated)

## Troubleshooting

**Common Issues:**
- **Path errors**: Ensure running from `doc_management_tools/` directory
- **Permission errors**: Check write permissions for `../docs/` directory
- **Import errors**: All dependencies are standard library

**Getting Help:**
- Run any script with `--help` for usage information
- Check the main documentation in `../README.md`
- Review error messages for specific guidance

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

