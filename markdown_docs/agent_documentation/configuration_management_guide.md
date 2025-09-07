# Configuration Management Documentation

## Overview

The JustNews V4 system implements a comprehensive, hierarchical configuration management system that supports centralized configuration, environment-specific overrides, validation, and dynamic reconfiguration. This documentation covers the complete configuration architecture, management tools, and best practices for production deployment.

## Architecture Overview

### Configuration Hierarchy

The system uses a layered configuration approach:

```
Environment Variables (Highest Priority)
    ↓
Application Runtime Overrides
    ↓
Environment-Specific Configs
    ↓
Base Configuration File (Lowest Priority)
```

### Core Components

1. **Central Configuration File** (`config/system_config.json`) - Master configuration repository
2. **Configuration Manager** (`config/system_config.py`) - Python API for configuration access
3. **Validation System** (`config/validate_config.py`) - Configuration validation and health checks
4. **Quick Reference Tool** (`config/config_quickref.py`) - Human-readable configuration display
5. **GPU Configuration System** (`config/gpu/`) - Specialized GPU configuration management
6. **Environment Overrides** - Runtime configuration customization

## Central Configuration File

### File Structure

The master configuration file (`config/system_config.json`) contains all system settings organized by functional areas:

```json
{
  "system": {
    "name": "JustNewsAgent",
    "version": "4.0",
    "environment": "justnews-v2-py312",
    "log_level": "INFO",
    "debug_mode": false
  },
  "mcp_bus": {
    "host": "localhost",
    "port": 8000,
    "timeout_seconds": 30,
    "max_retries": 3
  }
}
```

### Configuration Sections

#### System Configuration
```json
"system": {
  "name": "JustNewsAgent",
  "version": "4.0",
  "environment": "justnews-v2-py312",
  "conda_environment": "justnews-v2-py312",
  "log_level": "INFO",
  "debug_mode": false
}
```

#### MCP Bus Configuration
```json
"mcp_bus": {
  "host": "localhost",
  "port": 8000,
  "url": "http://localhost:8000",
  "timeout_seconds": 30,
  "max_retries": 3,
  "retry_delay_seconds": 1.0
}
```

#### Database Configuration
```json
"database": {
  "host": "localhost",
  "port": 5432,
  "database": "justnews",
  "user": "justnews_user",
  "password": "",
  "connection_pool": {
    "min_connections": 2,
    "max_connections": 10,
    "connection_timeout_seconds": 3,
    "command_timeout_seconds": 30
  },
  "ssl_mode": "prefer"
}
```

#### Crawling Configuration
```json
"crawling": {
  "enabled": true,
  "obey_robots_txt": true,
  "respect_rate_limits": true,
  "user_agent": "JustNewsAgent/4.0",
  "robots_cache_hours": 1,
  "rate_limiting": {
    "requests_per_minute": 20,
    "delay_between_requests_seconds": 2.0,
    "concurrent_sites": 3,
    "concurrent_browsers": 3,
    "batch_size": 10,
    "articles_per_site": 25,
    "max_total_articles": 100
  },
  "timeouts": {
    "page_load_timeout_seconds": 12000,
    "modal_dismiss_timeout_ms": 1000,
    "request_timeout_seconds": 30
  },
  "delays": {
    "between_batches_seconds": 0.5,
    "between_requests_random_min": 1.0,
    "between_requests_random_max": 3.0
  }
}
```

#### GPU Configuration
```json
"gpu": {
  "enabled": true,
  "devices": {
    "preferred": [0],
    "excluded": [],
    "memory_limits_gb": {
      "0": 24.0
    }
  },
  "memory_management": {
    "max_memory_per_agent_gb": 8.0,
    "safety_margin_percent": 15,
    "enable_cleanup": true,
    "preallocation": false
  },
  "performance": {
    "batch_size_optimization": true,
    "async_operations": true,
    "profiling_enabled": false,
    "metrics_collection_interval_seconds": 10.0
  },
  "health_monitoring": {
    "enabled": true,
    "check_interval_seconds": 30.0,
    "temperature_limits": {
      "warning_celsius": 75,
      "critical_celsius": 85,
      "shutdown_celsius": 95
    }
  }
}
```

#### Agent Configuration
```json
"agents": {
  "ports": {
    "scout": 8002,
    "analyst": 8004,
    "fact_checker": 8003,
    "synthesizer": 8005,
    "critic": 8006,
    "chief_editor": 8001,
    "memory": 8007,
    "reasoning": 8008,
    "dashboard": 8011,
    "db_worker": 8010
  },
  "timeouts": {
    "agent_response_timeout_seconds": 300,
    "health_check_timeout_seconds": 10
  },
  "batch_sizes": {
    "scout_batch_size": 10,
    "analyst_batch_size": 16,
    "synthesizer_batch_size": 8
  }
}
```

#### Training Configuration
```json
"training": {
  "enabled": true,
  "continuous_learning": true,
  "ewc_lambda": 0.1,
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 10,
  "validation_split": 0.2,
  "early_stopping_patience": 5,
  "model_save_interval": 100,
  "max_memory_samples": 10000
}
```

#### Security Configuration
```json
"security": {
  "max_requests_per_minute": 60,
  "rate_limit_window_seconds": 60,
  "enable_ip_filtering": false,
  "allowed_ips": [],
  "api_key_required": false,
  "cors_origins": ["*"],
  "session_timeout_minutes": 30
}
```

#### Monitoring Configuration
```json
"monitoring": {
  "enabled": true,
  "metrics_collection_interval_seconds": 60,
  "alert_thresholds": {
    "cpu_usage_percent": 90,
    "memory_usage_percent": 85,
    "disk_usage_percent": 90,
    "gpu_memory_percent": 90,
    "gpu_temperature_celsius": 85
  },
  "alert_cooldown_minutes": 5,
  "email_alerts": {
    "enabled": false,
    "smtp_server": "",
    "smtp_port": 587,
    "recipients": []
  },
  "log_rotation": {
    "max_file_size_mb": 100,
    "backup_count": 5
  }
}
```

#### Data Minimization Configuration
```json
"data_minimization": {
  "enabled": true,
  "retention_days": {
    "articles": 365,
    "logs": 90,
    "metrics": 30,
    "cache": 7
  },
  "compression": {
    "enabled": true,
    "algorithm": "gzip",
    "level": 6
  },
  "anonymization": {
    "enabled": true,
    "fields": ["ip_address", "user_agent", "session_id"]
  }
}
```

#### Performance Configuration
```json
"performance": {
  "optimization_level": "balanced",
  "memory_pool_size_mb": 1024,
  "thread_pool_size": 10,
  "async_queue_size": 1000,
  "cache_settings": {
    "enabled": true,
    "ttl_seconds": 3600,
    "max_size_mb": 512
  }
}
```

#### External Services Configuration
```json
"external_services": {
  "news_sources": {
    "verification_required": true,
    "max_age_days": 30,
    "auto_discovery": false
  },
  "apis": {
    "timeout_seconds": 10,
    "retry_attempts": 3,
    "rate_limit_per_minute": 100
  }
}
```

## Configuration Manager API

### Python API Usage

#### Basic Configuration Access
```python
from config.system_config import config

# Get single values
db_host = config.get('database.host')
gpu_enabled = config.get('gpu.enabled')

# Get entire sections
crawling_config = config.get_section('crawling')
gpu_config = config.get_section('gpu')

# Set values
config.set('crawling.rate_limiting.requests_per_minute', 15)
```

#### Utility Functions
```python
from config.system_config import (
    get_crawling_config,
    get_database_config,
    get_gpu_config,
    get_rate_limits,
    is_debug_mode,
    get_log_level
)

# Get specific configurations
crawl_config = get_crawling_config()
db_config = get_database_config()
gpu_config = get_gpu_config()
rate_limits = get_rate_limits()

# Check system state
debug_mode = is_debug_mode()
log_level = get_log_level()
```

#### Configuration Reloading
```python
# Reload configuration from file
config.reload()

# Save current configuration
config.save()

# Save to specific file
config.save('/path/to/custom_config.json')
```

### Dictionary-Style Access
```python
# Check if section exists
if 'gpu' in config:
    gpu_settings = config['gpu']

# Iterate over sections
for section_name in config.keys():
    section_data = config[section_name]

# Get all items
for section_name, section_data in config.items():
    print(f"{section_name}: {section_data}")
```

## Environment Overrides

### Environment Variable Mapping

The system supports runtime configuration overrides via environment variables:

#### Database Overrides
```bash
export POSTGRES_HOST=production-db.example.com
export POSTGRES_DB=justnews_prod
export POSTGRES_USER=justnews_app
export POSTGRES_PASSWORD=secure_password_here
```

#### Crawling Overrides
```bash
export CRAWLER_REQUESTS_PER_MINUTE=15
export CRAWLER_DELAY_BETWEEN_REQUESTS=3.0
export CRAWLER_CONCURRENT_SITES=2
```

#### System Overrides
```bash
export LOG_LEVEL=DEBUG
export DEBUG_MODE=true
```

#### GPU Overrides
```bash
export GPU_ENABLED=true
```

### Override Priority

1. **Environment Variables** (Highest priority)
2. **Runtime Configuration Changes**
3. **Environment-Specific Config Files**
4. **Base Configuration File** (Lowest priority)

## GPU Configuration System

### GPU-Specific Configuration Files

#### Main GPU Configuration (`config/gpu/gpu_config.json`)
```json
{
  "gpu_manager": {
    "max_memory_per_agent_gb": 6.0,
    "health_check_interval_seconds": 15.0,
    "allocation_timeout_seconds": 60.0,
    "memory_safety_margin_percent": 15,
    "enable_memory_cleanup": true,
    "enable_health_monitoring": true,
    "enable_performance_tracking": true
  }
}
```

#### Environment-Specific GPU Configs (`config/gpu/environment_config.json`)
```json
{
  "development": {
    "gpu_manager": {
      "max_memory_per_agent_gb": 4.0,
      "health_check_interval_seconds": 15.0
    },
    "performance": {
      "profiling_enabled": true,
      "metrics_collection_interval": 5.0
    }
  },
  "production": {
    "gpu_manager": {
      "max_memory_per_agent_gb": 8.0,
      "health_check_interval_seconds": 30.0
    },
    "performance": {
      "batch_size_optimization": true,
      "memory_preallocation": true
    }
  }
}
```

### GPU Configuration Management

```python
from config.gpu.gpu_config_manager import GPUConfigManager

# Initialize GPU configuration
gpu_config = GPUConfigManager()

# Get GPU settings for current environment
gpu_settings = gpu_config.get_environment_config()

# Check GPU availability
if gpu_config.is_gpu_available():
    device_count = gpu_config.get_device_count()
    memory_limits = gpu_config.get_memory_limits()
```

## Configuration Validation

### Validation System

The configuration validation system provides comprehensive checks:

```python
from config.validate_config import ConfigValidator

# Validate configuration
validator = ConfigValidator('config/system_config.json')
is_valid, report = validator.validate()

if not is_valid:
    print("Configuration errors found:")
    print(report)
    exit(1)
```

### Validation Checks

#### Structural Validation
- Required sections present
- Section types are correct
- Nested structure integrity

#### Logical Validation
- Database connection pool settings
- Rate limiting configurations
- GPU memory allocations
- Performance thresholds

#### Production Readiness
- Security settings validation
- Monitoring configuration checks
- Data minimization compliance

### Validation Report Example
```
=== JustNewsAgent Configuration Validation Report ===

❌ ERRORS:
  • Missing required section: database
  • Database min_connections cannot be greater than max_connections

⚠️  WARNINGS:
  • High requests per minute (100) may violate site policies
  • Very high memory alert threshold (95%) may miss issues

💡 SUGGESTIONS:
  • Consider reducing concurrent browsers or increasing concurrent sites
  • Long retention period for articles (400 days)
```

## Quick Reference Tool

### Usage

```bash
# Display all configuration
python config/config_quickref.py

# Display specific sections
python config/config_quickref.py --section crawling
python config/config_quickref.py --section gpu
```

### Output Format

```
🎯 JustNewsAgent Configuration Quick Reference
📁 Config File: /path/to/system_config.json

============================================================
 🤖 CRAWLING CONFIGURATION
============================================================
General Settings:
  • Enabled: true
  • Obey Robots.txt: true
  • Respect Rate Limits: true
  • User Agent: JustNewsAgent/4.0
  • Robots Cache (hours): 1

Rate Limiting:
  • Requests per Minute: 20
  • Delay Between Requests: 2.0s
  • Concurrent Sites: 3
  • Concurrent Browsers: 3
```

## Environment Management

### Environment-Specific Configurations

The system supports multiple deployment environments:

#### Development Environment
```json
{
  "system": {
    "environment": "development",
    "log_level": "DEBUG",
    "debug_mode": true
  },
  "crawling": {
    "rate_limiting": {
      "requests_per_minute": 5
    }
  }
}
```

#### Staging Environment
```json
{
  "system": {
    "environment": "staging",
    "log_level": "INFO"
  },
  "monitoring": {
    "email_alerts": {
      "enabled": true,
      "recipients": ["staging-alerts@company.com"]
    }
  }
}
```

#### Production Environment
```json
{
  "system": {
    "environment": "production",
    "log_level": "WARNING"
  },
  "security": {
    "enable_ip_filtering": true,
    "allowed_ips": ["10.0.0.0/8", "172.16.0.0/12"]
  },
  "monitoring": {
    "email_alerts": {
      "enabled": true,
      "recipients": ["production-alerts@company.com", "ops@company.com"]
    }
  }
}
```

### Environment Detection

```python
from config.system_config import config

def get_current_environment():
    return config.get('system.environment', 'development')

def is_production():
    return get_current_environment() == 'production'

def is_development():
    return get_current_environment() == 'development'
```

## Configuration Profiles

### Profile-Based Configuration

The system supports configuration profiles for different use cases:

```json
{
  "profiles": {
    "high_performance": {
      "gpu": {
        "memory_management": {
          "max_memory_per_agent_gb": 12.0
        }
      },
      "crawling": {
        "rate_limiting": {
          "concurrent_sites": 10,
          "requests_per_minute": 50
        }
      }
    },
    "conservative": {
      "gpu": {
        "memory_management": {
          "max_memory_per_agent_gb": 4.0
        }
      },
      "crawling": {
        "rate_limiting": {
          "concurrent_sites": 2,
          "requests_per_minute": 10
        }
      }
    }
  }
}
```

### Profile Activation

```python
# Activate configuration profile
config.activate_profile('high_performance')

# Check active profile
active_profile = config.get_active_profile()

# List available profiles
available_profiles = config.list_profiles()
```

## Dynamic Configuration

### Runtime Configuration Changes

```python
from config.system_config import config

# Update configuration at runtime
config.set('crawling.rate_limiting.requests_per_minute', 25)
config.set('gpu.memory_management.max_memory_per_agent_gb', 10.0)

# Changes are immediately effective
# Optionally save to persist changes
config.save()
```

### Hot Reloading

```python
# Enable hot reloading
config.enable_hot_reload(interval_seconds=30)

# Manual reload
config.reload()

# Check if configuration changed
if config.has_changed():
    print("Configuration updated, reloading...")
    config.reload()
```

## Security Considerations

### Configuration Security

#### Sensitive Data Handling
```json
{
  "security": {
    "encrypt_sensitive_fields": true,
    "sensitive_fields": ["database.password", "api_keys"],
    "key_rotation_days": 90
  }
}
```

#### Access Control
```python
# Configuration access control
config.set_access_level('database.password', 'admin_only')
config.set_access_level('gpu.devices', 'read_only')

# Check access permissions
if config.can_access('database.password', current_user):
    password = config.get('database.password')
```

### Audit Logging

```python
# Enable configuration audit logging
config.enable_audit_logging(log_file='/var/log/justnews/config_audit.log')

# Log configuration changes
config.log_change('crawling.rate_limiting.requests_per_minute',
                  old_value=20, new_value=25, user='admin')
```

## Backup and Recovery

### Configuration Backups

```python
from config.backup_manager import ConfigBackupManager

# Initialize backup manager
backup_mgr = ConfigBackupManager(backup_dir='/etc/justnews/config_backups')

# Create backup
backup_id = backup_mgr.create_backup()

# List available backups
backups = backup_mgr.list_backups()

# Restore from backup
backup_mgr.restore_backup(backup_id)
```

### Automated Backups

```json
{
  "backup": {
    "enabled": true,
    "schedule": "daily",
    "retention_days": 30,
    "compression": true,
    "encryption": true
  }
}
```

## Monitoring and Alerting

### Configuration Monitoring

```python
from config.monitor import ConfigMonitor

# Monitor configuration changes
monitor = ConfigMonitor(config)
monitor.watch_file('/etc/justnews/system_config.json')

# Alert on configuration changes
@monitor.on_change
def handle_config_change(changes):
    for change in changes:
        send_alert(f"Configuration changed: {change['key']} = {change['new_value']}")

# Start monitoring
monitor.start()
```

### Configuration Health Checks

```python
from config.health_checker import ConfigHealthChecker

# Check configuration health
health_checker = ConfigHealthChecker(config)
health_status = health_checker.check()

if not health_status['healthy']:
    for issue in health_status['issues']:
        print(f"Configuration issue: {issue}")
```

## Deployment Configuration

### Docker Configuration

```dockerfile
# Dockerfile with configuration
FROM python:3.12-slim

# Copy configuration
COPY config/system_config.json /app/config/
COPY config/gpu/ /app/config/gpu/

# Set environment variables
ENV CONFIG_FILE=/app/config/system_config.json
ENV ENVIRONMENT=production

# Run application
CMD ["python", "main.py"]
```

### Kubernetes ConfigMaps

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: justnews-config
data:
  system_config.json: |
    {
      "system": {
        "environment": "production",
        "log_level": "INFO"
      }
    }
```

### Helm Chart Integration

```yaml
# values.yaml
config:
  system:
    environment: production
    log_level: INFO
  database:
    host: "{{ .Values.postgresql.host }}"
    database: "{{ .Values.postgresql.database }}"
```

## Troubleshooting

### Common Configuration Issues

#### Configuration File Not Found
```bash
# Check file location
ls -la config/system_config.json

# Check file permissions
stat config/system_config.json

# Validate JSON syntax
python -m json.tool config/system_config.json
```

#### Environment Override Not Working
```bash
# Check environment variable
echo $POSTGRES_HOST

# Verify variable is exported
export | grep POSTGRES

# Check configuration loading order
python -c "from config.system_config import config; print(config.config_file)"
```

#### GPU Configuration Issues
```bash
# Check GPU availability
nvidia-smi

# Validate GPU configuration
python -c "from config.gpu.gpu_config_manager import GPUConfigManager; print(GPUConfigManager().is_gpu_available())"

# Check environment-specific GPU config
python -c "from config.gpu.gpu_config_manager import GPUConfigManager; print(GPUConfigManager().get_environment_config())"
```

### Configuration Validation

```bash
# Run configuration validation
python config/validate_config.py

# Check for specific issues
python -c "
from config.validate_config import ConfigValidator
validator = ConfigValidator('config/system_config.json')
is_valid, report = validator.validate()
print(report)
"
```

### Configuration Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

from config.system_config import config

# Debug configuration loading
config.reload()
print(f"Config file: {config.config_file}")
print(f"Config sections: {list(config.keys())}")

# Check specific values
print(f"Database host: {config.get('database.host')}")
print(f"GPU enabled: {config.get('gpu.enabled')}")
```

## Best Practices

### Configuration Management

1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configurations for each environment
3. **Documentation**: Document all configuration options
4. **Validation**: Always validate configuration before deployment
5. **Backup**: Regularly backup working configurations

### Security Best Practices

1. **No Secrets in Config**: Use environment variables for sensitive data
2. **Access Control**: Implement proper access controls for configuration
3. **Audit Logging**: Enable audit logging for configuration changes
4. **Encryption**: Encrypt sensitive configuration data at rest

### Performance Optimization

1. **Caching**: Cache frequently accessed configuration values
2. **Lazy Loading**: Load configuration sections on demand
3. **Validation Caching**: Cache validation results when possible
4. **Async Operations**: Use async operations for configuration updates

### Monitoring Best Practices

1. **Change Tracking**: Monitor configuration changes in production
2. **Health Checks**: Implement configuration health checks
3. **Alerting**: Set up alerts for configuration issues
4. **Metrics**: Collect metrics on configuration access patterns

---

*This comprehensive configuration management documentation covers all aspects of the JustNews V4 configuration system. For specific implementation details, refer to the individual configuration files and Python modules.*
