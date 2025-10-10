# JustNews Deployment Examples

This directory contains example configuration files for deploying JustNews services.

## Environment Configuration

### `justnews.env.example`
Global environment configuration template for all JustNews services. Copy to `/etc/justnews/global.env` and customize for your deployment.

### `gpu_orchestrator.env.example`
GPU orchestrator-specific environment configuration. Copy to `/etc/justnews/gpu_orchestrator.env` and customize GPU settings.

## Usage

1. Copy example files and remove `.example` extension:
   ```bash
   sudo cp justnews.env.example /etc/justnews/global.env
   sudo cp gpu_orchestrator.env.example /etc/justnews/gpu_orchestrator.env
   ```

2. Edit the files with your specific configuration:
   ```bash
   sudo nano /etc/justnews/global.env
   sudo nano /etc/justnews/gpu_orchestrator.env
   ```

3. Ensure proper permissions:
   ```bash
   sudo chown root:justnews /etc/justnews/*.env
   sudo chmod 640 /etc/justnews/*.env
   ```

## Related Files

- **Systemd Units**: See `../units/` for service definitions
- **Helper Scripts**: See `../scripts/` for operational scripts
- **Logrotate Config**: See `../logrotate.conf` for log rotation setup
