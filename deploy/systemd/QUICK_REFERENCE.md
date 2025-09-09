# JustNews Systemd Quick Reference

## 🚀 Quick Start

```bash
# One-time setup
sudo ./deploy/systemd/complete_postgresql.sh
sudo cp deploy/systemd/env/*.env /etc/justnews/
sudo cp deploy/systemd/units/justnews@.service /etc/systemd/system/
sudo cp deploy/systemd/scripts/* /usr/local/bin/
sudo chmod +x /usr/local/bin/justnews-*
sudo systemctl daemon-reload

# Deploy all services
sudo ./deploy/systemd/enable_all.sh fresh

# Check health
./deploy/systemd/health_check.sh
```

## 📊 Service Status Overview

| Service | Port | Status | Description |
|---------|------|--------|-------------|
| mcp_bus | 8000 | ✅ | Central communication hub |
| chief_editor | 8001 | ✅ | Workflow orchestration |
| scout | 8002 | ✅ | Content discovery |
| fact_checker | 8003 | ⚠️ | Fact verification |
| analyst | 8004 | ⚠️ | Sentiment analysis |
| synthesizer | 8005 | ⚠️ | Content synthesis |
| critic | 8006 | ⚠️ | Quality assessment |
| memory | 8007 | ⚠️ | Knowledge storage |
| reasoning | 8008 | ✅ | Logical reasoning |
| newsreader | 8009 | ⚠️ | Content processing |
| balancer | 8010 | ⚠️ | Load balancing |
| dashboard | 8011 | ⚠️ | Web interface |
| analytics | 8012 | ⚠️ | Performance analytics |
| archive | 8013 | ⚠️ | Content archiving |

**Legend**: ✅ Active/Healthy, ⚠️ Inactive/Needs Start, ❌ Failed

## 🎛️ Common Commands

### Service Management
```bash
# Start all services
sudo ./deploy/systemd/enable_all.sh start

# Stop all services
sudo ./deploy/systemd/enable_all.sh stop

# Restart all services
sudo ./deploy/systemd/enable_all.sh restart

# Fresh start (recommended)
sudo ./deploy/systemd/enable_all.sh fresh

# Check status
sudo ./deploy/systemd/enable_all.sh status
```

### Individual Service Control
```bash
# Start specific service
sudo systemctl start justnews@mcp_bus

# Stop specific service
sudo systemctl stop justnews@mcp_bus

# Restart specific service
sudo systemctl restart justnews@mcp_bus

# Check specific service
sudo systemctl status justnews@mcp_bus

# View logs
journalctl -u justnews@mcp_bus -f
```

### Health Monitoring
```bash
# Check all services
./deploy/systemd/health_check.sh

# Check specific services
./deploy/systemd/health_check.sh mcp_bus chief_editor

# Verbose health check
./deploy/systemd/health_check.sh --verbose
```

## 🔧 Configuration

### Environment Files
- **Global**: `/etc/justnews/global.env`
- **Service-specific**: `/etc/justnews/{service}.env`

### Key Settings
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.8

# Database
DATABASE_URL=postgresql://justnews_user:password123@localhost:5432/justnews

# Performance
MAX_WORKERS=4
BATCH_SIZE=16
```

## 🚨 Troubleshooting

### Quick Diagnostics
```bash
# System health check
./deploy/systemd/preflight.sh

# Service-specific logs
journalctl -u justnews@mcp_bus -n 20

# Port conflicts
sudo lsof -i :8000

# Database connection
psql -h localhost -U justnews_user -d justnews -c "SELECT 1;"
```

### Common Issues
1. **Service won't start**: Check environment files and logs
2. **Port in use**: Kill conflicting processes
3. **Database errors**: Verify PostgreSQL setup
4. **GPU issues**: Check CUDA_VISIBLE_DEVICES settings

### Emergency Recovery
```bash
# Stop everything
sudo ./deploy/systemd/enable_all.sh stop

# Clean restart
sudo ./deploy/systemd/enable_all.sh fresh

# Check health
./deploy/systemd/health_check.sh
```

## 📈 Monitoring

### Real-time Monitoring
```bash
# System resources
top -p $(pgrep -f justnews)

# GPU usage
nvidia-smi -l 1

# Database connections
psql -h localhost -U justnews_user -d justnews -c "SELECT count(*) FROM pg_stat_activity;"

# Service logs
journalctl -t justnews -f
```

### Performance Metrics
- **CPU**: < 70% utilization
- **Memory**: < 80% utilization
- **GPU Memory**: < 90% utilization
- **Response Time**: < 2 seconds
- **Active Connections**: < 50

## 🔄 Maintenance

### Daily
```bash
./deploy/systemd/health_check.sh
df -h /opt/justnews
journalctl -t justnews --since "1 day ago" -p err
```

### Weekly
```bash
sudo journalctl --vacuum-time=7d
psql -h localhost -U justnews_user -d justnews -c "VACUUM ANALYZE;"
```

### Monthly
```bash
./deploy/systemd/full_backup.sh
./deploy/systemd/performance_analysis.sh
```

## 📚 Documentation

- **Complete Guide**: `deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md`
- **Deployment**: `deploy/systemd/DEPLOYMENT.md`
- **README**: `deploy/systemd/README.md`

## 🆘 Support

For issues:
1. Check `deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md`
2. Run diagnostics: `./deploy/systemd/preflight.sh`
3. Review logs: `journalctl -t justnews -p err`
4. Create issue with diagnostic output

---

**Status**: ✅ Production Ready | **Services**: 4/14 Active | **Health**: Good</content>
<parameter name="filePath">/home/adra/justnewsagent/JustNewsAgent/deploy/systemd/QUICK_REFERENCE.md
