"""
Advanced Analytics Dashboard for JustNewsAgent

Provides comprehensive web-based analytics interface with:
- Real-time performance monitoring
- Historical trend analysis
- Agent performance profiling
- System health monitoring
- Bottleneck detection and recommendations
- Interactive charts and visualizations
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..common.advanced_analytics import get_analytics_engine

# Initialize analytics engine
analytics_engine = get_analytics_engine()

# FastAPI app for analytics dashboard
analytics_app = FastAPI(title="JustNewsAgent Advanced Analytics Dashboard")

# Templates and static files
templates_dir = Path(__file__).parent / "analytics" / "templates"
static_dir = Path(__file__).parent / "analytics" / "static"

templates_dir.mkdir(parents=True, exist_ok=True)
static_dir.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
analytics_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@analytics_app.get("/", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    """Main analytics dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@analytics_app.get("/api/health")
async def get_system_health():
    """Get system health metrics"""
    try:
        health = analytics_engine.get_system_health_score()
        return JSONResponse(content=health)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@analytics_app.get("/api/realtime/{hours}")
async def get_realtime_analytics(hours: int = 1):
    """Get real-time analytics for specified hours"""
    try:
        if hours < 1 or hours > 24:
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 24")

        analytics = analytics_engine.get_real_time_analytics(hours=hours)
        return JSONResponse(content=analytics.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")

@analytics_app.get("/api/agent/{agent_name}/{hours}")
async def get_agent_profile(agent_name: str, hours: int = 24):
    """Get performance profile for specific agent"""
    try:
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")

        profile = analytics_engine.get_agent_performance_profile(agent_name, hours=hours)
        return JSONResponse(content=profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent profile retrieval failed: {str(e)}")

@analytics_app.get("/api/trends/{hours}")
async def get_performance_trends(hours: int = 24):
    """Get performance trends analysis"""
    try:
        analytics = analytics_engine.get_real_time_analytics(hours=hours)
        return JSONResponse(content={
            "trends": analytics.performance_trends,
            "bottlenecks": analytics.bottleneck_indicators,
            "recommendations": analytics.optimization_recommendations
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trends analysis failed: {str(e)}")

@analytics_app.get("/api/report/{hours}")
async def get_analytics_report(hours: int = 24):
    """Get comprehensive analytics report"""
    try:
        report = analytics_engine.export_analytics_report(hours=hours)
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@analytics_app.get("/api/optimization-recommendations")
async def get_optimization_recommendations(hours: int = 24):
    """Get advanced optimization recommendations"""
    try:
        from ..common.advanced_optimization import generate_optimization_recommendations

        recommendations = generate_optimization_recommendations(hours)
        return JSONResponse(content=[
            {
                "id": rec.id,
                "category": rec.category.value,
                "priority": rec.priority.value,
                "title": rec.title,
                "description": rec.description,
                "impact_score": rec.impact_score,
                "confidence_score": rec.confidence_score,
                "complexity": rec.implementation_complexity,
                "time_savings": rec.estimated_time_savings,
                "affected_agents": rec.affected_agents,
                "steps": rec.implementation_steps
            }
            for rec in recommendations
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization analysis failed: {str(e)}")

@analytics_app.get("/api/optimization-insights")
async def get_optimization_insights():
    """Get optimization insights and analytics"""
    try:
        from ..common.advanced_optimization import get_optimization_insights

        insights = get_optimization_insights()
        return JSONResponse(content=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")

@analytics_app.post("/api/record-metric")
async def record_custom_metric(metric_data: dict[str, Any]):
    """Record a custom performance metric"""
    try:
        from ..common.advanced_analytics import PerformanceMetrics

        # Validate required fields
        required_fields = ["agent_name", "operation", "processing_time_s", "batch_size", "success"]
        for field in required_fields:
            if field not in metric_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Create metric object
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            agent_name=metric_data["agent_name"],
            operation=metric_data["operation"],
            processing_time_s=float(metric_data["processing_time_s"]),
            batch_size=int(metric_data["batch_size"]),
            success=bool(metric_data["success"]),
            gpu_memory_allocated_mb=float(metric_data.get("gpu_memory_allocated_mb", 0.0)),
            gpu_memory_reserved_mb=float(metric_data.get("gpu_memory_reserved_mb", 0.0)),
            gpu_utilization_pct=float(metric_data.get("gpu_utilization_pct") or 0.0),
            temperature_c=float(metric_data.get("temperature_c") or 0.0),
            power_draw_w=float(metric_data.get("power_draw_w") or 0.0),
            throughput_items_per_s=float(metric_data.get("throughput_items_per_s", 0.0))
        )

        # Record the metric
        analytics_engine.record_metric(metric)

        return JSONResponse(content={"status": "success", "message": "Metric recorded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metric recording failed: {str(e)}")

# Create analytics dashboard HTML template
analytics_dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JustNewsAgent Advanced Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.0.1/build/global/luxon.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c3e50;
            display: block;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 5px;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-healthy { background-color: #27ae60; }
        .status-warning { background-color: #f39c12; }
        .status-critical { background-color: #e74c3c; }

        .bottlenecks-list {
            list-style: none;
            padding: 0;
        }

        .bottlenecks-list li {
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
            color: #e74c3c;
        }

        .recommendations-list {
            list-style: none;
            padding: 0;
        }

        .recommendations-list li {
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
            color: #27ae60;
        }

        .agent-profile {
            margin-top: 20px;
        }

        .agent-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #9b59b6;
        }

        .agent-name {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .agent-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }

        .agent-metric {
            text-align: center;
            font-size: 0.9rem;
        }

        .agent-metric-value {
            font-weight: bold;
            color: #3498db;
        }

        .controls {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .control-group label {
            font-weight: 500;
            color: #2c3e50;
        }

        .control-group select,
        .control-group input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 0.9rem;
        }

        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s;
        }

        .btn:hover {
            background: #2980b9;
        }

        .btn-secondary {
            background: #95a5a6;
        }

        .btn-secondary:hover {
            background: #7f8c8d;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }

        .error {
            background: #fee;
            border: 1px solid #fcc;
            color: #c33;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }

        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }

            .metric-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .controls {
                flex-direction: column;
                align-items: stretch;
            }

            .control-group {
                justify-content: space-between;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 JustNewsAgent Advanced Analytics</h1>
            <p>Real-time Performance Monitoring & Optimization Insights</p>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="timeRange">Time Range:</label>
                <select id="timeRange">
                    <option value="1">Last Hour</option>
                    <option value="6">Last 6 Hours</option>
                    <option value="24" selected>Last 24 Hours</option>
                    <option value="72">Last 3 Days</option>
                    <option value="168">Last Week</option>
                </select>
            </div>
            <div class="control-group">
                <label for="agentSelect">Agent:</label>
                <select id="agentSelect">
                    <option value="all">All Agents</option>
                    <option value="scout">Scout</option>
                    <option value="analyst">Analyst</option>
                    <option value="synthesizer">Synthesizer</option>
                    <option value="fact_checker">Fact Checker</option>
                    <option value="newsreader">NewsReader</option>
                    <option value="memory">Memory</option>
                </select>
            </div>
            <button class="btn" onclick="refreshData()">🔄 Refresh</button>
            <button class="btn btn-secondary" onclick="exportReport()">📊 Export Report</button>
        </div>

        <div id="loading" class="loading">Loading analytics data...</div>
        <div id="error" class="error" style="display: none;"></div>

        <div id="dashboard" style="display: none;">
            <!-- System Health Card -->
            <div class="dashboard-grid">
                <div class="card">
                    <h3>🩺 System Health</h3>
                    <div class="metric-grid">
                        <div class="metric">
                            <span class="metric-value" id="healthScore">-</span>
                            <div class="metric-label">Health Score</div>
                        </div>
                        <div class="metric">
                            <span class="metric-value" id="successRate">-</span>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric">
                            <span class="metric-value" id="avgThroughput">-</span>
                            <div class="metric-label">Avg Throughput</div>
                        </div>
                        <div class="metric">
                            <span class="metric-value" id="peakMemory">-</span>
                            <div class="metric-label">Peak Memory</div>
                        </div>
                    </div>
                    <div id="healthStatus">
                        <span class="status-indicator" id="healthIndicator"></span>
                        <span id="healthStatusText">Loading...</span>
                    </div>
                </div>

                <!-- Performance Overview Card -->
                <div class="card">
                    <h3>📈 Performance Overview</h3>
                    <div class="metric-grid">
                        <div class="metric">
                            <span class="metric-value" id="totalOps">-</span>
                            <div class="metric-label">Total Operations</div>
                        </div>
                        <div class="metric">
                            <span class="metric-value" id="avgProcessingTime">-</span>
                            <div class="metric-label">Avg Processing Time</div>
                        </div>
                        <div class="metric">
                            <span class="metric-value" id="gpuUtilization">-</span>
                            <div class="metric-label">GPU Utilization</div>
                        </div>
                        <div class="metric">
                            <span class="metric-value" id="activeAgents">-</span>
                            <div class="metric-label">Active Agents</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Charts Row -->
            <div class="dashboard-grid">
                <div class="card">
                    <h3>📊 Performance Trends</h3>
                    <div class="chart-container">
                        <canvas id="performanceChart"></canvas>
                    </div>
                </div>

                <div class="card">
                    <h3>🔥 GPU Resource Usage</h3>
                    <div class="chart-container">
                        <canvas id="gpuChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Issues and Recommendations -->
            <div class="dashboard-grid">
                <div class="card">
                    <h3>⚠️ Current Bottlenecks</h3>
                    <ul class="bottlenecks-list" id="bottlenecksList">
                        <li>Loading...</li>
                    </ul>
                </div>

                <div class="card">
                    <h3>💡 Optimization Recommendations</h3>
                    <ul class="recommendations-list" id="recommendationsList">
                        <li>Loading...</li>
                    </ul>
                </div>
            </div>

            <!-- Optimization Recommendations -->
            <div class="dashboard-grid">
                <div class="card">
                    <h3>🎯 Advanced Optimization Recommendations</h3>
                    <div id="optimizationRecommendations">
                        Loading optimization recommendations...
                    </div>
                </div>

                <div class="card">
                    <h3>📊 Optimization Insights</h3>
                    <div id="optimizationInsights">
                        Loading optimization insights...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentData = {};
        let performanceChart = null;
        let gpuChart = null;

        async function loadData() {
            const timeRange = document.getElementById('timeRange').value;
            const agentSelect = document.getElementById('agentSelect').value;

            try {
                document.getElementById('error').style.display = 'none';
                document.getElementById('loading').style.display = 'block';
                document.getElementById('dashboard').style.display = 'none';

                // Load system health
                const healthResponse = await fetch('/api/health');
                const healthData = await healthResponse.json();

                // Load analytics data
                const analyticsResponse = await fetch(`/api/realtime/${timeRange}`);
                const analyticsData = await analyticsResponse.json();

                // Load agent profiles if specific agent selected
                let agentProfiles = {};
                if (agentSelect !== 'all') {
                    const agentResponse = await fetch(`/api/agent/${agentSelect}/${timeRange}`);
                    agentProfiles = await agentResponse.json();
                }

                currentData = { health: healthData, analytics: analyticsData, agentProfiles };

                updateDashboard();
                loadOptimizationData();

            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('error').textContent = `Error loading data: ${error.message}`;
                document.getElementById('error').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard').style.display = 'block';
            }
        }

        function updateDashboard() {
            const { health, analytics, agentProfiles } = currentData;

            // Update health metrics
            document.getElementById('healthScore').textContent = `${health.overall_health_score.toFixed(1)}%`;
            document.getElementById('successRate').textContent = `${analytics.success_rate_pct.toFixed(1)}%`;
            document.getElementById('avgThroughput').textContent = `${analytics.avg_throughput_items_per_s.toFixed(1)}/s`;
            document.getElementById('peakMemory').textContent = `${analytics.peak_gpu_memory_mb.toFixed(0)}MB`;

            // Update health status
            const healthIndicator = document.getElementById('healthIndicator');
            const healthStatusText = document.getElementById('healthStatusText');
            healthIndicator.className = 'status-indicator';

            if (health.status === 'healthy') {
                healthIndicator.classList.add('status-healthy');
                healthStatusText.textContent = 'System Healthy';
            } else if (health.status === 'warning') {
                healthIndicator.classList.add('status-warning');
                healthStatusText.textContent = 'System Warning';
            } else {
                healthIndicator.classList.add('status-critical');
                healthStatusText.textContent = 'System Critical';
            }

            // Update performance overview
            document.getElementById('totalOps').textContent = analytics.total_operations.toLocaleString();
            document.getElementById('avgProcessingTime').textContent = `${analytics.avg_processing_time_s.toFixed(2)}s`;
            document.getElementById('gpuUtilization').textContent = `${analytics.avg_gpu_utilization_pct.toFixed(1)}%`;
            document.getElementById('activeAgents').textContent = '6'; // Static for now

            // Update bottlenecks
            const bottlenecksList = document.getElementById('bottlenecksList');
            bottlenecksList.innerHTML = '';
            if (analytics.bottleneck_indicators.length === 0) {
                bottlenecksList.innerHTML = '<li>No bottlenecks detected</li>';
            } else {
                analytics.bottleneck_indicators.forEach(bottleneck => {
                    const li = document.createElement('li');
                    li.textContent = bottleneck;
                    bottlenecksList.appendChild(li);
                });
            }

            // Update recommendations
            const recommendationsList = document.getElementById('recommendationsList');
            recommendationsList.innerHTML = '';
            if (analytics.optimization_recommendations.length === 0) {
                recommendationsList.innerHTML = '<li>No recommendations available</li>';
            } else {
                analytics.optimization_recommendations.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    recommendationsList.appendChild(li);
                });
            }

            // Update agent profiles
            updateAgentProfiles(agentProfiles);

            // Update charts
            updateCharts();
        }

        function updateAgentProfiles(profiles) {
            const container = document.getElementById('agentProfiles');
            container.innerHTML = '';

            if (Object.keys(profiles).length === 0) {
                container.textContent = 'No agent profiles available';
                return;
            }

            Object.entries(profiles).forEach(([agentName, profile]) => {
                const agentCard = document.createElement('div');
                agentCard.className = 'agent-card';

                agentCard.innerHTML = `
                    <div class="agent-name">${agentName.charAt(0).toUpperCase() + agentName.slice(1)}</div>
                    <div class="agent-metrics">
                        <div class="agent-metric">
                            <div class="agent-metric-value">${profile.performance_stats.avg_processing_time_s.toFixed(2)}s</div>
                            <div>Avg Time</div>
                        </div>
                        <div class="agent-metric">
                            <div class="agent-metric-value">${profile.performance_stats.success_rate_pct.toFixed(1)}%</div>
                            <div>Success Rate</div>
                        </div>
                        <div class="agent-metric">
                            <div class="agent-metric-value">${profile.performance_stats.avg_throughput_items_per_s.toFixed(1)}/s</div>
                            <div>Throughput</div>
                        </div>
                        <div class="agent-metric">
                            <div class="agent-metric-value">${profile.performance_stats.peak_memory_mb.toFixed(0)}MB</div>
                            <div>Peak Memory</div>
                        </div>
                    </div>
                `;

                container.appendChild(agentCard);
            });
        }

        function updateCharts() {
            // Performance Trends Chart
            const ctx1 = document.getElementById('performanceChart').getContext('2d');
            if (performanceChart) {
                performanceChart.destroy();
            }

            performanceChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: ['-4h', '-3h', '-2h', '-1h', 'Now'],
                    datasets: [{
                        label: 'Processing Time (s)',
                        data: [2.1, 1.9, 2.3, 1.8, currentData.analytics.avg_processing_time_s],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Throughput (items/s)',
                        data: [45, 52, 48, 55, currentData.analytics.avg_throughput_items_per_s],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // GPU Usage Chart
            const ctx2 = document.getElementById('gpuChart').getContext('2d');
            if (gpuChart) {
                gpuChart.destroy();
            }

            gpuChart = new Chart(ctx2, {
                type: 'doughnut',
                data: {
                    labels: ['GPU Utilization', 'Available'],
                    datasets: [{
                        data: [currentData.analytics.avg_gpu_utilization_pct, 100 - currentData.analytics.avg_gpu_utilization_pct],
                        backgroundColor: ['#f39c12', '#ecf0f1'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                        }
                    }
                }
            });
        }

        async function refreshData() {
            await loadData();
            await loadOptimizationData();
        }

        async function exportReport() {
            try {
                const timeRange = document.getElementById('timeRange').value;
                const response = await fetch(`/api/report/${timeRange}`);
                const report = await response.json();

                const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `analytics_report_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } catch (error) {
                alert('Error exporting report: ' + error.message);
            }
        }

        async function loadOptimizationData() {
            try {
                // Load optimization recommendations
                const recResponse = await fetch(`/api/optimization-recommendations?hours=${currentData.timeRange || 24}`);
                const recommendations = await recResponse.json();

                // Load optimization insights
                const insightsResponse = await fetch('/api/optimization-insights');
                const insights = await insightsResponse.json();

                displayOptimizationRecommendations(recommendations);
                displayOptimizationInsights(insights);

            } catch (error) {
                console.error('Error loading optimization data:', error);
                document.getElementById('optimizationRecommendations').textContent = 'Error loading recommendations';
                document.getElementById('optimizationInsights').textContent = 'Error loading insights';
            }
        }

        function displayOptimizationRecommendations(recommendations) {
            const container = document.getElementById('optimizationRecommendations');
            container.innerHTML = '';

            if (!recommendations || recommendations.length === 0) {
                container.textContent = 'No optimization recommendations available';
                return;
            }

            recommendations.forEach(rec => {
                const recCard = document.createElement('div');
                recCard.className = 'agent-card';
                recCard.style.borderLeftColor = getPriorityColor(rec.priority);

                recCard.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div class="agent-name">${rec.title}</div>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span class="status-indicator" style="background-color: ${getPriorityColor(rec.priority)}"></span>
                            <span style="font-size: 0.8rem; color: #666;">${rec.priority.toUpperCase()}</span>
                            <span style="font-size: 0.8rem; color: #666;">${rec.category.replace('_', ' ').toUpperCase()}</span>
                        </div>
                    </div>
                    <div style="margin-bottom: 10px; color: #555; font-size: 0.9rem;">${rec.description}</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 10px;">
                        <div style="text-align: center; font-size: 0.8rem;">
                            <div style="font-weight: bold; color: #27ae60;">${rec.impact_score.toFixed(0)}%</div>
                            <div>Impact</div>
                        </div>
                        <div style="text-align: center; font-size: 0.8rem;">
                            <div style="font-weight: bold; color: #3498db;">${rec.confidence_score.toFixed(0)}%</div>
                            <div>Confidence</div>
                        </div>
                        <div style="text-align: center; font-size: 0.8rem;">
                            <div style="font-weight: bold; color: #e67e22;">${rec.complexity}</div>
                            <div>Complexity</div>
                        </div>
                        <div style="text-align: center; font-size: 0.8rem;">
                            <div style="font-weight: bold; color: #9b59b6;">${rec.time_savings.toFixed(1)}s</div>
                            <div>Time Saved</div>
                        </div>
                    </div>
                    <div style="font-size: 0.8rem; color: #666;">
                        <strong>Affected:</strong> ${rec.affected_agents.join(', ')}
                    </div>
                `;

                container.appendChild(recCard);
            });
        }

        function displayOptimizationInsights(insights) {
            const container = document.getElementById('optimizationInsights');
            container.innerHTML = '';

            if (!insights || insights.error) {
                container.textContent = 'Error loading optimization insights';
                return;
            }

            const insightsGrid = document.createElement('div');
            insightsGrid.style.display = 'grid';
            insightsGrid.style.gridTemplateColumns = 'repeat(auto-fit, minmax(150px, 1fr))';
            insightsGrid.style.gap = '15px';

            // Total recommendations
            const totalCard = createInsightCard(
                insights.total_recommendations_generated || 0,
                'Total Recommendations',
                '#3498db'
            );

            // Average impact
            const impactCard = createInsightCard(
                `${insights.average_impact_score?.toFixed(1) || 0}%`,
                'Avg Impact Score',
                '#27ae60'
            );

            // Most common category
            const categoryCard = createInsightCard(
                insights.most_common_category || 'N/A',
                'Top Category',
                '#e67e22'
            );

            insightsGrid.appendChild(totalCard);
            insightsGrid.appendChild(impactCard);
            insightsGrid.appendChild(categoryCard);

            container.appendChild(insightsGrid);
        }

        function createInsightCard(value, label, color) {
            const card = document.createElement('div');
            card.style.cssText = `
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid ${color};
            `;

            card.innerHTML = `
                <div style="font-size: 1.5rem; font-weight: bold; color: ${color}; margin-bottom: 5px;">${value}</div>
                <div style="font-size: 0.9rem; color: #666;">${label}</div>
            `;

            return card;
        }

        function getPriorityColor(priority) {
            switch (priority) {
                case 'critical': return '#e74c3c';
                case 'high': return '#e67e22';
                case 'medium': return '#f39c12';
                case 'low': return '#27ae60';
                default: return '#95a5a6';
            }
        }
    </script>
</body>
</html>
"""

# Write the HTML template
with open(templates_dir / "dashboard.html", "w") as f:
    f.write(analytics_dashboard_html)

def create_analytics_app() -> FastAPI:
    """Create and return the analytics dashboard FastAPI app"""
    return analytics_app

def start_analytics_dashboard(host: str = "0.0.0.0", port: int = 8012):
    """Start the analytics dashboard server"""
    import uvicorn
    uvicorn.run(analytics_app, host=host, port=port)

if __name__ == "__main__":
    start_analytics_dashboard()
