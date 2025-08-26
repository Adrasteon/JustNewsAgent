import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox
)
import subprocess
import threading
import requests

class DashboardGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dashboard Agent GUI")
        self.setGeometry(100, 100, 800, 600)

        # Robust error logging setup
        import logging
        self.logger = logging.getLogger("DashboardGUI")
        handler = logging.FileHandler("dashboard_gui_error.log", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Monitor thread control flag
        self.monitor_thread_running = True

        # Apply dark theme stylesheet
        dark_grey = "#232323"
        white = "#ffffff"
        self.setStyleSheet(f"""
            QMainWindow {{ background: {dark_grey}; color: {white}; }}
            QWidget {{ background: {dark_grey}; color: {white}; }}
            QTabWidget::pane {{ background: {dark_grey}; }}
            QTabBar::tab {{ background: {dark_grey}; color: {white}; padding: 8px; }}
            QTabBar::tab:selected {{ background: #333333; color: {white}; }}
            QLabel {{ color: {white}; }}
            QPushButton {{ background: #444444; color: {white}; border: 1px solid #666; border-radius: 4px; padding: 4px 12px; }}
            QPushButton:disabled {{ background: #222; color: #888; }}
        """)

        # Create the tab widget and add all tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self.create_monitoring_tab(), "Monitoring")
        self.tabs.addTab(self.create_analysis_tab(), "Analysis")
        self.tabs.addTab(self.create_services_tab(), "Services")
        self.tabs.addTab(self.create_web_crawl_tab(), "Web Crawl")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        # If Monitoring tab is selected, print a status update for all agents
        if self.tabs.tabText(index) == "Monitoring":
            self.print_monitor_status_update()

    def print_monitor_status_update(self):
        import time
        agent_ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]
        agent_names = [
            "MCP Bus", "Chief Editor Agent", "Scout Agent", "Fact Checker Agent", "Analyst Agent",
            "Synthesizer Agent", "Critic Agent", "Memory Agent", "Reasoning Agent", "NewsReader Agent"
        ]
        lines = []
        for name, port in zip(agent_names, agent_ports):
            try:
                if port == 8000:
                    resp = requests.get(f"http://localhost:{port}/agents", timeout=1)
                else:
                    resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    status = "Running"
                else:
                    status = "Stopped"
            except Exception:
                status = "Stopped"
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{ts}] {name} (port {port}): {status}")
        self.append_monitor_output("\n".join(lines))
    def create_web_crawl_tab(self):
        from PyQt5.QtWidgets import QCheckBox
        tab = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Web Crawl Targets")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # List of URLs with checkboxes
        self.crawl_urls = ["https://www.bbc.com/news", "https://www.cnn.com/world", "https://www.reuters.com/news/world"]
        self.crawl_url_checkboxes = []
        self.crawl_url_layout = QVBoxLayout()
        for url in self.crawl_urls:
            row = QHBoxLayout()
            cb = QCheckBox(url)
            cb.setChecked(True)
            cb.setStyleSheet("color: #fff; padding: 4px 8px;")
            cb.stateChanged.connect(lambda state, box=cb: self.update_crawl_url_style(box))
            row.addWidget(cb)
            self.crawl_url_checkboxes.append(cb)
            self.crawl_url_layout.addLayout(row)
        layout.addLayout(self.crawl_url_layout)

        # Add new URL row
        add_row = QHBoxLayout()
        self.new_url_input = QLabel("[Click + to add new URL]")
        self.new_url_input.setStyleSheet("color: #888; font-style: italic;")
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(32)
        add_btn.setStyleSheet("background: #2e7d32; color: #fff; font-weight: bold; font-size: 18px;")
        add_btn.clicked.connect(self.add_new_crawl_url)
        add_row.addWidget(self.new_url_input)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        # Start/Stop Crawl button
        self.crawl_active = False
        self.crawl_toggle_btn = QPushButton("Start Crawl")
        self.crawl_toggle_btn.setStyleSheet("background: #1976d2; color: #fff; font-weight: bold; padding: 8px 24px; border-radius: 6px;")
        self.crawl_toggle_btn.setFixedWidth(140)
        self.crawl_toggle_btn.clicked.connect(self.toggle_crawl)
        layout.addWidget(self.crawl_toggle_btn)

        # Status label
        self.crawl_status_label = QLabel("")
        self.crawl_status_label.setStyleSheet("color: #fff; margin-top: 8px;")
        layout.addWidget(self.crawl_status_label)

        # Spacer
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def update_crawl_url_style(self, cb):
        if cb.isChecked():
            cb.setStyleSheet("color: #fff; padding: 4px 8px;")
        else:
            cb.setStyleSheet("color: #888; padding: 4px 8px;")

    def add_new_crawl_url(self):
        from PyQt5.QtWidgets import QInputDialog, QCheckBox, QHBoxLayout
        url, ok = QInputDialog.getText(self, "Add Crawl Target", "Enter new target URL:")
        if ok and url:
            row = QHBoxLayout()
            cb = QCheckBox(url)
            cb.setChecked(True)
            cb.setStyleSheet("color: #fff; padding: 4px 8px;")
            cb.stateChanged.connect(lambda state, box=cb: self.update_crawl_url_style(box))
            row.addWidget(cb)
            self.crawl_url_checkboxes.append(cb)
            self.crawl_url_layout.addLayout(row)
            self.crawl_urls.append(url)

    def toggle_crawl(self):
        import threading
        self.crawl_active = not self.crawl_active
        if self.crawl_active:
            self.crawl_toggle_btn.setText("Stop Crawl")
            self.crawl_toggle_btn.setStyleSheet("background: #c62828; color: #fff; font-weight: bold; padding: 8px 24px; border-radius: 6px;")
            self.crawl_status_label.setText("Crawling started for selected targets.")
            # Get selected URLs
            selected_urls = [cb.text() for cb in self.crawl_url_checkboxes if cb.isChecked()]
            if not selected_urls:
                self.crawl_status_label.setText("No URLs selected.")
                self.crawl_active = False
                self.crawl_toggle_btn.setText("Start Crawl")
                self.crawl_toggle_btn.setStyleSheet("background: #1976d2; color: #fff; font-weight: bold; padding: 8px 24px; border-radius: 6px;")
                return
            # Log crawl start in monitor
            self.append_monitor_output(f"[Crawl] Starting crawl for: {', '.join(selected_urls)}")
            # Start crawl in background thread
            self.crawl_threads = []
            self.crawl_stats = {url: {"crawled": 0, "articles": 0, "last": None} for url in selected_urls}
            for url in selected_urls:
                t = threading.Thread(target=self.start_scout_crawl, args=(url,), daemon=True)
                t.start()
                self.crawl_threads.append(t)
            # Start polling for crawl stats
            self.crawl_polling = True
            self.poll_crawl_stats()
        else:
            self.crawl_toggle_btn.setText("Start Crawl")
            self.crawl_toggle_btn.setStyleSheet("background: #1976d2; color: #fff; font-weight: bold; padding: 8px 24px; border-radius: 6px;")
            self.crawl_status_label.setText("Crawling stopped.")
            self.append_monitor_output("[Crawl] Crawl stopped.")
            self.crawl_polling = False

    def start_scout_crawl(self, url):
        import requests
        import time
        try:
            # Call Scout Agent's enhanced_deep_crawl_site endpoint
            payload = {
                "args": [],
                "kwargs": {
                    "url": url,
                    "max_depth": 3,
                    "max_pages": 100,
                    "word_count_threshold": 500,
                    "quality_threshold": 0.6,
                    "analyze_content": True
                }
            }
            resp = requests.post("http://localhost:8002/enhanced_deep_crawl_site", json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                articles_found = len(result) if isinstance(result, list) else 0
                self.crawl_stats[url]["crawled"] = self.crawl_stats[url].get("crawled", 0) + 1
                self.crawl_stats[url]["articles"] = articles_found
                self.crawl_stats[url]["last"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.append_monitor_output(f"[Crawl] Finished crawl for {url}: {articles_found} articles found.")
            else:
                self.append_monitor_output(f"[Crawl] Error crawling {url}: {resp.status_code}")
        except Exception as e:
            self.append_monitor_output(f"[Crawl] Exception crawling {url}: {e}")

    def poll_crawl_stats(self):
        import requests
        import time
        if not getattr(self, "crawl_polling", False):
            return
        try:
            # For each URL, poll Scout Agent for crawl stats (simulate with get_production_crawler_info)
            resp = requests.post("http://localhost:8002/get_production_crawler_info", json={"args": [], "kwargs": {}})
            if resp.status_code == 200:
                info = resp.json()
                # Try to extract stats for each site
                for url in self.crawl_stats:
                    # Try to match by domain or site name
                    site_name = None
                    for site in info.get("supported_sites", []):
                        if site in url:
                            site_name = site
                            break
                    if site_name:
                        site_info = info.get("site_details", {}).get(site_name, {})
                        crawled = site_info.get("pages_crawled", 0)
                        articles = site_info.get("articles_found", 0)
                        self.crawl_stats[url]["crawled"] = crawled
                        self.crawl_stats[url]["articles"] = articles
                        self.crawl_stats[url]["last"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.append_monitor_output(f"[Crawl] Progress for {url}: {crawled} pages crawled, {articles} articles found.")
            else:
                self.append_monitor_output(f"[Crawl] Error polling crawl stats: {resp.status_code}")
        except Exception as e:
            self.append_monitor_output(f"[Crawl] Exception polling crawl stats: {e}")
        # Schedule next poll in 2 seconds
        from PyQt5.QtCore import QTimer
        if getattr(self, "crawl_polling", False):
            QTimer.singleShot(2000, self.poll_crawl_stats)


    def create_services_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # --- Top Spacer (one line) ---
        layout.addSpacing(10)

        # --- Start/Stop All Services Row ---
        all_row = QHBoxLayout()
        all_label = QLabel("All Agents")
        all_label.setStyleSheet("font-weight: bold;")
        all_status = QLabel("")
        all_start_btn = QPushButton("Start All")
        all_stop_btn = QPushButton("Stop All")
        all_start_btn.setFixedWidth(60)  # Half the previous default width (was default, now 60)
        all_stop_btn.setFixedWidth(60)
        all_start_btn.clicked.connect(self.start_all_agents)
        all_stop_btn.clicked.connect(self.stop_all_agents)
        all_row.addWidget(all_label)
        all_row.addWidget(all_status)
        all_row.addWidget(all_start_btn)
        all_row.addWidget(all_stop_btn)
        layout.addLayout(all_row)

        # --- Extra Spacer between All Agents and rest ---
        layout.addSpacing(20)

        self.agent_info = [
            ("MCP Bus", 8000),
            ("Chief Editor Agent", 8001),
            ("Scout Agent", 8002),
            ("Fact Checker Agent", 8003),
            ("Analyst Agent", 8004),
            ("Synthesizer Agent", 8005),
            ("Critic Agent", 8006),
            ("Memory Agent", 8007),
            ("Reasoning Agent", 8008),
            ("NewsReader Agent", 8009),
        ]

        self.agent_buttons = {}
        self.all_status_label = all_status

        for name, port in self.agent_info:
            row = QHBoxLayout()
            label = QLabel(f"{name} (port {port})")
            status_label = QLabel("Checking...")
            start_btn = QPushButton("Start")
            stop_btn = QPushButton("Stop")
            start_btn.setFixedWidth(60)
            stop_btn.setFixedWidth(60)
            start_btn.clicked.connect(lambda _, n=name: self.start_agent(n))
            stop_btn.clicked.connect(lambda _, n=name: self.stop_agent(n))
            row.addWidget(label)
            row.addWidget(status_label)
            row.addWidget(start_btn)
            row.addWidget(stop_btn)
            layout.addLayout(row)
            self.agent_buttons[name] = (status_label, start_btn, stop_btn, port)

            self.agent_activity = {}  # name: (activity_label, last_activity_label)

            for name, port in self.agent_info:
                row = QHBoxLayout()
                label = QLabel(f"{name} (port {port})")
                status_label = QLabel("Checking...")
                activity_label = QLabel("●")
                activity_label.setStyleSheet("color: #888; font-size: 18px; margin-left: 8px;")
                last_activity_label = QLabel("")
                last_activity_label.setStyleSheet("color: #aaa; font-size: 11px; margin-left: 8px;")
                start_btn = QPushButton("Start")
                stop_btn = QPushButton("Stop")
                start_btn.setFixedWidth(60)
                stop_btn.setFixedWidth(60)
                start_btn.clicked.connect(lambda _, n=name: self.start_agent(n))
                stop_btn.clicked.connect(lambda _, n=name: self.stop_agent(n))
                row.addWidget(label)
                row.addWidget(status_label)
                row.addWidget(activity_label)
                row.addWidget(last_activity_label)
                row.addWidget(start_btn)
                row.addWidget(stop_btn)
                layout.addLayout(row)
                self.agent_buttons[name] = (status_label, start_btn, stop_btn, port)
                self.agent_activity[name] = (activity_label, last_activity_label)
        # Initial status check
        threading.Thread(target=self.update_all_status, daemon=True).start()

        tab.setLayout(layout)
        return tab

    def start_all_agents(self):
        self.all_status_label.setText("Starting...")
        self.all_status_label.setStyleSheet("color: orange;")
        for name in self.agent_buttons:
            status_label, start_btn, stop_btn, _ = self.agent_buttons[name]
            status_label.setText("Starting...")
            status_label.setStyleSheet("color: orange;")
            start_btn.setEnabled(False)
            stop_btn.setEnabled(False)
        threading.Thread(target=self._start_all_agents_thread, daemon=True).start()

    def _start_all_agents_thread(self):
        subprocess.call(["/bin/bash", "./start_services_daemon.sh"])
        self.update_all_status()
        self.all_status_label.setText("")

    def stop_all_agents(self):
        self.all_status_label.setText("Stopping...")
        self.all_status_label.setStyleSheet("color: orange;")
        for name in self.agent_buttons:
            status_label, start_btn, stop_btn, _ = self.agent_buttons[name]
            status_label.setText("Stopping...")
            status_label.setStyleSheet("color: orange;")
            start_btn.setEnabled(False)
            stop_btn.setEnabled(False)
        threading.Thread(target=self._stop_all_agents_thread, daemon=True).start()

    def _stop_all_agents_thread(self):
        subprocess.call(["/bin/bash", "./stop_services.sh"])
        self.update_all_status()
        self.all_status_label.setText("")

    def update_all_status(self):
        for name, (_, _, _, port) in self.agent_buttons.items():
            self.update_agent_status(name, port)

    def update_agent_status(self, name, port):
        import time
        status_label, start_btn, stop_btn, _ = self.agent_buttons[name]
        status_label.setText("Checking")
        status_label.setStyleSheet("color: orange;")
        start_btn.setEnabled(False)
        stop_btn.setEnabled(False)
        success = False
        for attempt in range(3):
            try:
                if port == 8000:
                    resp = requests.get(f"http://localhost:{port}/agents", timeout=1)
                else:
                    resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    status_label.setText("Running")
                    status_label.setStyleSheet("color: green;")
                    start_btn.setEnabled(False)
                    stop_btn.setEnabled(True)
                    success = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not success:
            status_label.setText("Stopped")
            status_label.setStyleSheet("color: red;")
            start_btn.setEnabled(True)
            stop_btn.setEnabled(False)
        activity_label, last_activity_label = self.agent_activity[name]
        activity_label.setStyleSheet("color: #888; font-size: 18px; margin-left: 8px;")
        activity_label.setText("●")
        last_activity_label.setText("")

    def start_agent(self, name):
        # Start the agent using the start_services_daemon.sh script
        msg = QMessageBox()
        msg.setWindowTitle("Start Agent")
        msg.setText(f"Starting {name}... (this may take a few seconds)")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        threading.Thread(target=self._start_agent_thread, args=(name,), daemon=True).start()

    def _start_agent_thread(self, name):
        # Map display name to script argument
        agent_map = {
            "MCP Bus": "mcp_bus",
            "Chief Editor Agent": "chief_editor",
            "Scout Agent": "scout",
            "Fact Checker Agent": "fact_checker",
            "Analyst Agent": "analyst",
            "Synthesizer Agent": "synthesizer",
            "Critic Agent": "critic",
            "Memory Agent": "memory",
            "Reasoning Agent": "reasoning",
            "NewsReader Agent": "newsreader",
        }
        script_arg = agent_map.get(name)
        if script_arg:
            subprocess.call(["/bin/bash", "./start_services_daemon.sh", script_arg])
        # Wait a bit and update status
        self.update_agent_status(name, self.agent_buttons[name][3])

    def stop_agent(self, name):
        # Stop the agent using the stop_services.sh script
        msg = QMessageBox()
        msg.setWindowTitle("Stop Agent")
        msg.setText(f"Stopping {name}... (this may take a few seconds)")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        threading.Thread(target=self._stop_agent_thread, args=(name,), daemon=True).start()

    def _stop_agent_thread(self, name):
        agent_map = {
            "MCP Bus": "mcp_bus",
            "Chief Editor Agent": "chief_editor",
            "Scout Agent": "scout",
            "Fact Checker Agent": "fact_checker",
            "Analyst Agent": "analyst",
            "Synthesizer Agent": "synthesizer",
            "Critic Agent": "critic",
            "Memory Agent": "memory",
            "Reasoning Agent": "reasoning",
            "NewsReader Agent": "newsreader",
            "Dashboard Agent": "dashboard",
        }
        script_arg = agent_map.get(name)
        if script_arg:
            subprocess.call(["/bin/bash", "./stop_services.sh", script_arg])
        # Wait a bit and update status
        self.update_agent_status(name, self.agent_buttons[name][3])

    def create_monitoring_tab(self):
        from PyQt5.QtWidgets import QTextEdit
        tab = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Real-time Agent Activity Monitor")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Scrollable output pane
        self.monitor_output = QTextEdit()
        self.monitor_output.setReadOnly(True)
        self.monitor_output.setStyleSheet("background: #181818; color: #fff; font-family: monospace; font-size: 13px;")
        self.monitor_output.setMinimumHeight(300)
        layout.addWidget(self.monitor_output)

        tab.setLayout(layout)

        # Start real-time monitoring thread with robust error handling
        self.monitor_thread = threading.Thread(target=self.update_monitor_output, daemon=True)
        self.monitor_thread.start()

        return tab

    def update_monitor_output(self):
        import time
        agent_ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]
        agent_names = [
            "MCP Bus", "Chief Editor Agent", "Scout Agent", "Fact Checker Agent", "Analyst Agent",
            "Synthesizer Agent", "Critic Agent", "Memory Agent", "Reasoning Agent", "NewsReader Agent"
        ]
        last_status = {}
        try:
            # Print initial status for all agents
            initial_lines = []
            for name, port in zip(agent_names, agent_ports):
                try:
                    if port == 8000:
                        resp = requests.get(f"http://localhost:{port}/agents", timeout=1)
                    else:
                        resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                    if resp.status_code == 200:
                        status = "Running"
                    else:
                        status = "Stopped"
                except Exception as e:
                    status = "Stopped"
                    self.logger.warning(f"Initial status check failed for {name} (port {port}): {e}")
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                initial_lines.append(f"[{ts}] {name} (port {port}): Initial status: {status}")
                last_status[name] = status
            self.append_monitor_output("\n".join(initial_lines))
            # Now only log status changes
            while self.monitor_thread_running:
                output_lines = []
                for name, port in zip(agent_names, agent_ports):
                    try:
                        if port == 8000:
                            resp = requests.get(f"http://localhost:{port}/agents", timeout=1)
                        else:
                            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                        if resp.status_code == 200:
                            status = "Running"
                        else:
                            status = "Stopped"
                    except Exception as e:
                        status = "Stopped"
                        self.logger.warning(f"Status check failed for {name} (port {port}): {e}")
                    prev = last_status.get(name)
                    if prev != status:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        output_lines.append(f"[{ts}] {name} (port {port}): {status}")
                        last_status[name] = status
                if output_lines:
                    try:
                        self.append_monitor_output("\n".join(output_lines))
                    except Exception as e:
                        self.logger.error(f"Error updating monitor output: {e}")
                time.sleep(2)
        except Exception as e:
            self.logger.error(f"Monitor thread crashed: {e}")

    def append_monitor_output(self, text):
        # Append text to the monitor output pane in a thread-safe and robust way
        from PyQt5.QtCore import QTimer
        def append():
            try:
                if hasattr(self, 'monitor_output') and self.monitor_output is not None:
                    self.monitor_output.append(text)
                    self.monitor_output.moveCursor(self.monitor_output.textCursor().End)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error in append_monitor_output: {e}")
        try:
            QTimer.singleShot(0, append)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"QTimer.singleShot failed in append_monitor_output: {e}")
    def closeEvent(self, event):
        # Gracefully stop monitor thread on close
        self.monitor_thread_running = False
        if hasattr(self, 'logger'):
            self.logger.info("Dashboard GUI closed. Monitor thread stopped.")
        super().closeEvent(event)

    def create_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Example content for Analysis tab
        layout.addWidget(QLabel("Tools for analyzing articles will be displayed here."))

        tab.setLayout(layout)
        return tab

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Example content for Settings tab
        layout.addWidget(QLabel("Configuration options for the Dashboard Agent will be displayed here."))

        tab.setLayout(layout)
        return tab

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DashboardGUI()
    gui.show()
    sys.exit(app.exec_())
