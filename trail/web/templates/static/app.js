/** Trail REPL Web Console — Frontend Application */

const API = '/api/sessions';

class TrailApp {
  constructor() {
    this.currentSession = null;
    this.sessionList = [];
    this.history = [];
    this.historyIndex = -1;
    this.input = document.getElementById('input');
    this.output = document.getElementById('output');
    this.loading = document.getElementById('loading');
    this.outputPanel = document.getElementById('output-panel');
    this.panelContent = document.getElementById('panel-content');
    this.panelTitle = document.getElementById('panel-title');
    this.sidebar = document.getElementById('sidebar');
    this.sessionListEl = document.getElementById('session-list');
    this.setupInput();
    this.init();
  }

  async init() {
    this.showSystemMessage('Trail REPL Web Console — use trail repl for terminal, or keep here.');
    await this.newSession();
  }

  setupInput() {
    this.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.submit();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.historyNav(-1);
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.historyNav(1);
      } else if (e.key === 'Tab') {
        e.preventDefault();
        this.tabComplete();
      } else if (e.key === '/') {
        e.preventDefault();
        this.showHelp();
      }
    });

    // Keep focus on input
    document.getElementById('terminal').addEventListener('click', () => {
      this.input.focus();
    });
  }

  async newSession() {
    try {
      const res = await fetch(API, { method: 'POST' });
      const data = await res.json();
      this.currentSession = data.session_id;
      this.history = [];
      this.historyIndex = -1;
      this.output.innerHTML = '';
      this.showSystemMessage(`New session: ${this.currentSession}`);
      await this.refreshSessionList();
      this.input.focus();
    } catch (err) {
      this.showError('Failed to create session: ' + err.message);
    }
  }

  async submit() {
    const text = this.input.value.trim();
    if (!text || !this.currentSession) return;

    this.input.value = '';
    this.appendInputLine(text);
    this.loading.classList.remove('hidden');

    try {
      const res = await fetch(`${API}/${this.currentSession}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: text })
      });

      const data = await res.json();
      this.loading.classList.add('hidden');

      if (res.ok) {
        this.appendResult(data);
        this.history.push(text);
        this.historyIndex = this.history.length;
        await this.refreshSessionList();
      } else {
        this.showError(data.error || 'Server error');
      }
    } catch (err) {
      this.loading.classList.add('hidden');
      this.showError('Connection error: ' + err.message);
    }
  }

  historyNav(dir) {
    if (this.history.length === 0) return;
    this.historyIndex += dir;
    if (this.historyIndex < 0) this.historyIndex = 0;
    if (this.historyIndex >= this.history.length) this.historyIndex = this.history.length;

    if (this.historyIndex < this.history.length) {
      this.input.value = this.history[this.historyIndex];
    } else {
      this.input.value = '';
    }
  }

  async tabComplete() {
    if (!this.currentSession) return;
    const cursorPos = this.input.selectionStart;
    const before = this.input.value.substring(0, cursorPos);
    const word = before.split(/[\s(,]/).pop() || '';

    try {
      const res = await fetch(`${API}/${this.currentSession}/info`);
      const data = await res.json();
      const defs = data.definitions || [];

      // Build completion list: functions + fields + user defs
      const completions = [
        'income.revenue', 'income.operating_income', 'income.operating_margin',
        'balance.total_assets', 'balance.total_equity', 'balance.total_debt',
        'cash.free_cash_flow', 'price.adj_close',
        ...defs
      ];

      const matches = completions.filter(c => c.startsWith(word));
      if (matches.length === 1) {
        const prefix = before.substring(0, before.length - word.length);
        this.input.value = prefix + matches[0];
      }
    } catch (e) {
      // Silently fail — no completions
    }
  }

  showHelp() {
    this.showSystemMessage([
      'Trail REPL Commands:',
      '',
      '  Expressions:   income.revenue / balance.total_assets',
      '  Assignments:   margin = income.revenue / balance.total_assets',
      '  Functions:     def my_func(x) = x * 2',
      '  Models:        model m { export score = 1 if revenue > 100 else 0 }',
      '  Catalog:       ? functions | ? sources | ? fields | ? fields income',
      '',
      '  / - show this help',
      '  ↑/↓ - history navigation',
      '  Tab - autocomplete',
      ''
    ].join('\n'));
  }

  appendInputLine(text) {
    const div = document.createElement('div');
    div.className = 'output-line input-line';
    div.innerHTML = `<span class="prompt">trail&gt;</span> <span class="input-text">${this.escapeHtml(text)}</span>`;
    this.output.appendChild(div);
    this.scrollBottom();
  }

  appendResult(data) {
    if (data.type === 'error') {
      this.showError(data.error || 'Unknown error');
    } else if (data.type === 'result' && data.output) {
      this.showResultTable(data.output);
    } else {
      // noop - def, model, signal declarations
      const div = document.createElement('div');
      div.className = 'output-line noop-line';
      div.innerHTML = '<span class="noop-text">✓ (definition stored)</span>';
      this.output.appendChild(div);
      this.scrollBottom();
    }
  }

  showResultTable(data) {
    const { rows, columns, preview } = data;
    const displayRows = preview || [];

    // Show compact table in output
    const tableDiv = document.createElement('div');
    tableDiv.className = 'output-line result-line';

    // Row count
    const countDiv = document.createElement('div');
    countDiv.className = 'row-count';
    countDiv.textContent = `(${rows} rows, ${columns.length} columns)`;
    this.output.appendChild(countDiv);

    // Mini table preview (first 8 rows, max 6 columns)
    const maxCols = Math.min(columns.length, 6);
    const maxRows = Math.min(displayRows.length, 8);

    let tableHtml = '<table class="result-table"><thead><tr>';
    for (let i = 0; i < maxCols; i++) {
      tableHtml += `<th>${this.escapeHtml(columns[i])}</th>`;
    }
    tableHtml += '</tr></thead><tbody>';

    for (let i = 0; i < maxRows; i++) {
      tableHtml += '<tr>';
      for (let j = 0; j < maxCols; j++) {
        const val = displayRows[i]?.[columns[j]];
        const display = val === undefined ? '' : String(val);
        tableHtml += `<td>${this.escapeHtml(display)}</td>`;
      }
      tableHtml += '</tr>';
    }

    if (rows > maxRows) {
      tableHtml += `<tr><td class="truncated" colspan="${maxCols}">... ${rows - maxRows} more rows</td></tr>`;
    }

    tableHtml += '</tbody></table>';
    tableDiv.innerHTML = tableHtml;
    this.output.appendChild(tableDiv);

    // Click to expand full table
    const expandBtn = document.createElement('button');
    expandBtn.className = 'row-count';
    expandBtn.textContent = `[Click to expand all ${rows} rows]`;
    expandBtn.style.cursor = 'pointer';
    expandBtn.style.color = '#58a6ff';
    expandBtn.style.borderBottom = '1px dashed #58a6ff';
    expandBtn.style.display = 'inline-block';
    expandBtn.style.marginTop = '4px';
    expandBtn.onclick = () => this.expandTable(data);
    this.output.appendChild(expandBtn);

    this.scrollBottom();
  }

  async expandTable(data) {
    const { rows, columns, preview } = data;
    this.panelTitle.textContent = `Results — ${rows} rows, ${columns.length} columns`;

    let tableHtml = '<table class="panel-table"><thead><tr>';
    for (const col of columns) {
      tableHtml += `<th>${this.escapeHtml(col)}</th>`;
    }
    tableHtml += '</tr></thead><tbody>';

    const displayRows = preview || [];
    for (const row of displayRows) {
      tableHtml += '<tr>';
      for (const col of columns) {
        const val = row?.[col];
        const display = val === undefined ? '' : String(val);
        tableHtml += `<td>${this.escapeHtml(display)}</td>`;
      }
      tableHtml += '</tr>';
    }

    if (rows > displayRows.length) {
      tableHtml += `<tr><td colspan="${columns.length}" style="color:#8b949e;font-style:italic">... and ${rows - displayRows.length} more rows (preview)</td></tr>`;
    }

    tableHtml += '</tbody></table>';
    this.panelContent.innerHTML = tableHtml;
    this.outputPanel.classList.remove('hidden');
  }

  closePanel() {
    this.outputPanel.classList.add('hidden');
  }

  clearOutput() {
    this.output.innerHTML = '';
    this.history = [];
    this.historyIndex = -1;
    this.closePanel();
  }

  async refreshSessionList() {
    try {
      const res = await fetch(API);
      this.sessionList = await res.json();
      this.renderSessionList();
    } catch (err) {
      console.error('Failed to refresh session list:', err);
    }
  }

  renderSessionList() {
    this.sessionListEl.innerHTML = this.sessionList.map(s => `
      <div class="session-item ${s.session_id === this.currentSession ? 'active' : ''}"
           onclick="app.switchSession('${s.session_id}')">
        <div class="sid">${s.session_id}</div>
        <div class="meta">${s.history_count} commands · ${s.definitions.length} defs</div>
        <div class="defs">${s.definitions.join(', ') || 'no definitions'}</div>
        <button class="session-delete-btn" onclick="event.stopPropagation(); app.deleteSession('${s.session_id}')">✕</button>
      </div>
    `).join('');
  }

  async switchSession(sid) {
    if (sid === this.currentSession) return;

    try {
      const res = await fetch(`${API}/${sid}/history`);
      const history = await res.json();

      this.currentSession = sid;
      this.output.innerHTML = '';
      this.history = [];
      this.historyIndex = -1;

      // Replay history
      for (const entry of history) {
        this.appendInputLine(entry.input);
        if (entry.type === 'error') {
          this.showError(entry.error);
        } else if (entry.type === 'result' && entry.output) {
          this.showResultTable(entry.output);
        } else {
          const div = document.createElement('div');
          div.className = 'output-line noop-line';
          div.innerHTML = '<span class="noop-text">✓ (definition stored)</span>';
          this.output.appendChild(div);
        }
      }

      this.scrollBottom();
      this.renderSessionList();
      this.input.focus();
    } catch (err) {
      this.showError('Failed to switch session: ' + err.message);
    }
  }

  async deleteSession(sid) {
    try {
      await fetch(`${API}/${sid}/delete`, { method: 'POST' });
      if (sid === this.currentSession) {
        await this.newSession();
      } else {
        await this.refreshSessionList();
      }
    } catch (err) {
      this.showError('Failed to delete session: ' + err.message);
    }
  }

  toggleSessionList() {
    this.sidebar.classList.toggle('hidden');
  }

  showError(text) {
    const div = document.createElement('div');
    div.className = 'output-line error-line';
    div.innerHTML = `<span class="error-text">✕ ${this.escapeHtml(text)}</span>`;
    this.output.appendChild(div);
    this.scrollBottom();
  }

  showSystemMessage(text) {
    const div = document.createElement('div');
    div.className = 'output-line system-line';
    div.innerHTML = `<span class="system-text">${this.escapeHtml(text)}</span>`;
    this.output.appendChild(div);
    this.scrollBottom();
  }

  scrollBottom() {
    this.output.scrollTop = this.output.scrollHeight;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new TrailApp();
});
