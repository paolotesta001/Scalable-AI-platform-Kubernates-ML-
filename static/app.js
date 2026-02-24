/* ============================================================
   Smart Nutrition Tracker — Web Interface
   Vanilla JS single-page app with A2A/PNP protocol switching
   ============================================================ */

// ---- STATE ----
const APP = {
    protocol: 'a2a',
    userId: 1,
    messages: [],
    userProfile: null,
    meals: [],
    currentPage: 'chat',
    loading: false,
    sortField: 'meal_date',
    sortDir: 'desc',
};

// ---- API LAYER ----
const API = {
    async fetchJSON(url, opts = {}) {
        const resp = await fetch(url, opts);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
    },

    async sendChat(text) {
        if (APP.protocol === 'pnp') return this.sendPNP(text);
        if (APP.protocol === 'toon') return this.sendTOON(text);
        return this.sendA2A(text);
    },

    async sendA2A(text) {
        return this.fetchJSON('/a2a/messages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sender: { id: 'user', role: 'user' },
                type: 'query',
                payload: { text, user_id: APP.userId },
            }),
        });
    },

    async sendPNP(text) {
        return this.fetchJSON('/pnp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                src: 'user',
                t: 'q',
                p: { text, user_id: APP.userId },
            }),
        });
    },

    async sendTOON(text) {
        return this.fetchJSON('/toon', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                src: 'user',
                kind: 'q',
                body: { text, user_id: APP.userId },
            }),
        });
    },

    getUser: (id) => API.fetchJSON(`/api/user/${id}`),
    getMeals: (id, limit = 50, type = '') => {
        let url = `/api/meals/${id}?limit=${limit}`;
        if (type) url += `&meal_type=${type}`;
        return API.fetchJSON(url);
    },
    getDaily: (id) => API.fetchJSON(`/api/daily/${id}`),
    createUser: (data) => API.fetchJSON('/api/user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    }),
    getAllUsers: () => API.fetchJSON('/api/users'),
    updateUser: (id, data) => API.fetchJSON(`/api/user/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    }),
    deleteUser: (id) => API.fetchJSON(`/api/user/${id}`, { method: 'DELETE' }),
    updateMeal: (id, data) => API.fetchJSON(`/api/meal/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    }),
    deleteMeal: (id) => API.fetchJSON(`/api/meal/${id}`, { method: 'DELETE' }),
    getTraces: (limit = 10) => API.fetchJSON(`/api/traces?limit=${limit}`),
    getTrace: (id) => API.fetchJSON(`/api/trace/${id}`),
    getAgents: () => API.fetchJSON('/agents'),
    getMetrics: () => API.fetchJSON('/metrics'),
    getBenchmark: () => API.fetchJSON('/api/benchmark-results'),
};

// ---- RESPONSE NORMALIZER ----
function normalizeResponse(raw) {
    if (APP.protocol === 'a2a') {
        const p = raw.payload || {};
        return {
            text: p.text || '',
            workflow: p.workflow || '',
            agentsUsed: p.agents_used || [],
            performance: p.performance || {},
        };
    }
    const p = raw.p || {};
    return {
        text: p.text || '',
        workflow: p.workflow || '',
        agentsUsed: [],
        performance: p.performance || {},
    };
}

// ---- UTILITIES ----
function fmt(n) { return n != null ? Number(n).toFixed(1) : '-'; }

function simpleMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function showToast(msg, type = 'success') {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

// ---- NAVIGATION ----
function navigateTo(page) {
    APP.currentPage = page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');
    document.querySelectorAll('.nav-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.page === page));

    if (page === 'dashboard') loadDashboard();
    if (page === 'meals') loadMeals();
    if (page === 'profile') loadProfile();
}

function setProtocol(proto) {
    APP.protocol = proto;
    document.querySelectorAll('.toggle-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.protocol === proto));
}

// ---- CHAT ----
function addChatMessage(role, text, meta = {}) {
    const container = document.getElementById('chat-messages');
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble chat-${role}` + (meta.error ? ' error' : '');
    bubble.innerHTML = `<div class="bubble-content">${simpleMarkdown(text)}</div>`;

    if (role === 'agent' && !meta.error) {
        const footer = document.createElement('div');
        footer.className = 'bubble-footer';
        const parts = [];
        if (meta.workflow) parts.push(`<span class="tag">${meta.workflow}</span>`);
        if (meta.agentsUsed?.length) parts.push(meta.agentsUsed.join(', '));
        if (meta.time) parts.push(`${(meta.time / 1000).toFixed(1)}s`);
        if (meta.protocol) parts.push(`via ${meta.protocol.toUpperCase()}`);
        if (meta.performance?.total_tokens) parts.push(`${meta.performance.total_tokens} tok`);
        footer.innerHTML = parts.join(' &middot; ');
        bubble.appendChild(footer);
    }

    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

function showTyping() {
    const el = document.createElement('div');
    el.id = 'typing';
    el.className = 'typing-indicator';
    el.innerHTML = '<span>.</span><span>.</span><span>.</span>';
    document.getElementById('chat-messages').appendChild(el);
    el.scrollIntoView();
}

function hideTyping() {
    const el = document.getElementById('typing');
    if (el) el.remove();
}

async function handleChatSend() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text || APP.loading) return;

    addChatMessage('user', text);
    input.value = '';
    APP.loading = true;
    document.getElementById('chat-send').disabled = true;
    showTyping();

    try {
        const t0 = performance.now();
        const raw = await API.sendChat(text);
        const elapsed = performance.now() - t0;
        const data = normalizeResponse(raw);
        hideTyping();
        addChatMessage('agent', data.text, {
            workflow: data.workflow,
            agentsUsed: data.agentsUsed,
            time: elapsed,
            protocol: APP.protocol,
            performance: data.performance,
        });
    } catch (err) {
        hideTyping();
        addChatMessage('agent', `Error: ${err.message}`, { error: true });
    } finally {
        APP.loading = false;
        document.getElementById('chat-send').disabled = false;
        input.focus();
    }
}

// ---- DASHBOARD ----
async function loadDashboard() {
    const [agents, daily, meals, metrics, traces, registry] = await Promise.allSettled([
        API.getAgents(),
        API.getDaily(APP.userId),
        API.getMeals(APP.userId, 10),
        API.getMetrics(),
        API.getTraces(10),
        API.fetchJSON('/api/registry'),
    ]);

    const regMap = {};
    if (registry.status === 'fulfilled') {
        (registry.value.agents || []).forEach(a => { regMap[a.agent_id] = a; });
    }
    if (agents.status === 'fulfilled') renderAgentStatus(agents.value, regMap);
    if (daily.status === 'fulfilled') renderNutrition(daily.value);
    if (meals.status === 'fulfilled') renderRecentMeals(meals.value);
    if (metrics.status === 'fulfilled') renderMetrics(metrics.value);
    if (traces.status === 'fulfilled') renderTraceList(traces.value);
}

const AGENT_NAMES = {
    food_logger: { name: 'Food Logger', emoji: '📸' },
    meal_planner: { name: 'Meal Planner', emoji: '🍽' },
    health_advisor: { name: 'Health Advisor', emoji: '💪' },
    db_writer: { name: 'DB Writer', emoji: '🗄' },
};

function renderAgentStatus(data, regMap = {}) {
    const el = document.getElementById('agent-status-list');
    let html = '';
    // Orchestrator
    const orchOnline = data.orchestrator === 'online';
    const orchReg = regMap['orchestrator'];
    html += `<div class="agent-row">
        <div class="status-dot ${orchOnline ? 'online' : 'offline'}"></div>
        <span class="agent-name">Orchestrator</span>
        <span class="agent-reg-badge ${orchReg?.source === 'dynamic' ? 'dynamic' : 'static'}">${orchReg?.source || 'static'}</span>
        <span class="agent-status-text">${orchOnline ? 'online' : 'offline'}</span>
    </div>`;
    // Sub-agents
    for (const [id, info] of Object.entries(data.agents || {})) {
        const a = AGENT_NAMES[id] || { name: id, emoji: '?' };
        const online = info.status === 'ok';
        const reg = regMap[id];
        const caps = reg?.capabilities?.length ? reg.capabilities.join(', ') : '';
        const source = reg?.source || 'static';
        html += `<div class="agent-row">
            <div class="status-dot ${online ? 'online' : 'offline'}"></div>
            <span class="agent-name">${a.name}</span>
            <span class="agent-reg-badge ${source}">${source}</span>
            <span class="agent-status-text">${online ? 'online' : 'offline'}</span>
        </div>`;
        if (caps) {
            html += `<div class="agent-caps">${caps}</div>`;
        }
    }
    el.innerHTML = html;
}

function renderNutrition(data) {
    const el = document.getElementById('nutrition-bars');
    const r = data?.result || {};
    const target = APP.userProfile?.daily_cal_target || 2000;
    const currentCal = r.total_calories || 0;

    // Calorie goal alert
    let alertHtml = '';
    if (currentCal > target) {
        const over = (currentCal - target).toFixed(0);
        alertHtml = `<div class="calorie-alert">You've exceeded your daily calorie target by ${over} kcal! (${fmt(currentCal)} / ${target} kcal)</div>`;
    } else if (currentCal > target * 0.9 && currentCal > 0) {
        const remaining = (target - currentCal).toFixed(0);
        alertHtml = `<div class="calorie-warning">Almost at your limit! ${remaining} kcal remaining.</div>`;
    }

    const nutrients = [
        { label: 'Calories', cur: currentCal, max: target, unit: 'kcal', color: currentCal > target ? 'var(--danger)' : 'var(--accent)' },
        { label: 'Protein', cur: r.total_protein || 0, max: Math.round(target * 0.25 / 4), unit: 'g', color: 'var(--info)' },
        { label: 'Carbs', cur: r.total_carbs || 0, max: Math.round(target * 0.50 / 4), unit: 'g', color: 'var(--warning)' },
        { label: 'Fat', cur: r.total_fat || 0, max: Math.round(target * 0.25 / 9), unit: 'g', color: 'var(--danger)' },
    ];

    el.innerHTML = alertHtml + nutrients.map(n => {
        const pct = Math.min(100, n.max > 0 ? (n.cur / n.max) * 100 : 0);
        return `<div class="nutrient-bar">
            <div class="nutrient-header">
                <span class="nutrient-label">${n.label}</span>
                <span class="nutrient-value">${fmt(n.cur)} / ${n.max} ${n.unit}</span>
            </div>
            <div class="nutrient-track">
                <div class="nutrient-fill" style="width:${pct}%;background:${n.color}"></div>
            </div>
        </div>`;
    }).join('');
}

function renderRecentMeals(data) {
    const el = document.getElementById('recent-meals-list');
    const meals = data?.result || [];
    if (!meals.length) {
        el.innerHTML = '<p class="empty-state">No meals logged yet</p>';
        return;
    }
    el.innerHTML = meals.map(m => `
        <div class="meal-item">
            <span class="meal-name">${m.food_name || '?'}</span>
            <span class="meal-type-badge">${m.meal_type || '?'}</span>
            <span class="meal-cal">${fmt(m.calories)} kcal</span>
        </div>
    `).join('');
}

function renderMetrics(data) {
    const el = document.getElementById('system-metrics');
    const mem = data?.memory || {};
    el.innerHTML = `
        <div class="metric-row"><span>RSS Memory</span><span class="metric-value">${mem.rss_mb || 0} MB</span></div>
        <div class="metric-row"><span>VMS Memory</span><span class="metric-value">${mem.vms_mb || 0} MB</span></div>
        <div class="metric-row"><span>Active Agents</span><span class="metric-value">${(data?.agents || []).length}</span></div>
        <div class="metric-row"><span>Protocol</span><span class="metric-value">${APP.protocol.toUpperCase()}</span></div>
    `;
}

// ---- AGENT COMMUNICATION / TRACES ----
const AGENT_DISPLAY = {
    orchestrator: 'Orchestrator',
    'food-logger': 'Food Logger',
    'meal-planner': 'Meal Planner',
    'health-advisor': 'Health Advisor',
    'db-writer': 'DB Writer',
};

function renderTraceList(data) {
    const el = document.getElementById('trace-list');
    const traces = data?.traces || [];
    if (!traces.length) {
        el.innerHTML = '<p class="empty-state">No traces yet. Send a message in Chat first.</p>';
        return;
    }
    el.innerHTML = traces.map(t => {
        const time = t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : '?';
        const errClass = t.has_error ? ' has-error' : '';
        return `<div class="trace-list-item${errClass}" data-trace="${t.trace_id}">
            <span class="trace-id">${t.trace_id}</span>
            <span class="trace-time">${time}</span>
            <span class="trace-action">${t.first_action || '?'}</span>
            <span class="trace-steps-count">${t.actions_count} steps</span>
        </div>`;
    }).join('');

    // Click handler for each trace
    el.querySelectorAll('.trace-list-item').forEach(item => {
        item.addEventListener('click', () => {
            el.querySelectorAll('.trace-list-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            loadTraceFlow(item.dataset.trace);
        });
    });
}

async function loadTraceFlow(traceId) {
    const container = document.getElementById('trace-flow');
    container.style.display = 'block';
    container.innerHTML = '<p class="empty-state">Loading trace...</p>';

    try {
        const data = await API.getTrace(traceId);
        renderTraceFlow(data, container);
    } catch (e) {
        container.innerHTML = `<p class="empty-state">Error: ${e.message}</p>`;
    }
}

function renderTraceFlow(data, container) {
    const entries = data?.entries || [];
    if (!entries.length) {
        container.innerHTML = '<p class="empty-state">No entries for this trace</p>';
        return;
    }

    // Build flowchart nodes
    let nodesHtml = '';
    let totalDuration = 0;
    let totalTokens = 0;
    let llmCalls = 0;
    let agentsInvolved = new Set();

    entries.forEach((entry, idx) => {
        const agentName = AGENT_DISPLAY[entry.agent] || entry.agent;
        agentsInvolved.add(agentName);
        totalDuration += entry.duration_ms || 0;
        totalTokens += (entry.tokens_in || 0) + (entry.tokens_out || 0);
        if (entry.llm_call) llmCalls++;

        let nodeClass = '';
        if (entry.error) nodeClass = ' error';
        else if (entry.llm_call) nodeClass = ' llm';
        else if (entry.action && entry.action.includes('cache')) nodeClass = ' cache';

        // Badges
        let badges = '';
        if (entry.duration_ms) badges += `<span class="flow-badge time">${entry.duration_ms.toFixed(0)}ms</span>`;
        if (entry.protocol) badges += `<span class="flow-badge proto">${entry.protocol}</span>`;
        if (entry.llm_call) badges += `<span class="flow-badge llm-badge">LLM</span>`;
        if (entry.tokens_in || entry.tokens_out) badges += `<span class="flow-badge tokens">${(entry.tokens_in || 0) + (entry.tokens_out || 0)} tok</span>`;
        if (entry.error) badges += `<span class="flow-badge err">ERR</span>`;

        // Arrow before node (except first)
        if (idx > 0) {
            nodesHtml += '<div class="flow-arrow">&#x2192;</div>';
        }

        nodesHtml += `<div class="flow-node${nodeClass}">
            <div class="flow-node-agent">${agentName}</div>
            <div class="flow-node-action">${entry.action || '?'}</div>
            <div class="flow-node-meta">${badges}</div>
        </div>`;
    });

    // Summary
    const summaryHtml = `<div class="flow-summary">
        <span>Total: <span class="value">${totalDuration.toFixed(0)}ms</span></span>
        <span>Steps: <span class="value">${entries.length}</span></span>
        <span>LLM calls: <span class="value">${llmCalls}</span></span>
        <span>Tokens: <span class="value">${totalTokens}</span></span>
        <span>Agents: <span class="value">${agentsInvolved.size}</span></span>
    </div>`;

    container.innerHTML = `<div class="trace-flow-container">
        <div class="trace-flow-header">
            <span class="trace-flow-title">Trace: ${data.trace_id}</span>
            <button class="trace-flow-close" onclick="closeTraceFlow()">Close</button>
        </div>
        <div class="flow-chain">${nodesHtml}</div>
        ${summaryHtml}
    </div>`;
}

function closeTraceFlow() {
    document.getElementById('trace-flow').style.display = 'none';
    document.querySelectorAll('.trace-list-item').forEach(i => i.classList.remove('active'));
}

// ---- MEAL HISTORY ----
async function loadMeals() {
    const filter = document.getElementById('meal-type-filter').value;
    try {
        const data = await API.getMeals(APP.userId, 100, filter);
        APP.meals = data?.result || [];
        renderMealsTable();
    } catch (e) {
        document.getElementById('meals-tbody').innerHTML =
            `<tr><td colspan="7" class="empty-state">Failed to load meals: ${e.message}</td></tr>`;
    }
}

function renderMealsTable() {
    const tbody = document.getElementById('meals-tbody');
    if (!APP.meals.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No meals found</td></tr>';
        return;
    }
    tbody.innerHTML = APP.meals.map(m => {
        const safeName = (m.food_name || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
        return `<tr>
            <td>${m.meal_date || '-'}</td>
            <td>${m.meal_type || '-'}</td>
            <td>${m.food_name || '-'}</td>
            <td class="num">${fmt(m.calories)}</td>
            <td class="num">${fmt(m.protein_g)}</td>
            <td class="num">${fmt(m.carbs_g)}</td>
            <td class="num">${fmt(m.fat_g)}</td>
            <td class="meal-actions">
                <button class="btn-small btn-edit" onclick="editMeal(${m.id})">Edit</button>
                <button class="btn-small btn-delete" onclick="deleteMealItem(${m.id}, '${safeName}')">Delete</button>
            </td>
        </tr>`;
    }).join('');

    // Update sort indicator
    document.querySelectorAll('#meals-table th').forEach(th => {
        th.classList.toggle('sorted', th.dataset.sort === APP.sortField);
    });
}

function sortMeals(field) {
    if (APP.sortField === field) {
        APP.sortDir = APP.sortDir === 'asc' ? 'desc' : 'asc';
    } else {
        APP.sortField = field;
        APP.sortDir = 'desc';
    }
    APP.meals.sort((a, b) => {
        let va = a[field], vb = b[field];
        if (va == null) va = '';
        if (vb == null) vb = '';
        if (typeof va === 'number' || !isNaN(Number(va))) {
            return APP.sortDir === 'asc' ? Number(va) - Number(vb) : Number(vb) - Number(va);
        }
        return APP.sortDir === 'asc' ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
    });
    renderMealsTable();
}

// ---- PROFILE ----
async function loadProfile() {
    // Load current user profile for dashboard use
    try {
        const data = await API.getUser(APP.userId);
        APP.userProfile = data?.result || null;
    } catch (e) {
        APP.userProfile = null;
    }
    // Load all users list
    await loadAllUsers();
}

async function loadAllUsers() {
    const el = document.getElementById('users-list');
    try {
        const data = await API.getAllUsers();
        const users = data?.result || [];
        if (!users.length) {
            el.innerHTML = '<p class="empty-state">No users yet. Create one using the form.</p>';
            updateUserSelector([]);
            return;
        }
        updateUserSelector(users);
        el.innerHTML = users.map(u => `
            <div class="user-card ${u.id === APP.userId ? 'selected' : ''}">
                <div class="user-card-header">
                    <span class="user-card-name">${u.username}</span>
                    <span class="user-card-id">ID: ${u.id}</span>
                </div>
                <div class="user-card-details">
                    <span>${u.age || '-'} y/o</span>
                    <span>${u.weight_kg ? u.weight_kg + ' kg' : '-'}</span>
                    <span>${u.height_cm ? u.height_cm + ' cm' : '-'}</span>
                    <span>${u.gender || '-'}</span>
                    <span>${u.daily_cal_target || 2000} kcal</span>
                </div>
                <div class="user-card-actions">
                    <button class="btn-small btn-select" onclick="selectUser(${u.id})">Select</button>
                    <button class="btn-small btn-edit" onclick="editUser(${u.id})">Edit</button>
                    <button class="btn-small btn-delete" onclick="deleteUser(${u.id}, '${u.username}')">Delete</button>
                </div>
            </div>
        `).join('');
    } catch (e) {
        el.innerHTML = `<p class="empty-state">Error loading users: ${e.message}</p>`;
    }
}

function updateUserSelector(users) {
    const sel = document.getElementById('user-select');
    sel.innerHTML = '';
    users.forEach(u => {
        const opt = document.createElement('option');
        opt.value = u.id;
        opt.textContent = `${u.username} (ID: ${u.id})`;
        sel.appendChild(opt);
    });
    if (users.length) {
        // Keep current selection if valid
        const valid = users.find(u => u.id === APP.userId);
        if (valid) {
            sel.value = APP.userId;
        } else {
            sel.value = users[0].id;
            APP.userId = users[0].id;
        }
    }
}

function selectUser(id) {
    APP.userId = id;
    document.getElementById('user-select').value = id;
    loadAllUsers();
    showToast(`Switched to user ${id}`);
}

async function editUser(id) {
    // Fetch user data and populate form
    try {
        const data = await API.getUser(id);
        const user = data?.result;
        if (!user) { showToast('User not found', 'error'); return; }

        const form = document.getElementById('profile-form');
        form.edit_user_id.value = id;
        form.username.value = user.username || '';
        form.age.value = user.age || '';
        form.weight_kg.value = user.weight_kg || '';
        form.height_cm.value = user.height_cm || '';
        form.gender.value = user.gender || '';
        form.activity_level.value = user.activity_level || 'moderate';
        form.goal.value = user.goal || 'maintain';
        form.daily_cal_target.value = user.daily_cal_target || 2000;
        form.dietary_prefs.value = (user.dietary_prefs || []).join(', ');
        form.allergies.value = (user.allergies || []).join(', ');

        document.getElementById('profile-form-title').textContent = `Edit User: ${user.username}`;
        document.getElementById('profile-submit-btn').textContent = 'Save Changes';
        document.getElementById('profile-cancel-btn').style.display = 'inline-block';
        document.getElementById('profile-form-msg').textContent = '';
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

function cancelEdit() {
    const form = document.getElementById('profile-form');
    form.reset();
    form.edit_user_id.value = '';
    document.getElementById('profile-form-title').textContent = 'Create New User';
    document.getElementById('profile-submit-btn').textContent = 'Create User';
    document.getElementById('profile-cancel-btn').style.display = 'none';
    document.getElementById('profile-form-msg').textContent = '';
}

async function deleteUser(id, username) {
    if (!confirm(`Delete user "${username}" (ID: ${id})? This will also delete all their meals and data.`)) return;
    try {
        const resp = await API.deleteUser(id);
        if (resp.error) {
            showToast(resp.text || 'Failed to delete', 'error');
        } else {
            showToast(`User "${username}" deleted`);
            if (APP.userId === id) {
                APP.userId = 1;
            }
            cancelEdit();
            loadAllUsers();
        }
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function handleProfileSubmit(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const data = Object.fromEntries(fd.entries());
    const editId = data.edit_user_id;
    delete data.edit_user_id;

    data.age = data.age ? parseInt(data.age) : null;
    data.weight_kg = data.weight_kg ? parseFloat(data.weight_kg) : null;
    data.height_cm = data.height_cm ? parseFloat(data.height_cm) : null;
    data.daily_cal_target = parseInt(data.daily_cal_target) || 2000;
    data.dietary_prefs = data.dietary_prefs ? data.dietary_prefs.split(',').map(s => s.trim()).filter(Boolean) : [];
    data.allergies = data.allergies ? data.allergies.split(',').map(s => s.trim()).filter(Boolean) : [];

    const msgEl = document.getElementById('profile-form-msg');
    try {
        let resp;
        if (editId) {
            // Update existing user
            resp = await API.updateUser(editId, data);
        } else {
            // Create new user
            resp = await API.createUser(data);
        }

        if (resp.error) {
            msgEl.className = 'form-message error';
            msgEl.textContent = resp.text || 'Operation failed';
        } else {
            msgEl.className = 'form-message success';
            if (editId) {
                msgEl.textContent = `User "${data.username}" updated!`;
            } else {
                const newId = resp.result?.id;
                msgEl.textContent = `User "${data.username}" created! ID: ${newId}`;
                if (newId) {
                    APP.userId = newId;
                }
            }
            cancelEdit();
            loadAllUsers();
        }
    } catch (err) {
        msgEl.className = 'form-message error';
        msgEl.textContent = `Error: ${err.message}`;
    }
}

// ---- MEAL EDIT/DELETE ----
function editMeal(id) {
    const meal = APP.meals.find(m => m.id === id);
    if (!meal) return;
    const form = document.getElementById('meal-edit');
    form.meal_id.value = id;
    form.food_name.value = meal.food_name || '';
    form.meal_type.value = meal.meal_type || 'snack';
    form.calories.value = meal.calories || '';
    form.protein_g.value = meal.protein_g || '';
    form.carbs_g.value = meal.carbs_g || '';
    form.fat_g.value = meal.fat_g || '';
    document.getElementById('meal-edit-form').style.display = 'block';
    document.getElementById('meal-edit-title').textContent = `Edit: ${meal.food_name}`;
    document.getElementById('meal-edit-form').scrollIntoView({ behavior: 'smooth' });
}

function cancelMealEdit() {
    document.getElementById('meal-edit-form').style.display = 'none';
    document.getElementById('meal-edit').reset();
}

async function handleMealEdit(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const data = Object.fromEntries(fd.entries());
    const id = data.meal_id;
    delete data.meal_id;

    data.calories = data.calories ? parseFloat(data.calories) : null;
    data.protein_g = data.protein_g ? parseFloat(data.protein_g) : null;
    data.carbs_g = data.carbs_g ? parseFloat(data.carbs_g) : null;
    data.fat_g = data.fat_g ? parseFloat(data.fat_g) : null;

    try {
        const resp = await API.updateMeal(id, data);
        if (resp.error) {
            showToast(resp.text || 'Failed to update meal', 'error');
        } else {
            showToast('Meal updated');
            cancelMealEdit();
            loadMeals();
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

async function deleteMealItem(id, name) {
    if (!confirm(`Delete meal "${name}"?`)) return;
    try {
        const resp = await API.deleteMeal(id);
        if (resp.error) {
            showToast(resp.text || 'Failed to delete meal', 'error');
        } else {
            showToast(`Meal "${name}" deleted`);
            cancelMealEdit();
            loadMeals();
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

// ---- BENCHMARK ----
async function loadBenchmark() {
    const container = document.getElementById('benchmark-results');
    try {
        const data = await API.getBenchmark();
        if (data.error) {
            container.innerHTML = `<p class="empty-state">${data.error}</p>`;
            return;
        }
        const results = data.results || [];
        if (!results.length) {
            container.innerHTML = '<p class="empty-state">No benchmark results found</p>';
            return;
        }
        renderBenchmarkTable(results, container);
        renderLatencyChart(results, container);
        renderSizeChart(results, container);
    } catch (e) {
        container.innerHTML = `<p class="empty-state">Error: ${e.message}</p>`;
    }
}

function renderBenchmarkTable(results, container) {
    let html = `<div class="benchmark-table"><table><thead><tr>
        <th>Protocol</th><th>Avg (ms)</th><th>P50 (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th>
        <th>Msgs/s</th><th>Req (B)</th><th>Resp (B)</th><th>Ser (us)</th><th>Errors</th>
    </tr></thead><tbody>`;

    results.forEach(r => {
        html += `<tr>
            <td>${r.name}</td>
            <td class="num">${fmt(r.avg_latency_ms)}</td>
            <td class="num">${fmt(r.p50_latency_ms)}</td>
            <td class="num">${fmt(r.p95_latency_ms)}</td>
            <td class="num">${fmt(r.p99_latency_ms)}</td>
            <td class="num">${fmt(r.throughput_msgs_sec)}</td>
            <td class="num">${fmt(r.avg_request_bytes)}</td>
            <td class="num">${fmt(r.avg_response_bytes)}</td>
            <td class="num">${fmt(r.avg_serialization_us)}</td>
            <td class="num">${r.errors}</td>
        </tr>`;
    });
    html += '</tbody></table></div>';
    container.innerHTML = html;
}

function renderLatencyChart(results, container) {
    const maxLat = Math.max(...results.map(r => r.avg_latency_ms || 0), 1);
    let html = '<div class="chart-title">Average Latency (ms) - lower is better</div><div class="bar-chart">';
    results.forEach(r => {
        const pct = ((r.avg_latency_ms || 0) / maxLat) * 100;
        html += `<div class="bar-group">
            <div class="bar" style="height:${Math.max(pct, 2)}%">
                <span class="bar-value">${fmt(r.avg_latency_ms)}</span>
            </div>
            <div class="bar-label">${r.name}</div>
        </div>`;
    });
    html += '</div>';
    container.innerHTML += html;
}

function renderSizeChart(results, container) {
    const maxSize = Math.max(...results.map(r => r.avg_request_bytes || 0), 1);
    let html = '<div class="chart-title" style="margin-top:24px">Request Size (bytes) - smaller is better</div><div class="bar-chart">';
    results.forEach(r => {
        const pct = ((r.avg_request_bytes || 0) / maxSize) * 100;
        html += `<div class="bar-group">
            <div class="bar size-bar" style="height:${Math.max(pct, 2)}%">
                <span class="bar-value">${Math.round(r.avg_request_bytes || 0)}</span>
            </div>
            <div class="bar-label">${r.name}</div>
        </div>`;
    });
    html += '</div>';
    container.innerHTML += html;
}

// ---- INIT ----
function init() {
    // Navigation
    document.querySelectorAll('.nav-btn').forEach(btn =>
        btn.addEventListener('click', () => navigateTo(btn.dataset.page)));

    // Protocol toggle
    document.querySelectorAll('.toggle-btn').forEach(btn =>
        btn.addEventListener('click', () => setProtocol(btn.dataset.protocol)));

    // User selector
    document.getElementById('user-select').addEventListener('change', e => {
        APP.userId = parseInt(e.target.value);
        loadProfile();
    });

    // Chat
    document.getElementById('chat-send').addEventListener('click', handleChatSend);
    document.getElementById('chat-input').addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChatSend(); }
    });

    // Meal history
    document.querySelectorAll('#meals-table th[data-sort]').forEach(th =>
        th.addEventListener('click', () => sortMeals(th.dataset.sort)));
    document.getElementById('meal-type-filter').addEventListener('change', loadMeals);
    document.getElementById('meals-refresh').addEventListener('click', loadMeals);
    document.getElementById('meal-edit').addEventListener('submit', handleMealEdit);
    document.getElementById('meal-edit-cancel').addEventListener('click', cancelMealEdit);

    // Profile
    document.getElementById('profile-form').addEventListener('submit', handleProfileSubmit);
    document.getElementById('profile-cancel-btn').addEventListener('click', cancelEdit);

    // Traces
    document.getElementById('traces-refresh').addEventListener('click', async () => {
        try {
            const data = await API.getTraces(10);
            renderTraceList(data);
        } catch (e) { showToast(`Error: ${e.message}`, 'error'); }
    });

    // Benchmark
    document.getElementById('benchmark-load').addEventListener('click', loadBenchmark);

    // Welcome message
    addChatMessage('agent',
        '**Welcome to Smart Nutrition Tracker!**\n\n' +
        'I can help you with:\n' +
        '- Log food: "I had pasta and salad for lunch"\n' +
        '- Meal plans: "Plan my meals for tomorrow, high protein"\n' +
        '- Health advice: "How am I doing with my nutrition?"\n\n' +
        'Switch between A2A, PNP, and TOON protocols using the toggle above.');

    // Load user profile
    loadProfile();
}

document.addEventListener('DOMContentLoaded', init);
