/* ============================================================
   Smart Nutrition Tracker PWA — Full Client-Side App
   IndexedDB for storage, Gemini API for AI, no backend needed
   ============================================================ */

// ---- CONFIG ----
const GEMINI_MODEL = 'gemini-2.0-flash';
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;
const DB_NAME = 'NutritionTrackerDB';
const DB_VERSION = 1;

// ---- SYSTEM PROMPTS (same as backend agents) ----
const PROMPTS = {
    orchestrator: `You are a task router for a nutrition tracking app.
Analyze the user's request and decide the best workflow.

You have 3 agents:
- FOOD LOGGER: Logs food/meals, identifies what was eaten, estimates calories/macros.
  Use when: "I had pizza", "Log 2 eggs", "Just ate a banana", "I drank a smoothie"
- MEAL PLANNER: Creates personalized meal plans and recipe suggestions.
  Use when: "Plan my meals", "What should I eat?", "Give me a high-protein lunch idea"
- HEALTH ADVISOR: Health tips, nutrition analysis, progress review, nutrient alerts.
  Use when: "How am I doing?", "Am I eating enough protein?", "Give me health tips"

Respond ONLY with a valid JSON object (no markdown, no backticks):
{
    "plan": "log_food" | "meal_plan" | "health_advice" | "direct_answer",
    "query": "the refined query for the chosen agent (or empty string)",
    "direct_answer": "your answer if plan is direct_answer (or empty string)"
}`,

    food_logger: `You are a nutrition expert AI. Given a food description,
extract structured nutrition information.

Respond with a JSON object containing:
{
    "food_name": "name of the food item",
    "meal_type": "breakfast" | "lunch" | "dinner" | "snack",
    "calories": estimated total calories (number),
    "protein_g": grams of protein (number),
    "carbs_g": grams of carbohydrates (number),
    "fat_g": grams of fat (number),
    "confidence_note": "brief note about estimation confidence"
}

If the description mentions multiple items, return a JSON array of objects.
Be realistic with calorie and macro estimates. Use standard USDA-like values.
ALL numeric fields MUST be numbers (not strings).
If the user does not specify the meal type, infer it from context or default to "snack".

You MUST respond with valid JSON only. No markdown, no explanation, no code fences.`,

    meal_planner: `You are a professional nutritionist AI. Generate personalized meal plans.

Consider the user's:
- Daily calorie target
- Dietary preferences (vegetarian, vegan, keto, etc.)
- Allergies
- Health goals (lose weight, maintain, gain muscle)
- Activity level

Provide a structured meal plan with:
- Meals for the requested period (1 day or 1 week)
- Each meal: name, approximate calories, protein/carbs/fat
- Brief nutritional notes

Be practical and realistic. Use common, accessible foods.
Format your response in a readable way with clear sections.`,

    health_advisor: `You are a supportive health and nutrition advisor AI.

Based on the user's nutrition history and profile, provide:
- Analysis of their eating patterns
- Nutrient balance assessment (are they getting enough protein, fiber, etc.?)
- Specific, actionable health tips
- Encouragement and positive reinforcement
- Warnings if calorie intake is too low or too high
- Suggestions for missing nutrients

Be friendly, evidence-based, and non-judgmental.
Disclaimer: Always note you are an AI and not a medical professional.`,
};

// ============================================================
// INDEXEDDB LAYER
// ============================================================

let db = null;

function openDB() {
    return new Promise((resolve, reject) => {
        if (db) return resolve(db);
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = (e) => {
            const d = e.target.result;
            if (!d.objectStoreNames.contains('users')) {
                const us = d.createObjectStore('users', { keyPath: 'id', autoIncrement: true });
                us.createIndex('username', 'username', { unique: true });
            }
            if (!d.objectStoreNames.contains('meals')) {
                const ms = d.createObjectStore('meals', { keyPath: 'id', autoIncrement: true });
                ms.createIndex('user_id', 'user_id');
                ms.createIndex('meal_date', 'meal_date');
            }
        };
        req.onsuccess = (e) => { db = e.target.result; resolve(db); };
        req.onerror = (e) => reject(e.target.error);
    });
}

function dbTx(store, mode = 'readonly') {
    return db.transaction(store, mode).objectStore(store);
}

function dbReq(req) {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

const DB = {
    async createUser(data) {
        await openDB();
        return dbReq(dbTx('users', 'readwrite').add(data));
    },
    async getUser(id) {
        await openDB();
        return dbReq(dbTx('users').get(id));
    },
    async getAllUsers() {
        await openDB();
        return dbReq(dbTx('users').getAll());
    },
    async updateUser(id, data) {
        await openDB();
        const user = await dbReq(dbTx('users').get(id));
        if (!user) throw new Error('User not found');
        Object.assign(user, data);
        return dbReq(dbTx('users', 'readwrite').put(user));
    },
    async deleteUser(id) {
        await openDB();
        // Delete user's meals too
        const meals = await this.getMeals(id, 9999);
        const tx = db.transaction(['users', 'meals'], 'readwrite');
        tx.objectStore('users').delete(id);
        meals.forEach(m => tx.objectStore('meals').delete(m.id));
        return new Promise((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    },
    async addMeal(data) {
        await openDB();
        data.meal_date = data.meal_date || new Date().toISOString().split('T')[0];
        data.created_at = new Date().toISOString();
        return dbReq(dbTx('meals', 'readwrite').add(data));
    },
    async getMeals(userId, limit = 50, mealType = '') {
        await openDB();
        const all = await dbReq(dbTx('meals').index('user_id').getAll(userId));
        let filtered = mealType ? all.filter(m => m.meal_type === mealType) : all;
        filtered.sort((a, b) => (b.meal_date || '').localeCompare(a.meal_date || ''));
        return filtered.slice(0, limit);
    },
    async getMeal(id) {
        await openDB();
        return dbReq(dbTx('meals').get(id));
    },
    async updateMeal(id, data) {
        await openDB();
        const meal = await dbReq(dbTx('meals').get(id));
        if (!meal) throw new Error('Meal not found');
        Object.assign(meal, data);
        return dbReq(dbTx('meals', 'readwrite').put(meal));
    },
    async deleteMeal(id) {
        await openDB();
        return dbReq(dbTx('meals', 'readwrite').delete(id));
    },
    async getDailySummary(userId) {
        const today = new Date().toISOString().split('T')[0];
        const meals = await this.getMeals(userId, 9999);
        const todayMeals = meals.filter(m => m.meal_date === today);
        return {
            total_calories: todayMeals.reduce((s, m) => s + (m.calories || 0), 0),
            total_protein: todayMeals.reduce((s, m) => s + (m.protein_g || 0), 0),
            total_carbs: todayMeals.reduce((s, m) => s + (m.carbs_g || 0), 0),
            total_fat: todayMeals.reduce((s, m) => s + (m.fat_g || 0), 0),
            meal_count: todayMeals.length,
        };
    },
    async exportAll() {
        await openDB();
        const users = await dbReq(dbTx('users').getAll());
        const meals = await dbReq(dbTx('meals').getAll());
        return { users, meals, exported_at: new Date().toISOString() };
    },
    async importAll(data) {
        await openDB();
        const tx = db.transaction(['users', 'meals'], 'readwrite');
        // Clear existing
        tx.objectStore('users').clear();
        tx.objectStore('meals').clear();
        // Import
        (data.users || []).forEach(u => tx.objectStore('users').add(u));
        (data.meals || []).forEach(m => tx.objectStore('meals').add(m));
        return new Promise((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    },
    async clearAll() {
        await openDB();
        const tx = db.transaction(['users', 'meals'], 'readwrite');
        tx.objectStore('users').clear();
        tx.objectStore('meals').clear();
        return new Promise((resolve, reject) => {
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    },
};

// ============================================================
// GEMINI AI LAYER
// ============================================================

function getApiKey() {
    return localStorage.getItem('gemini_api_key') || '';
}

async function callGemini(prompt, systemPrompt, config = {}) {
    const apiKey = getApiKey();
    if (!apiKey) throw new Error('No API key set. Go to Settings to add your Gemini API key.');

    const body = {
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
            temperature: config.temperature || 0.7,
            maxOutputTokens: config.maxTokens || 1024,
        },
    };
    if (systemPrompt) {
        body.systemInstruction = { parts: [{ text: systemPrompt }] };
    }

    const resp = await fetch(`${GEMINI_URL}?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error?.message || `Gemini API error: ${resp.status}`);
    }

    const data = await resp.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

function parseJSON(text) {
    // Strip markdown code fences if present
    let clean = text.trim();
    if (clean.startsWith('```')) {
        clean = clean.replace(/^```(?:json)?\s*\n?/, '').replace(/\n?```\s*$/, '');
    }
    return JSON.parse(clean);
}

// ============================================================
// AI AGENT LOGIC (client-side orchestration)
// ============================================================

async function classifyIntent(text) {
    const raw = await callGemini(text, PROMPTS.orchestrator, { temperature: 0.1, maxTokens: 300 });
    return parseJSON(raw);
}

async function logFood(query, userId) {
    const raw = await callGemini(query, PROMPTS.food_logger, { temperature: 0.3, maxTokens: 1024 });
    let items = parseJSON(raw);
    if (!Array.isArray(items)) items = [items];

    const saved = [];
    for (const item of items) {
        const meal = {
            user_id: userId,
            food_name: item.food_name || 'Unknown',
            meal_type: item.meal_type || 'snack',
            calories: Number(item.calories) || 0,
            protein_g: Number(item.protein_g) || 0,
            carbs_g: Number(item.carbs_g) || 0,
            fat_g: Number(item.fat_g) || 0,
        };
        await DB.addMeal(meal);
        saved.push(meal);
    }

    const lines = saved.map(m =>
        `**${m.food_name}** (${m.meal_type}) - ${m.calories} kcal | P: ${m.protein_g}g | C: ${m.carbs_g}g | F: ${m.fat_g}g`
    );
    return `Logged ${saved.length} item${saved.length > 1 ? 's' : ''}:\n\n${lines.join('\n')}`;
}

async function planMeal(query, userProfile) {
    let context = query;
    if (userProfile) {
        context += `\n\nUser profile:
- Calorie target: ${userProfile.daily_cal_target || 2000} kcal/day
- Goal: ${userProfile.goal || 'maintain'}
- Activity: ${userProfile.activity_level || 'moderate'}
- Dietary preferences: ${(userProfile.dietary_prefs || []).join(', ') || 'none'}
- Allergies: ${(userProfile.allergies || []).join(', ') || 'none'}`;
    }
    return callGemini(context, PROMPTS.meal_planner, { temperature: 0.7, maxTokens: 2048 });
}

async function healthAdvice(query, userProfile, userId) {
    let context = query;
    if (userProfile) {
        context += `\n\nUser profile:
- Age: ${userProfile.age || 'unknown'}, Gender: ${userProfile.gender || 'unknown'}
- Weight: ${userProfile.weight_kg || 'unknown'} kg, Height: ${userProfile.height_cm || 'unknown'} cm
- Goal: ${userProfile.goal || 'maintain'}, Activity: ${userProfile.activity_level || 'moderate'}
- Calorie target: ${userProfile.daily_cal_target || 2000} kcal/day`;
    }

    // Add recent nutrition data
    try {
        const daily = await DB.getDailySummary(userId);
        const recent = await DB.getMeals(userId, 10);
        context += `\n\nToday's nutrition: ${daily.total_calories} kcal, P: ${daily.total_protein}g, C: ${daily.total_carbs}g, F: ${daily.total_fat}g (${daily.meal_count} meals)`;
        if (recent.length) {
            context += '\n\nRecent meals: ' + recent.map(m => `${m.food_name} (${m.calories} kcal)`).join(', ');
        }
    } catch (e) { /* ignore */ }

    return callGemini(context, PROMPTS.health_advisor, { temperature: 0.7, maxTokens: 1536 });
}

async function processChat(text, userId, userProfile) {
    const intent = await classifyIntent(text);

    if (intent.plan === 'direct_answer') {
        return { text: intent.direct_answer || "I'm not sure how to help with that. Try asking about food, meal plans, or health advice!", workflow: 'direct' };
    }

    let response;
    const query = intent.query || text;

    if (intent.plan === 'log_food') {
        response = await logFood(query, userId);
        return { text: response, workflow: 'food_logger' };
    }
    if (intent.plan === 'meal_plan') {
        response = await planMeal(query, userProfile);
        return { text: response, workflow: 'meal_planner' };
    }
    if (intent.plan === 'health_advice') {
        response = await healthAdvice(query, userProfile, userId);
        return { text: response, workflow: 'health_advisor' };
    }

    return { text: "I'm not sure how to help with that. Try logging food, asking for a meal plan, or getting health advice!", workflow: 'unknown' };
}

// ============================================================
// APP STATE
// ============================================================

const APP = {
    userId: null,
    userProfile: null,
    meals: [],
    currentPage: 'chat',
    loading: false,
    sortField: 'meal_date',
    sortDir: 'desc',
};

// ============================================================
// UTILITIES
// ============================================================

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

// ============================================================
// THEME
// ============================================================

function applyTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        document.getElementById('theme-toggle').textContent = '\u2600';
    } else {
        document.documentElement.removeAttribute('data-theme');
        document.getElementById('theme-toggle').textContent = '\u263D';
    }
}

// ============================================================
// NAVIGATION
// ============================================================

function navigateTo(page) {
    APP.currentPage = page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');
    document.querySelectorAll('.nav-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.page === page));

    if (page === 'dashboard') loadDashboard();
    if (page === 'meals') loadMeals();
    if (page === 'profile') loadProfile();
    if (page === 'settings') loadSettings();
}

// ============================================================
// CHAT
// ============================================================

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
        if (meta.time) parts.push(`${(meta.time / 1000).toFixed(1)}s`);
        footer.innerHTML = parts.join(' &middot; ');
        if (parts.length) bubble.appendChild(footer);
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

    if (!getApiKey()) {
        showToast('Set your Gemini API key in Settings first', 'error');
        navigateTo('settings');
        return;
    }

    addChatMessage('user', text);
    input.value = '';
    APP.loading = true;
    document.getElementById('chat-send').disabled = true;
    showTyping();

    try {
        const t0 = performance.now();
        const result = await processChat(text, APP.userId, APP.userProfile);
        const elapsed = performance.now() - t0;
        hideTyping();
        addChatMessage('agent', result.text, {
            workflow: result.workflow,
            time: elapsed,
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

// ============================================================
// DASHBOARD
// ============================================================

async function loadDashboard() {
    try {
        const [daily, meals] = await Promise.all([
            DB.getDailySummary(APP.userId),
            DB.getMeals(APP.userId, 10),
        ]);
        renderNutrition(daily);
        renderRecentMeals(meals);
    } catch (e) {
        console.error('Dashboard error:', e);
    }
}

function renderNutrition(data) {
    const el = document.getElementById('nutrition-bars');
    const target = APP.userProfile?.daily_cal_target || 2000;
    const currentCal = data.total_calories || 0;

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
        { label: 'Protein', cur: data.total_protein || 0, max: Math.round(target * 0.25 / 4), unit: 'g', color: 'var(--info)' },
        { label: 'Carbs', cur: data.total_carbs || 0, max: Math.round(target * 0.50 / 4), unit: 'g', color: 'var(--warning)' },
        { label: 'Fat', cur: data.total_fat || 0, max: Math.round(target * 0.25 / 9), unit: 'g', color: 'var(--danger)' },
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

function renderRecentMeals(meals) {
    const el = document.getElementById('recent-meals-list');
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

// ============================================================
// MEAL HISTORY
// ============================================================

async function loadMeals() {
    const filter = document.getElementById('meal-type-filter').value;
    try {
        APP.meals = await DB.getMeals(APP.userId, 100, filter);
        renderMealsTable();
    } catch (e) {
        document.getElementById('meals-tbody').innerHTML =
            `<tr><td colspan="8" class="empty-state">Failed to load meals: ${e.message}</td></tr>`;
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
                <button class="btn-small btn-delete" onclick="deleteMealItem(${m.id}, '${safeName}')">Del</button>
            </td>
        </tr>`;
    }).join('');
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
    const id = Number(data.meal_id);
    delete data.meal_id;

    data.calories = data.calories ? parseFloat(data.calories) : null;
    data.protein_g = data.protein_g ? parseFloat(data.protein_g) : null;
    data.carbs_g = data.carbs_g ? parseFloat(data.carbs_g) : null;
    data.fat_g = data.fat_g ? parseFloat(data.fat_g) : null;

    try {
        await DB.updateMeal(id, data);
        showToast('Meal updated');
        cancelMealEdit();
        loadMeals();
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

async function deleteMealItem(id, name) {
    if (!confirm(`Delete meal "${name}"?`)) return;
    try {
        await DB.deleteMeal(id);
        showToast(`Meal "${name}" deleted`);
        cancelMealEdit();
        loadMeals();
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

// ============================================================
// PROFILE
// ============================================================

async function loadProfile() {
    try {
        if (APP.userId) {
            APP.userProfile = await DB.getUser(APP.userId);
        }
    } catch (e) {
        APP.userProfile = null;
    }
    await loadAllUsers();
}

async function loadAllUsers() {
    const el = document.getElementById('users-list');
    try {
        const users = await DB.getAllUsers();
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
                    <button class="btn-small btn-delete" onclick="deleteUser(${u.id}, '${(u.username || '').replace(/'/g, "\\'")}')">Delete</button>
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
    if (!users.length) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No users';
        sel.appendChild(opt);
        return;
    }
    users.forEach(u => {
        const opt = document.createElement('option');
        opt.value = u.id;
        opt.textContent = `${u.username} (${u.id})`;
        sel.appendChild(opt);
    });
    const valid = users.find(u => u.id === APP.userId);
    if (valid) {
        sel.value = APP.userId;
    } else {
        sel.value = users[0].id;
        APP.userId = users[0].id;
    }
}

function selectUser(id) {
    APP.userId = id;
    document.getElementById('user-select').value = id;
    localStorage.setItem('selected_user_id', id);
    loadProfile();
    showToast(`Switched to user ${id}`);
}

async function editUser(id) {
    try {
        const user = await DB.getUser(id);
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
    if (!confirm(`Delete user "${username}" (ID: ${id})? This will also delete all their meals.`)) return;
    try {
        await DB.deleteUser(id);
        showToast(`User "${username}" deleted`);
        if (APP.userId === id) {
            const users = await DB.getAllUsers();
            APP.userId = users.length ? users[0].id : null;
            if (APP.userId) localStorage.setItem('selected_user_id', APP.userId);
        }
        cancelEdit();
        loadAllUsers();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function handleProfileSubmit(e) {
    e.preventDefault();
    const fd = new FormData(e.target);
    const data = Object.fromEntries(fd.entries());
    const editId = data.edit_user_id ? Number(data.edit_user_id) : null;
    delete data.edit_user_id;

    data.age = data.age ? parseInt(data.age) : null;
    data.weight_kg = data.weight_kg ? parseFloat(data.weight_kg) : null;
    data.height_cm = data.height_cm ? parseFloat(data.height_cm) : null;
    data.daily_cal_target = parseInt(data.daily_cal_target) || 2000;
    data.dietary_prefs = data.dietary_prefs ? data.dietary_prefs.split(',').map(s => s.trim()).filter(Boolean) : [];
    data.allergies = data.allergies ? data.allergies.split(',').map(s => s.trim()).filter(Boolean) : [];

    const msgEl = document.getElementById('profile-form-msg');
    try {
        if (editId) {
            await DB.updateUser(editId, data);
            msgEl.className = 'form-message success';
            msgEl.textContent = `User "${data.username}" updated!`;
        } else {
            const newId = await DB.createUser(data);
            msgEl.className = 'form-message success';
            msgEl.textContent = `User "${data.username}" created! ID: ${newId}`;
            APP.userId = newId;
            localStorage.setItem('selected_user_id', newId);
        }
        cancelEdit();
        loadAllUsers();
        loadProfile();
    } catch (err) {
        msgEl.className = 'form-message error';
        msgEl.textContent = `Error: ${err.message}`;
    }
}

// ============================================================
// SETTINGS
// ============================================================

function loadSettings() {
    const input = document.getElementById('api-key-input');
    const status = document.getElementById('api-key-status');
    const key = getApiKey();
    if (key) {
        input.value = key;
        status.className = 'form-message success';
        status.textContent = 'API key is set and ready.';
    } else {
        input.value = '';
        status.className = 'form-message error';
        status.textContent = 'No API key set. AI features will not work.';
    }
}

function saveApiKey() {
    const key = document.getElementById('api-key-input').value.trim();
    if (!key) {
        showToast('Enter an API key first', 'error');
        return;
    }
    localStorage.setItem('gemini_api_key', key);
    showToast('API key saved!');
    loadSettings();
}

function clearApiKey() {
    localStorage.removeItem('gemini_api_key');
    document.getElementById('api-key-input').value = '';
    showToast('API key cleared');
    loadSettings();
}

async function exportData() {
    try {
        const data = await DB.exportAll();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `nutrition-tracker-backup-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Data exported!');
    } catch (e) {
        showToast(`Export failed: ${e.message}`, 'error');
    }
}

function importData() {
    document.getElementById('import-file').click();
}

async function handleImportFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    try {
        const text = await file.text();
        const data = JSON.parse(text);
        if (!data.users || !data.meals) throw new Error('Invalid backup file');
        if (!confirm(`Import ${data.users.length} users and ${data.meals.length} meals? This will replace all current data.`)) return;
        await DB.importAll(data);
        showToast('Data imported!');
        loadProfile();
    } catch (err) {
        showToast(`Import failed: ${err.message}`, 'error');
    }
    e.target.value = '';
}

async function clearAllData() {
    if (!confirm('Delete ALL data? This cannot be undone.')) return;
    if (!confirm('Are you sure? All users, meals, and settings will be deleted.')) return;
    try {
        await DB.clearAll();
        localStorage.removeItem('selected_user_id');
        APP.userId = null;
        APP.userProfile = null;
        APP.meals = [];
        showToast('All data deleted');
        loadProfile();
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

// ============================================================
// INIT
// ============================================================

async function init() {
    // Open database
    await openDB();

    // Register service worker
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./sw.js').catch(() => {});
    }

    // Theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    applyTheme(savedTheme);
    document.getElementById('theme-toggle').addEventListener('click', () => {
        const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        applyTheme(next);
        localStorage.setItem('theme', next);
    });

    // Navigation
    document.querySelectorAll('.nav-btn').forEach(btn =>
        btn.addEventListener('click', () => navigateTo(btn.dataset.page)));

    // User selector
    document.getElementById('user-select').addEventListener('change', e => {
        APP.userId = Number(e.target.value) || null;
        if (APP.userId) localStorage.setItem('selected_user_id', APP.userId);
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

    // Settings
    document.getElementById('api-key-save').addEventListener('click', saveApiKey);
    document.getElementById('api-key-clear').addEventListener('click', clearApiKey);
    document.getElementById('export-data').addEventListener('click', exportData);
    document.getElementById('import-data').addEventListener('click', importData);
    document.getElementById('import-file').addEventListener('change', handleImportFile);
    document.getElementById('clear-all-data').addEventListener('click', clearAllData);

    // Restore selected user
    const savedUserId = localStorage.getItem('selected_user_id');
    if (savedUserId) APP.userId = Number(savedUserId);

    // Load users
    const users = await DB.getAllUsers();
    if (users.length) {
        if (!APP.userId || !users.find(u => u.id === APP.userId)) {
            APP.userId = users[0].id;
        }
        updateUserSelector(users);
        APP.userProfile = await DB.getUser(APP.userId);
    }

    // Welcome message
    const hasKey = !!getApiKey();
    addChatMessage('agent',
        '**Welcome to Smart Nutrition Tracker!**\n\n' +
        'I can help you with:\n' +
        '- Log food: "I had pasta and salad for lunch"\n' +
        '- Meal plans: "Plan my meals for tomorrow, high protein"\n' +
        '- Health advice: "How am I doing with my nutrition?"\n\n' +
        (hasKey ? 'Your API key is set. Start chatting!' :
            'Go to **Settings** to add your free Gemini API key first.'));

    // If no users exist, prompt to create one
    if (!users.length) {
        addChatMessage('agent', 'No user profile found. Go to **Profile** to create one for personalized meal plans and health advice.');
    }
}

document.addEventListener('DOMContentLoaded', init);
