/**
 * Precision Multi-Class Irrigation Intelligence - front-end controller.
 * The feature schema and every metric are fetched from the Flask API so the
 * interface can never drift from the recorded experiment.
 */

const AppState = {
  theme: 'light',
  currentStep: 1,
  totalSteps: 4,
  autoRefreshInterval: null,
  autoRefreshEnabled: true,
  historyData: [],
  historyFiltered: [],
  historyPage: 1,
  historyPageSize: 8,
  latestPrediction: null,
  schema: null,
  modelName: 'the deployment model'
};

function fieldsInStep(step) {
  const panel = document.getElementById(`step-panel-${step}`);
  if (!panel) return [];
  return Array.from(panel.querySelectorAll('[name]'));
}

function readField(el) {
  if (el.type === 'checkbox') return el.checked ? 'Yes' : 'No';
  return el.value;
}

function initTheme() {
  const savedTheme = localStorage.getItem('precision_irrigation_theme');
  if (savedTheme === 'dark' || savedTheme === 'light') {
    AppState.theme = savedTheme;
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    AppState.theme = 'dark';
  } else {
    AppState.theme = 'light';
  }

  applyTheme(AppState.theme);

  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (!localStorage.getItem('precision_irrigation_theme')) {
        AppState.theme = e.matches ? 'dark' : 'light';
        applyTheme(AppState.theme);
      }
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const toggleBtn = document.getElementById('theme-toggle-btn');
  if (toggleBtn) {
    toggleBtn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  }
}

function toggleTheme() {
  AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('precision_irrigation_theme', AppState.theme);
  applyTheme(AppState.theme);
  showToast(`Switched to ${AppState.theme} mode`, 'info');
}

function toggleMobileNav() {
  const drawer = document.getElementById('mobile-drawer');
  const toggleBtn = document.querySelector('.mobile-nav-toggle');
  if (!drawer) return;

  const isOpen = drawer.classList.contains('open');
  if (isOpen) {
    drawer.classList.remove('open');
    if (toggleBtn) toggleBtn.setAttribute('aria-expanded', 'false');
  } else {
    drawer.classList.add('open');
    if (toggleBtn) toggleBtn.setAttribute('aria-expanded', 'true');
  }
}

function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  if (!toast) return;

  if (toast._timer) clearTimeout(toast._timer);

  let iconSvg = '';
  if (type === 'success') {
    iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
  } else if (type === 'error') {
    iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
  } else {
    iconSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
  }

  toast.innerHTML = `${iconSvg}<span>${message}</span>`;
  toast.className = `show ${type}`;

  toast._timer = setTimeout(() => {
    toast.className = '';
  }, 3500);
}

function badgeClass(level) {
  const val = (level || '').toLowerCase().trim();
  if (val.includes('high')) return 'badge badge-high';
  if (val.includes('med')) return 'badge badge-medium';
  if (val.includes('low')) return 'badge badge-low';
  return 'badge badge-none';
}

function fmtDate(ts) {
  if (!ts) return 'Just now';
  const date = new Date(ts);
  if (isNaN(date.getTime())) return 'Just now';
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
  });
}

function initStepper() {
  const stepPanels = document.querySelectorAll('.step-panel');
  if (stepPanels.length === 0) return;

  goToStep(1);

  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const valBadge = document.getElementById(`${slider.id}_val`);
    if (valBadge) {
      valBadge.textContent = slider.value;
      slider.addEventListener('input', () => {
        valBadge.textContent = slider.value;
        updateReviewSummary();
      });
    }
  });

  const formInputs = document.querySelectorAll('#prediction-form input, #prediction-form select');
  formInputs.forEach((input) => {
    input.addEventListener('change', updateReviewSummary);
    input.addEventListener('input', updateReviewSummary);
  });
}

function goToStep(stepNumber) {
  if (stepNumber < 1 || stepNumber > AppState.totalSteps) return;

  if (stepNumber > AppState.currentStep) {
    if (!validateStep(AppState.currentStep)) return;
  }

  AppState.currentStep = stepNumber;

  document.querySelectorAll('.step-panel').forEach((panel) => {
    panel.classList.remove('active');
  });
  const currentPanel = document.getElementById(`step-panel-${stepNumber}`);
  if (currentPanel) currentPanel.classList.add('active');

  document.querySelectorAll('.stepper-step').forEach((stepBtn) => {
    const stepIdx = parseInt(stepBtn.getAttribute('data-step'), 10);
    stepBtn.classList.remove('active', 'completed');

    if (stepIdx === stepNumber) {
      stepBtn.classList.add('active');
    } else if (stepIdx < stepNumber) {
      stepBtn.classList.add('completed');
    }
  });

  const progressFill = document.getElementById('stepper-progress-fill');
  if (progressFill) {
    const pct = ((stepNumber - 1) / (AppState.totalSteps - 1)) * 100;
    progressFill.style.width = `${pct}%`;
  }

  if (stepNumber === 4) {
    updateReviewSummary();
  }
}

function nextStep() {
  goToStep(AppState.currentStep + 1);
}

function prevStep() {
  goToStep(AppState.currentStep - 1);
}

function validateStep(step) {
  for (const el of fieldsInStep(step)) {
    if (el.type === 'checkbox') continue;

    if (el.required && el.value === '') {
      showToast(`Please provide a value for ${labelFor(el)}.`, 'error');
      el.focus();
      return false;
    }

    if (el.type === 'number' || el.type === 'range') {
      const value = Number(el.value);
      const min = Number(el.dataset.min);
      const max = Number(el.dataset.max);
      if (!Number.isFinite(value)) {
        showToast(`${labelFor(el)} must be a number.`, 'error');
        el.focus();
        return false;
      }
      if (Number.isFinite(min) && Number.isFinite(max) && (value < min || value > max)) {
        showToast(
          `${labelFor(el)} is outside the range the model was trained on (${min} to ${max}).`,
          'error'
        );
        el.focus();
        return false;
      }
    }
  }
  return true;
}

function labelFor(el) {
  const label = document.querySelector(`label[for="${el.id}"] span`);
  return label ? label.textContent.trim() : el.name.replace(/_/g, ' ');
}

function updateReviewSummary() {
  const grid = document.getElementById('review-grid');
  if (!grid) return;

  const items = [];
  for (let step = 1; step <= 3; step += 1) {
    fieldsInStep(step).forEach((el) => {
      items.push(`
        <div class="review-item">
          <span class="review-key">${labelFor(el)}</span>
          <span class="review-val">${readField(el) || '-'}</span>
        </div>
      `);
    });
  }
  grid.innerHTML = items.join('');
}

const SCENARIO_SHAPES = {
  arid: {
    numeric: {
      Soil_Moisture: 'p25', Rainfall_mm: 'p25', Humidity: 'p25', Organic_Carbon: 'median',
      Temperature_C: 'p75', Wind_Speed_kmh: 'p75', Sunlight_Hours: 'p75',
      Soil_pH: 'median', Electrical_Conductivity: 'median',
      Field_Area_hectare: 'median', Previous_Irrigation_mm: 'median'
    },
    categorical: { Crop_Growth_Stage: 'Flowering', Soil_Type: 'Sandy', Season: 'Zaid' },
    mulching: false
  },
  postRain: {
    numeric: {
      Soil_Moisture: 'p75', Rainfall_mm: 'p75', Humidity: 'p75', Organic_Carbon: 'median',
      Temperature_C: 'p25', Wind_Speed_kmh: 'p25', Sunlight_Hours: 'p25',
      Soil_pH: 'median', Electrical_Conductivity: 'median',
      Field_Area_hectare: 'median', Previous_Irrigation_mm: 'median'
    },
    categorical: { Crop_Growth_Stage: 'Harvest', Soil_Type: 'Clay', Season: 'Kharif' },
    mulching: true
  },
  moderate: { numeric: {}, categorical: {}, mulching: false }
};

function loadScenario(scenarioKey) {
  const shape = SCENARIO_SHAPES[scenarioKey];
  const schema = AppState.schema;
  if (!shape || !schema) return;

  Object.entries(schema.numeric_ranges).forEach(([column, stats]) => {
    const el = document.getElementById(column);
    if (!el) return;
    const statName = shape.numeric[column] || 'median';
    const value = stats[statName];
    el.value = Number(value).toFixed(2);
    const badge = document.getElementById(`${column}_val`);
    if (badge) badge.textContent = Number(value).toFixed(1);
  });

  Object.keys(schema.categorical_levels).forEach((column) => {
    const el = document.getElementById(column);
    if (!el || el.tagName !== 'SELECT') return;
    const wanted = shape.categorical[column];
    const levels = schema.categorical_levels[column];
    el.value = wanted && levels.includes(wanted) ? wanted : levels[0];
  });

  const mulching = document.getElementById('Mulching_Used');
  if (mulching) mulching.checked = Boolean(shape.mulching);

  updateReviewSummary();
  showToast('Loaded field parameters drawn from the training distribution', 'info');
  goToStep(4);
}

function submitForm() {
  for (let s = 1; s <= 3; s++) {
    if (!validateStep(s)) {
      goToStep(s);
      return;
    }
  }

  const btn = document.getElementById('submit-btn');
  const emptyState = document.getElementById('result-empty');
  const loadingState = document.getElementById('result-loading');
  const resultCard = document.getElementById('result-card');

  if (emptyState) emptyState.style.display = 'none';
  if (resultCard) resultCard.classList.remove('visible');
  if (loadingState) loadingState.classList.add('visible');

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<div class="spinner"></div><span>Running ${AppState.modelName} inference...</span>`;
  }

  const payload = {};
  for (let step = 1; step <= 3; step += 1) {
    fieldsInStep(step).forEach((el) => {
      payload[el.name] = el.type === 'number' || el.type === 'range'
        ? Number(el.value)
        : readField(el);
    });
  }

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
    .then((res) => res.json())
    .then((data) => {
      if (loadingState) loadingState.classList.remove('visible');
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M2 12h20"></path></svg><span>Run ${AppState.modelName} inference</span>`;
      }

      if (data.error) {
        const detail = data.invalid_fields
          ? `${data.error} ${Object.keys(data.invalid_fields).join(', ')}`
          : data.error;
        showToast(detail, 'error');
        if (emptyState) emptyState.style.display = 'flex';
        return;
      }

      AppState.latestPrediction = data;
      renderPredictionResult(data);
      showToast(`${data.model_used}: ${data.irrigation_required} irrigation need`, 'success');
    })
    .catch((err) => {
      console.error('Prediction API Error:', err);
      if (loadingState) loadingState.classList.remove('visible');
      if (emptyState) emptyState.style.display = 'flex';
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M2 12h20"></path></svg><span>Run ${AppState.modelName} inference</span>`;
      }
      showToast('Inference request failed. Please check server connection.', 'error');
    });
}

function renderPredictionResult(data) {
  const resultCard = document.getElementById('result-card');
  if (!resultCard) return;

  const level = (data.irrigation_required || 'None').trim();

  const badgeEl = document.getElementById('res-badge');
  if (badgeEl) {
    badgeEl.className = badgeClass(level);
    badgeEl.textContent = `${level} IRRIGATION NEED`;
  }

  const confEl = document.getElementById('res-confidence');
  if (confEl) {
    confEl.textContent = data.confidence !== undefined && data.confidence !== null
      ? `${(data.confidence * 100).toFixed(1)}%`
      : '—';
  }

  const highProbEl = document.getElementById('res-high-prob');
  if (highProbEl) {
    const highProb = data.class_probabilities ? data.class_probabilities.High : undefined;
    highProbEl.textContent = Number.isFinite(highProb) ? `${(highProb * 100).toFixed(1)}%` : '\u2014';
  }

  renderClassProbabilities(data.class_probabilities || {});

  const reasonsList = document.getElementById('res-reasons');
  if (reasonsList) {
    reasonsList.innerHTML = '';
    const reasons = Array.isArray(data.reasoning) && data.reasoning.length > 0
      ? data.reasoning
      : [`${data.model_used} assigned this field to the ${level} irrigation-need class.`];

    reasons.forEach((reason) => {
      const li = document.createElement('li');
      li.className = 'reasoning-item';
      li.innerHTML = `<span class="reasoning-dot"></span><span>${reason}</span>`;
      reasonsList.appendChild(li);
    });
  }

  renderFeatureImportances(data.feature_importances);

  const timeEl = document.getElementById('res-time');
  if (timeEl) {
    timeEl.textContent = fmtDate(data.timestamp);
  }

  resultCard.classList.add('visible');
}

function renderClassProbabilities(probabilities) {
  const lowPct = probabilities['Low'] !== undefined ? probabilities['Low'] * 100 : 0;
  const medPct = probabilities['Medium'] !== undefined ? probabilities['Medium'] * 100 : 0;
  const highPct = probabilities['High'] !== undefined ? probabilities['High'] * 100 : 0;

  const setBar = (type, pct) => {
    const textEl = document.getElementById(`prob-${type}-pct`);
    const fillEl = document.getElementById(`prob-${type}-fill`);
    if (textEl) textEl.textContent = `${pct.toFixed(1)}%`;
    if (fillEl) fillEl.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  };

  setBar('low', lowPct);
  setBar('med', medPct);
  setBar('high', highPct);
}

function renderFeatureImportances(importances) {
  const fiContainer = document.getElementById('res-fi');
  if (!fiContainer) return;

  fiContainer.innerHTML = '';

  if (!importances || Object.keys(importances).length === 0) {
    fiContainer.innerHTML = `
      <div style="font-size: 0.8rem; color: var(--text-muted); padding: 0.5rem 0;">
        Feature importances are unavailable for the loaded model.
      </div>
    `;
    return;
  }

  const items = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  const maxVal = Math.max(...items.map((it) => it[1]), 0.001);

  items.forEach(([key, val]) => {
    const pct = Math.round((val / maxVal) * 100);
    const label = key.replace(/_/g, ' ');

    const row = document.createElement('div');
    row.className = 'fi-row';
    row.innerHTML = `
      <span class="fi-label" title="${label}">${label}</span>
      <div class="fi-bar-bg">
        <div class="fi-bar-fill" style="width: ${pct}%"></div>
      </div>
      <span class="fi-pct">${pct}%</span>
    `;
    fiContainer.appendChild(row);
  });
}

function resetForm() {
  const form = document.getElementById('prediction-form');
  if (form) form.reset();

  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const valBadge = document.getElementById(`${slider.id}_val`);
    if (valBadge) valBadge.textContent = slider.value;
  });

  const emptyState = document.getElementById('result-empty');
  const resultCard = document.getElementById('result-card');
  if (resultCard) resultCard.classList.remove('visible');
  if (emptyState) emptyState.style.display = 'flex';

  goToStep(1);
  showToast('Prediction form reset', 'info');
}

function initHistoryPage() {
  const historyContent = document.getElementById('history-content');
  if (!historyContent) return;

  loadHistory();

  const searchInput = document.getElementById('history-search');
  const cropFilter = document.getElementById('history-crop-filter');
  const urgencyFilter = document.getElementById('history-urgency-filter');
  const pageSizeSelect = document.getElementById('history-pagesize');

  if (searchInput) searchInput.addEventListener('input', applyHistoryFilters);
  if (cropFilter) cropFilter.addEventListener('change', applyHistoryFilters);
  if (urgencyFilter) urgencyFilter.addEventListener('change', applyHistoryFilters);
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener('change', () => {
      AppState.historyPageSize = parseInt(pageSizeSelect.value, 10) || 8;
      AppState.historyPage = 1;
      renderHistoryTable();
    });
  }

  setupAutoRefresh();
}

function setupAutoRefresh() {
  if (AppState.autoRefreshInterval) clearInterval(AppState.autoRefreshInterval);
  if (AppState.autoRefreshEnabled) {
    AppState.autoRefreshInterval = setInterval(() => {
      loadHistory(false);
    }, 30000);
  }
}

function toggleAutoRefresh() {
  AppState.autoRefreshEnabled = !AppState.autoRefreshEnabled;
  const toggleBtn = document.getElementById('auto-refresh-toggle');
  if (toggleBtn) {
    toggleBtn.classList.toggle('active', AppState.autoRefreshEnabled);
    toggleBtn.setAttribute('aria-pressed', String(AppState.autoRefreshEnabled));
  }
  setupAutoRefresh();
  showToast(AppState.autoRefreshEnabled ? 'Auto-refresh enabled (30s)' : 'Auto-refresh paused', 'info');
}

function loadHistory(showNotification = true) {
  fetch('/api/history')
    .then((res) => res.json())
    .then((data) => {
      AppState.historyData = Array.isArray(data.predictions) ? data.predictions : [];
      updateHistoryStatistics(AppState.historyData);
      populateCropFilterOptions(AppState.historyData);
      applyHistoryFilters();
      if (showNotification && AppState.historyData.length > 0) {
        showToast('History logs synchronized', 'info');
      }
    })
    .catch((err) => {
      console.error('Failed to load history:', err);
      renderHistoryError();
    });
}

function updateHistoryStatistics(predictions) {
  const totalEl = document.getElementById('stat-total');
  const confEl = document.getElementById('stat-conf');
  const cropEl = document.getElementById('stat-crop');
  const needEl = document.getElementById('stat-need');

  if (totalEl) totalEl.textContent = predictions.length;

  if (predictions.length === 0) {
    if (confEl) confEl.textContent = '—';
    if (cropEl) cropEl.textContent = '—';
    if (needEl) needEl.textContent = '—';
    return;
  }

  const confValues = predictions
    .map((p) => Number(p.confidence))
    .filter((c) => Number.isFinite(c));

  if (confEl) {
    if (confValues.length > 0) {
      const avg = confValues.reduce((sum, v) => sum + v, 0) / confValues.length;
      confEl.textContent = `${(avg * 100).toFixed(1)}%`;
    } else {
      confEl.textContent = '—';
    }
  }

  const cropCounts = {};
  const needCounts = {};

  predictions.forEach((p) => {
    if (p.crop_type) cropCounts[p.crop_type] = (cropCounts[p.crop_type] || 0) + 1;
    if (p.irrigation_required) needCounts[p.irrigation_required] = (needCounts[p.irrigation_required] || 0) + 1;
  });

  if (cropEl) {
    const topCrop = Object.entries(cropCounts).sort((a, b) => b[1] - a[1])[0];
    cropEl.textContent = topCrop ? topCrop[0] : '—';
  }

  if (needEl) {
    const topNeed = Object.entries(needCounts).sort((a, b) => b[1] - a[1])[0];
    needEl.textContent = topNeed ? topNeed[0] : '—';
  }
}

function populateCropFilterOptions(predictions) {
  const cropFilter = document.getElementById('history-crop-filter');
  if (!cropFilter) return;

  const currentVal = cropFilter.value;
  const uniqueCrops = [...new Set(predictions.map((p) => p.crop_type).filter(Boolean))].sort();

  cropFilter.innerHTML = '<option value="">All Crop Types</option>';
  uniqueCrops.forEach((crop) => {
    const opt = document.createElement('option');
    opt.value = crop;
    opt.textContent = crop;
    if (crop === currentVal) opt.selected = true;
    cropFilter.appendChild(opt);
  });
}

function applyHistoryFilters() {
  const searchQuery = (document.getElementById('history-search')?.value || '').toLowerCase().trim();
  const selectedCrop = (document.getElementById('history-crop-filter')?.value || '').toLowerCase();
  const selectedUrgency = (document.getElementById('history-urgency-filter')?.value || '').toLowerCase();

  AppState.historyFiltered = AppState.historyData.filter((item) => {
    const crop = (item.crop_type || '').toLowerCase();
    const need = (item.irrigation_required || '').toLowerCase();
    const model = (item.model_used || '').toLowerCase();

    const matchesSearch = !searchQuery || crop.includes(searchQuery) || need.includes(searchQuery) || model.includes(searchQuery);
    const matchesCrop = !selectedCrop || crop === selectedCrop;
    const matchesUrgency = !selectedUrgency || need === selectedUrgency;

    return matchesSearch && matchesCrop && matchesUrgency;
  });

  AppState.historyPage = 1;
  renderHistoryTable();
}

function renderHistoryTable() {
  const container = document.getElementById('history-content');
  if (!container) return;

  if (AppState.historyFiltered.length === 0) {
    container.innerHTML = `
      <div style="padding: 4.5rem 1.5rem; text-align: center; color: var(--text-muted);">
        <div style="width: 56px; height: 56px; margin: 0 auto 1rem; border-radius: 50%; background: var(--bg-surface-subtle); display: flex; align-items: center; justify-content: center; color: var(--text-muted);">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
          </svg>
        </div>
        <h3 style="font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.35rem;">No History Records Found</h3>
        <p style="font-size: 0.875rem;">Run predictions from the dashboard or adjust active filters.</p>
      </div>
    `;
    renderPagination(0);
    return;
  }

  const total = AppState.historyFiltered.length;
  const startIdx = (AppState.historyPage - 1) * AppState.historyPageSize;
  const pageItems = AppState.historyFiltered.slice(startIdx, startIdx + AppState.historyPageSize);

  let rowsHtml = '';
  pageItems.forEach((p) => {
    const level = (p.irrigation_required || 'None').toLowerCase();
    const conf = Number(p.confidence);

    rowsHtml += `
      <tr>
        <td>
          <div style="font-weight: 700; color: var(--text-primary);">${fmtDate(p.timestamp)}</div>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 2px;">Log ID #${p.id || '—'}</div>
        </td>
        <td>
          <div style="display: flex; align-items: center; gap: 0.5rem; font-weight: 600;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--brand-emerald)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 20V10"></path>
              <path d="M12 14c-4-1-6-3.5-6-7 3.5 0 6 2 6 5"></path>
              <path d="M12 12c1-3 3-5 6-5 0 3.5-2 6-6 7"></path>
            </svg>
            <span>${p.crop_type || 'Unknown'}</span>
          </div>
        </td>
        <td>
          <span class="${badgeClass(level)}">
            ${p.irrigation_required || '—'}
          </span>
        </td>
        <td>
          <div style="font-family: var(--font-mono); font-weight: 700; color: var(--text-primary);">
            ${Number.isFinite(conf) ? `${(conf * 100).toFixed(1)}%` : '—'}
          </div>
        </td>
        <td>
          <span class="model-pill-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
            ${p.model_used || AppState.modelName}
          </span>
        </td>
      </tr>
    `;
  });

  container.innerHTML = `
    <div class="table-responsive">
      <table class="history-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Crop</th>
            <th>Irrigation Need</th>
            <th>Confidence</th>
            <th>Deployment Model</th>
          </tr>
        </thead>
        <tbody>
          ${rowsHtml}
        </tbody>
      </table>
    </div>
  `;

  renderPagination(total);
}

function renderPagination(totalCount) {
  const container = document.getElementById('history-pagination');
  if (!container) return;

  if (totalCount <= AppState.historyPageSize) {
    container.innerHTML = `
      <div class="pagination-info">Showing all ${totalCount} records</div>
    `;
    return;
  }

  const totalPages = Math.ceil(totalCount / AppState.historyPageSize);
  const cur = AppState.historyPage;
  const startItem = (cur - 1) * AppState.historyPageSize + 1;
  const endItem = Math.min(cur * AppState.historyPageSize, totalCount);

  let pagesHtml = '';

  pagesHtml += `
    <button class="page-btn" onclick="changeHistoryPage(${cur - 1})" ${cur === 1 ? 'disabled' : ''} aria-label="Previous page">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
    </button>
  `;

  for (let i = 1; i <= totalPages; i++) {
    if (i === 1 || i === totalPages || (i >= cur - 1 && i <= cur + 1)) {
      pagesHtml += `
        <button class="page-btn ${i === cur ? 'active' : ''}" onclick="changeHistoryPage(${i})">${i}</button>
      `;
    } else if (i === cur - 2 || i === cur + 2) {
      pagesHtml += `<span style="padding: 0 4px; color: var(--text-muted);">…</span>`;
    }
  }

  pagesHtml += `
    <button class="page-btn" onclick="changeHistoryPage(${cur + 1})" ${cur === totalPages ? 'disabled' : ''} aria-label="Next page">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
    </button>
  `;

  container.innerHTML = `
    <div class="pagination-info">Showing ${startItem}–${endItem} of ${totalCount} logs</div>
    <div class="pagination-controls">${pagesHtml}</div>
  `;
}

function changeHistoryPage(newPage) {
  const totalPages = Math.ceil(AppState.historyFiltered.length / AppState.historyPageSize);
  if (newPage < 1 || newPage > totalPages) return;
  AppState.historyPage = newPage;
  renderHistoryTable();
}

function renderHistoryError() {
  const container = document.getElementById('history-content');
  if (!container) return;
  container.innerHTML = `
    <div style="padding: 4rem 1.5rem; text-align: center; color: var(--status-high-fg);">
      <div style="width: 56px; height: 56px; margin: 0 auto 1rem; border-radius: 50%; background: var(--status-high-bg); display: flex; align-items: center; justify-content: center;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      </div>
      <h3 style="font-weight: 700; margin-bottom: 0.35rem;">History Synchronization Failed</h3>
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.25rem;">Could not connect to /api/history endpoint.</p>
      <button class="btn btn-secondary" onclick="loadHistory(true)">Retry Synchronization</button>
    </div>
  `;
}

function bootstrapSchema() {
  return fetch('/api/schema')
    .then((res) => res.json())
    .then((data) => {
      if (!data.error) {
        AppState.schema = data;
        AppState.modelName = data.final_model;
      }
    })
    .catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  bootstrapSchema();
  initStepper();
  initHistoryPage();
});