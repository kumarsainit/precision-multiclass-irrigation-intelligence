/**
 * Precision Multi-Class Irrigation Intelligence — Main Application Controller (2026)
 * Pure Vanilla JavaScript (ES6+) for Flask + Jinja2 UI
 * Zero Emojis | Verified XGBoost Model Alignment | WCAG Compliant
 */

// --- Global App State ---
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
  latestPrediction: null
};

// ==========================================================================
// 1. Theme Management (Light / Dark Mode with Persistence & No Flash)
// ==========================================================================

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

  // Listen for system changes if user hasn't explicitly set localStorage
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

// ==========================================================================
// 2. Navigation & Mobile Drawer
// ==========================================================================

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

// ==========================================================================
// 3. Toast Notification System
// ==========================================================================

function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  if (!toast) return;

  // Clear existing timer
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

// ==========================================================================
// 4. Utility Functions (Formatting & Helpers)
// ==========================================================================

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

function formatPercent(val, decimals = 1) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  return `${(Number(val) * 100).toFixed(decimals)}%`;
}

// ==========================================================================
// 5. Guided Stepper Form Controller (Home Page)
// ==========================================================================

function initStepper() {
  const stepPanels = document.querySelectorAll('.step-panel');
  if (stepPanels.length === 0) return;

  goToStep(1);

  // Setup range sliders live output
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

  // Setup change listeners for summary
  const formInputs = document.querySelectorAll('#prediction-form input, #prediction-form select');
  formInputs.forEach((input) => {
    input.addEventListener('change', updateReviewSummary);
    input.addEventListener('input', updateReviewSummary);
  });
}

function goToStep(stepNumber) {
  if (stepNumber < 1 || stepNumber > AppState.totalSteps) return;

  // If moving forward, validate previous step
  if (stepNumber > AppState.currentStep) {
    if (!validateStep(AppState.currentStep)) return;
  }

  AppState.currentStep = stepNumber;

  // Update panels
  document.querySelectorAll('.step-panel').forEach((panel) => {
    panel.classList.remove('active');
  });
  const currentPanel = document.getElementById(`step-panel-${stepNumber}`);
  if (currentPanel) currentPanel.classList.add('active');

  // Update stepper header buttons & progress bar
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
  if (step === 1) {
    const cropType = document.getElementById('crop_type');
    const cropStage = document.getElementById('crop_growth_stage');
    const fieldArea = document.getElementById('field_area_hectare');

    if (!cropType || !cropType.value) {
      showToast('Please select a Crop Type.', 'error');
      cropType?.focus();
      return false;
    }
    if (!cropStage || !cropStage.value) {
      showToast('Please select a Crop Growth Stage.', 'error');
      cropStage?.focus();
      return false;
    }
    if (!fieldArea || fieldArea.value === '' || Number(fieldArea.value) <= 0) {
      showToast('Please enter a valid Field Area in hectares.', 'error');
      fieldArea?.focus();
      return false;
    }
  } else if (step === 2) {
    const temp = document.getElementById('temperature_c');
    const humidity = document.getElementById('humidity');
    const rainfall = document.getElementById('rainfall_mm');

    if (!temp || temp.value === '') {
      showToast('Please enter Temperature (°C).', 'error');
      temp?.focus();
      return false;
    }
    if (!humidity || humidity.value === '') {
      showToast('Please enter Humidity (%).', 'error');
      humidity?.focus();
      return false;
    }
    if (!rainfall || rainfall.value === '') {
      showToast('Please enter Rainfall (mm).', 'error');
      rainfall?.focus();
      return false;
    }
  } else if (step === 3) {
    const waterSource = document.getElementById('water_source');
    const soilPh = document.getElementById('soil_ph');

    if (!waterSource || !waterSource.value) {
      showToast('Please select a Water Source.', 'error');
      waterSource?.focus();
      return false;
    }
    if (!soilPh || soilPh.value === '') {
      showToast('Please enter Soil pH level.', 'error');
      soilPh?.focus();
      return false;
    }
  }
  return true;
}

function updateReviewSummary() {
  const getVal = (id, fallback = '—') => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    if (el.type === 'checkbox') return el.checked ? 'Yes (Active)' : 'No (None)';
    return el.value || fallback;
  };

  const setSummary = (key, text) => {
    const el = document.getElementById(`rev-${key}`);
    if (el) el.textContent = text;
  };

  setSummary('crop', `${getVal('crop_type')} (${getVal('crop_growth_stage')})`);
  setSummary('area', `${getVal('field_area_hectare')} ha`);
  setSummary('temp', `${getVal('temperature_c')} °C`);
  setSummary('humidity', `${getVal('humidity')} %`);
  setSummary('rain', `${getVal('rainfall_mm')} mm`);
  setSummary('wind', `${getVal('windspeed_kmph')} km/h`);
  setSummary('sun', `${getVal('sunlight_hours')} hrs`);
  setSummary('moisture', `${getVal('soil_moisture')} %`);
  setSummary('ph', `${getVal('soil_ph')}`);
  setSummary('carbon', `${getVal('organic_carbon')} %`);
  setSummary('source', `${getVal('water_source')}`);
  setSummary('mulching', getVal('mulching_used'));
}

// Quick Scenario Presets
const FieldScenarios = {
  arid: {
    crop_type: 'Tomato',
    crop_growth_stage: 'Flowering',
    field_area_hectare: 2.5,
    temperature_c: 37.5,
    humidity: 28,
    rainfall_mm: 0,
    windspeed_kmph: 24,
    sunlight_hours: 11,
    soil_moisture: 18,
    soil_ph: 7.4,
    organic_carbon: 0.9,
    water_source: 'Borewell',
    previous_irrigation_mm: 0,
    mulching_used: false
  },
  postRain: {
    crop_type: 'Rice',
    crop_growth_stage: 'Vegetative',
    field_area_hectare: 4.0,
    temperature_c: 27.0,
    humidity: 82,
    rainfall_mm: 35.0,
    windspeed_kmph: 12,
    sunlight_hours: 6.5,
    soil_moisture: 78,
    soil_ph: 6.5,
    organic_carbon: 2.0,
    water_source: 'Canal',
    previous_irrigation_mm: 20,
    mulching_used: true
  },
  moderate: {
    crop_type: 'Wheat',
    crop_growth_stage: 'Flowering',
    field_area_hectare: 1.5,
    temperature_c: 24.5,
    humidity: 52,
    rainfall_mm: 4.0,
    windspeed_kmph: 11,
    sunlight_hours: 8.5,
    soil_moisture: 42,
    soil_ph: 6.8,
    organic_carbon: 1.4,
    water_source: 'Drip System',
    previous_irrigation_mm: 12,
    mulching_used: true
  }
};

function loadScenario(scenarioKey) {
  const scenario = FieldScenarios[scenarioKey];
  if (!scenario) return;

  Object.entries(scenario).forEach(([key, val]) => {
    const el = document.getElementById(key);
    if (!el) return;
    if (el.type === 'checkbox') {
      el.checked = Boolean(val);
    } else {
      el.value = val;
    }
    const valBadge = document.getElementById(`${key}_val`);
    if (valBadge) valBadge.textContent = val;
  });

  updateReviewSummary();
  showToast(`Loaded ${scenarioKey.toUpperCase()} field parameters`, 'info');
  goToStep(4);
}

// ==========================================================================
// 6. Prediction API Execution & Result Dashboard
// ==========================================================================

function submitForm() {
  // Validate all steps
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
    btn.innerHTML = `<div class="spinner"></div><span>Running XGBoost Inference...</span>`;
  }

  const payload = {
    model: 'xgboost',
    crop_type: document.getElementById('crop_type')?.value || '',
    crop_growth_stage: document.getElementById('crop_growth_stage')?.value || '',
    field_area_hectare: parseFloat(document.getElementById('field_area_hectare')?.value || 1),
    temperature_c: parseFloat(document.getElementById('temperature_c')?.value || 25),
    humidity: parseFloat(document.getElementById('humidity')?.value || 60),
    rainfall_mm: parseFloat(document.getElementById('rainfall_mm')?.value || 0),
    windspeed_kmph: parseFloat(document.getElementById('windspeed_kmph')?.value || 10),
    sunlight_hours: parseFloat(document.getElementById('sunlight_hours')?.value || 8),
    soil_moisture: parseFloat(document.getElementById('soil_moisture')?.value || 45),
    soil_ph: parseFloat(document.getElementById('soil_ph')?.value || 6.5),
    organic_carbon: parseFloat(document.getElementById('organic_carbon')?.value || 1.5),
    water_source: document.getElementById('water_source')?.value || '',
    previous_irrigation_mm: parseFloat(document.getElementById('previous_irrigation_mm')?.value || 0),
    mulching_used: Boolean(document.getElementById('mulching_used')?.checked)
  };

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
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M2 12h20"></path></svg><span>Calculate Irrigation Requirement</span>`;
      }

      if (data.error) {
        showToast(data.error, 'error');
        if (emptyState) emptyState.style.display = 'flex';
        return;
      }

      AppState.latestPrediction = data;
      renderPredictionResult(data);
      showToast(`XGBoost Inference Complete: ${data.irrigation_required} Need`, 'success');
    })
    .catch((err) => {
      console.error('Prediction API Error:', err);
      if (loadingState) loadingState.classList.remove('visible');
      if (emptyState) emptyState.style.display = 'flex';
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M2 12h20"></path></svg><span>Calculate Irrigation Requirement</span>`;
      }
      showToast('Inference request failed. Please check server connection.', 'error');
    });
}

function renderPredictionResult(data) {
  const resultCard = document.getElementById('result-card');
  if (!resultCard) return;

  const level = (data.irrigation_required || 'None').trim();

  // 1. Badge Class & Label
  const badgeEl = document.getElementById('res-badge');
  if (badgeEl) {
    badgeEl.className = badgeClass(level);
    badgeEl.textContent = `${level} IRRIGATION NEED`;
  }

  // 2. Confidence Metric
  const confEl = document.getElementById('res-confidence');
  if (confEl) {
    confEl.textContent = data.confidence !== undefined && data.confidence !== null
      ? `${(data.confidence * 100).toFixed(1)}%`
      : '—';
  }

  // 3. Recommended Water Volume
  const waterEl = document.getElementById('res-water');
  if (waterEl) {
    if (data.water_recommendation_mm !== null && data.water_recommendation_mm !== undefined) {
      waterEl.textContent = `${data.water_recommendation_mm} mm`;
    } else {
      waterEl.textContent = 'Standard Dosage';
    }
  }

  // 4. Multi-class Probabilities Breakdown & Dynamic Chart
  renderClassProbabilities(data.class_probabilities || {});

  // 5. Reasoning list
  const reasonsList = document.getElementById('res-reasons');
  if (reasonsList) {
    reasonsList.innerHTML = '';
    const reasons = Array.isArray(data.reasoning) && data.reasoning.length > 0
      ? data.reasoning
      : ['Predicted with XGBoost multi-class decision boundaries on field parameters.'];

    reasons.forEach((reason) => {
      const li = document.createElement('li');
      li.className = 'reasoning-item';
      li.innerHTML = `<span class="reasoning-dot"></span><span>${reason}</span>`;
      reasonsList.appendChild(li);
    });
  }

  // 6. Feature Importance Drivers (from backend)
  renderFeatureImportances(data.feature_importances);

  // 7. Meta Time & Engine
  const timeEl = document.getElementById('res-time');
  if (timeEl) {
    timeEl.textContent = fmtDate(data.timestamp);
  }

  resultCard.classList.add('visible');
}

function renderClassProbabilities(probabilities) {
  // Probabilities keys: High, Low, Medium
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
        Primary decision weights derived from trained XGBoost model trees.
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

  // Reset range slider displays
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

// ==========================================================================
// 7. Prediction History & Audit Trail (/history)
// ==========================================================================

function initHistoryPage() {
  const historyContent = document.getElementById('history-content');
  if (!historyContent) return;

  loadHistory();

  // Search & Filter listeners
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

  // Setup auto refresh timer
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

  // Average confidence
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

  // Most frequent crop
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

  // Calculate slice for current page
  const total = AppState.historyFiltered.length;
  const startIdx = (AppState.historyPage - 1) * AppState.historyPageSize;
  const pageItems = AppState.historyFiltered.slice(startIdx, startIdx + AppState.historyPageSize);

  let rowsHtml = '';
  pageItems.forEach((p) => {
    const level = (p.irrigation_required || 'None').toLowerCase();
    const water = Number(p.water_recommendation_mm);
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
          <div style="font-weight: 700; color: ${Number.isFinite(water) && water > 0 ? 'var(--brand-orange)' : 'var(--text-muted)'};">
            ${Number.isFinite(water) && water > 0 ? `${water} mm` : 'Default'}
          </div>
        </td>
        <td>
          <div style="font-family: var(--font-mono); font-weight: 700; color: var(--text-primary);">
            ${Number.isFinite(conf) ? `${(conf * 100).toFixed(1)}%` : '—'}
          </div>
        </td>
        <td>
          <span class="model-pill-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
            XGBoost Final Deployment Model
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
            <th>Rec. Volume</th>
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

  // Previous Button
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

  // Next Button
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

// ==========================================================================
// 8. Model Specification Page Controller (/models)
// ==========================================================================

const VerifiedXGBoostMetrics = {
  accuracy: 0.9853,
  macroF1: 0.9704,
  macroRocAuc: 0.9976,
  microAvgPrecision: 0.9970,
  highClassF1: 0.9397,
  classes: {
    High: { precision: 0.9521, recall: 0.9276, f1: 0.9397 },
    Low: { precision: 0.9867, recall: 0.9951, f1: 0.9909 },
    Medium: { precision: 0.9860, recall: 0.9752, f1: 0.9806 }
  }
};

function initModelsPage() {
  const container = document.getElementById('models-metrics-container');
  if (!container) return;

  // Render Verified Metrics Progress Cards
  const metricItems = [
    { label: 'Test Accuracy', val: VerifiedXGBoostMetrics.accuracy },
    { label: 'Test Macro F1', val: VerifiedXGBoostMetrics.macroF1 },
    { label: 'Test Macro ROC-AUC', val: VerifiedXGBoostMetrics.macroRocAuc },
    { label: 'Test Micro Average Precision', val: VerifiedXGBoostMetrics.microAvgPrecision },
    { label: 'High-Class F1 Score', val: VerifiedXGBoostMetrics.highClassF1 }
  ];

  let metricsHtml = '';
  metricItems.forEach((m) => {
    const pct = (m.val * 100).toFixed(2);
    metricsHtml += `
      <div class="metric-progress-card">
        <div class="metric-progress-header">
          <span class="metric-progress-label">${m.label}</span>
          <span class="metric-progress-val">${pct}%</span>
        </div>
        <div class="metric-progress-track">
          <div class="metric-progress-fill" style="width: ${pct}%;"></div>
        </div>
      </div>
    `;
  });
  container.innerHTML = metricsHtml;

  // Render Class-wise report cards
  const classContainer = document.getElementById('class-breakdown-container');
  if (classContainer) {
    let classHtml = '';
    Object.entries(VerifiedXGBoostMetrics.classes).forEach(([className, scores]) => {
      classHtml += `
        <div class="class-metric-card">
          <div class="class-metric-title">
            <span class="${badgeClass(className)}">${className} Class</span>
            <span>Performance</span>
          </div>
          <div class="class-stat-row">
            <span class="class-stat-name">Precision</span>
            <span class="class-stat-val">${(scores.precision * 100).toFixed(2)}%</span>
          </div>
          <div class="class-stat-row">
            <span class="class-stat-name">Recall</span>
            <span class="class-stat-val">${(scores.recall * 100).toFixed(2)}%</span>
          </div>
          <div class="class-stat-row">
            <span class="class-stat-name">F1 Score</span>
            <span class="class-stat-val">${(scores.f1 * 100).toFixed(2)}%</span>
          </div>
        </div>
      `;
    });
    classContainer.innerHTML = classHtml;
  }
}

// ==========================================================================
// 9. Document Ready Initialization
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initStepper();
  initHistoryPage();
  initModelsPage();
});