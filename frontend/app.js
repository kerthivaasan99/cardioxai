// API Configuration
const API_URL = 'http://localhost:5000';

// Global state
let currentData = null;
let originalProbability = 0;
let gaugeChart = null;
let currentPredictionData = null;
let currentInputData = null;
let healthScoreChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    setupEventListeners();
    checkAPIHealth();
    initializeCharts();
});

// Particle Background
function initParticles() {
    const container = document.getElementById('particles');
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDuration = (Math.random() * 20 + 10) + 's';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.opacity = Math.random() * 0.5 + 0.1;
        container.appendChild(particle);
    }
}

function initializeCharts() {
    // Pre-initialize
    console.log('Initializing futuristic dashboard...');
}

function setupEventListeners() {
    // Form submission
    document.getElementById('patientForm').addEventListener('submit', handleSubmit);
    
    // Real-time What-If sliders with debounce
    const debouncedUpdate = debounce(runRealTimeWhatIf, 400);
    
    ['whatifCholesterol', 'whatifSystolic', 'whatifDiastolic', 'whatifWeight'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', (e) => {
                updateSliderDisplay(id, e.target.value);
                debouncedUpdate();
            });
        }
    });
    
    ['whatifActive', 'whatifSmoke'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', debouncedUpdate);
        }
    });
    
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function updateSliderDisplay(id, value) {
    const displays = {
        'whatifCholesterol': {
            el: 'cholesterolValue',
            map: {1: 'Normal', 2: 'Above Normal', 3: 'High'}
        },
        'whatifSystolic': {
            el: 'systolicValue',
            suffix: ' mmHg'
        },
        'whatifDiastolic': {
            el: 'diastolicValue',
            suffix: ' mmHg'
        },
        'whatifWeight': {
            el: 'weightValue',
            suffix: ' kg'
        }
    };
    
    const config = displays[id];
    if (!config) return;
    
    const displayEl = document.getElementById(config.el);
    if (config.map) {
        displayEl.textContent = config.map[value];
    } else {
        displayEl.textContent = value + (config.suffix || '');
    }
}

async function checkAPIHealth() {
    const statusItems = {
        model: document.getElementById('modelStatus'),
        shap: document.getElementById('shapStatus'),
        lime: document.getElementById('limeStatus'),
        cf: document.getElementById('cfStatus')
    };
    
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        updateStatus(statusItems.model, data.model_loaded);
        updateStatus(statusItems.shap, data.model_loaded);
        updateStatus(statusItems.lime, data.model_loaded);
        updateStatus(statusItems.cf, data.model_loaded);
        
        if (data.model_loaded) {
            showNotification('AI systems operational', 'success');
        } else {
            showNotification('Model not loaded. Train model first.', 'error');
        }
    } catch (error) {
        Object.values(statusItems).forEach(item => {
            item.classList.remove('active');
            item.classList.add('error');
            item.querySelector('.status-value').textContent = 'OFFLINE';
        });
        showNotification('Backend connection failed', 'error');
    }
}

function updateStatus(element, isActive) {
    if (isActive) {
        element.classList.add('active');
        element.classList.remove('error');
        element.querySelector('.status-value').textContent = 'ACTIVE';
    } else {
        element.classList.remove('active');
        element.classList.add('error');
        element.querySelector('.status-value').textContent = 'ERROR';
    }
}

function showNotification(message, type = 'info') {
    const container = document.getElementById('notificationContainer');
    const notification = document.createElement('div');
    notification.className = `notification-modern ${type}`;
    
    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ'
    };
    
    notification.innerHTML = `
        <span style="font-size: 1.2rem;">${icons[type]}</span>
        <span>${message}</span>
    `;
    
    container.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'notificationSlide 0.4s ease reverse';
        setTimeout(() => notification.remove(), 400);
    }, 4000);
}

async function handleSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const ageYears = parseFloat(formData.get('age'));
    const ageDays = ageYears * 365.25;
    
    const data = {
        age: ageDays,
        gender: parseInt(formData.get('gender')),
        height: parseFloat(formData.get('height')),
        weight: parseFloat(formData.get('weight')),
        ap_hi: parseFloat(formData.get('ap_hi')),
        ap_lo: parseFloat(formData.get('ap_lo')),
        cholesterol: parseInt(formData.get('cholesterol')),
        gluc: parseInt(formData.get('gluc')),
        smoke: formData.get('smoke') ? 1 : 0,
        alco: formData.get('alco') ? 1 : 0,
        active: formData.get('active') ? 1 : 0
    };

    currentInputData = { ...data, age_years: ageYears };
    
    showAIOverlay(true);
    await runAIStages();
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        showAIOverlay(false);
        
        if (result.success) {
            currentPredictionData = result;
            originalProbability = result.probability.cvd;
            result.patient_metrics.age_years_display = ageYears;
            displayResults(result);
            showNotification('Analysis complete!', 'success');
        } else {
            showNotification(result.error, 'error');
        }
    } catch (error) {
        showAIOverlay(false);
        showNotification('Prediction failed: ' + error.message, 'error');
    }
}

async function runAIStages() {
    const stages = [
        { id: 'aiStage1', progress: 25, delay: 500 },
        { id: 'aiStage2', progress: 50, delay: 800 },
        { id: 'aiStage3', progress: 75, delay: 800 },
        { id: 'aiStage4', progress: 100, delay: 600 }
    ];
    
    const progressBar = document.getElementById('aiProgressBar');
    
    for (const stage of stages) {
        document.querySelectorAll('.stage-item').forEach(s => s.classList.remove('active'));
        document.getElementById(stage.id).classList.add('active');
        progressBar.style.width = stage.progress + '%';
        await sleep(stage.delay);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function showAIOverlay(show) {
    const overlay = document.getElementById('aiOverlay');
    if (show) {
        overlay.classList.add('active');
        document.getElementById('aiProgressBar').style.width = '0%';
    } else {
        overlay.classList.remove('active');
    }
}

function displayResults(data) {
    // Transition
    const inputSection = document.getElementById('inputSection');
    inputSection.style.opacity = '0';
    inputSection.style.transform = 'translateY(-20px)';
    
    setTimeout(() => {
        inputSection.style.display = 'none';
        
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'flex';
        resultsSection.style.flexDirection = 'column';
        
        setTimeout(() => {
            resultsSection.style.opacity = '1';
            resultsSection.style.transform = 'translateY(0)';
        }, 50);
    }, 300);
    
    // Update risk display
    const prob = data.probability.cvd;
    const percentage = Math.round(prob * 100);
    const healthScore = 100 - percentage;
    
    // Risk badge
    const badge = document.getElementById('riskBadge');
    badge.className = 'risk-badge ' + data.risk_level.level;
    badge.textContent = data.risk_level.level.toUpperCase() + ' RISK';
    
    // Animate percentage
    animateValue('riskPercentage', 0, percentage, 1500, '%');
    document.getElementById('riskLevelText').textContent = data.risk_level.level + ' Risk';
    
    // Render gauge
    renderGaugeChart(prob);
    
    // Health Score Ring
    updateHealthScoreRing(healthScore);
    document.getElementById('healthScoreValue').textContent = healthScore;
    
    // Confidence
    const confidence = Math.round((data.probability.cvd > 0.5 ? data.probability.cvd : 1 - data.probability.cvd) * 100);
    document.getElementById('confidenceBar').style.width = confidence + '%';
    document.getElementById('confidenceValue').textContent = confidence + '%';
    
    // Patient metrics
    renderPatientMetrics(data.patient_metrics);
    
    // Clinical insight
    generateClinicalInsight(data);
    
    // Initialize What-If
    initializeWhatIfSliders(data.input_data);
    
    // Render XAI
    renderSHAP(data.shap_explanation);
    renderLIME(data.lime_explanation);
    
    // Show recommendations card
    document.getElementById('recommendationsCard').style.display = 'block';
    
    // Initial counterfactual load
    setTimeout(() => showCounterfactuals(true), 500);
}

function animateValue(id, start, end, duration, suffix = '') {
    const obj = document.getElementById(id);
    const range = end - start;
    const minTimer = 50;
    let stepTime = Math.abs(Math.floor(duration / range));
    stepTime = Math.max(stepTime, minTimer);
    
    let startTime = new Date().getTime();
    let endTime = startTime + duration;
    let timer;
    
    function run() {
        let now = new Date().getTime();
        let remaining = Math.max((endTime - now) / duration, 0);
        let value = Math.round(end - (remaining * range));
        obj.innerHTML = value + suffix;
        if (value == end) {
            clearInterval(timer);
        }
    }
    
    timer = setInterval(run, stepTime);
    run();
}

function renderGaugeChart(probability) {
    const ctx = document.getElementById('gaugeChart').getContext('2d');
    
    if (gaugeChart) {
        gaugeChart.destroy();
    }
    
    const color = probability > 0.7 ? '#ef4444' : probability > 0.3 ? '#f59e0b' : '#10b981';
    
    gaugeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Risk', 'Safe'],
            datasets: [{
                data: [probability, 1 - probability],
                backgroundColor: [color, 'rgba(255,255,255,0.05)'],
                borderWidth: 0,
                cutout: '85%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            rotation: -90,
            circumference: 180,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            animation: {
                animateRotate: true,
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function updateHealthScoreRing(score) {
    const ring = document.getElementById('healthScoreRing');
    const circumference = 2 * Math.PI * 15.9155;
    const offset = circumference - (score / 100) * circumference;
    
    setTimeout(() => {
        ring.style.strokeDasharray = `${score}, 100`;
    }, 100);
}

function renderPatientMetrics(metrics) {
    const container = document.getElementById('patientMetrics');
    
    const items = [
        { icon: '🎂', label: 'Age', value: currentInputData.age_years + ' years' },
        { icon: '⚖️', label: 'BMI', value: metrics.bmi },
        { icon: '🩺', label: 'Blood Pressure', value: metrics.bp_category },
        { icon: '📏', label: 'Height', value: currentInputData.height + ' cm' },
        { icon: '⚖️', label: 'Weight', value: currentInputData.weight + ' kg' },
        { icon: '🩸', label: 'Cholesterol', value: getCholesterolLabel(currentInputData.cholesterol) }
    ];
    
    container.innerHTML = items.map((item, i) => `
        <div class="metric-item" style="animation: fadeInUp 0.4s ease ${i * 0.1}s both;">
            <div class="metric-item-label">
                <span class="metric-icon">${item.icon}</span>
                <span>${item.label}</span>
            </div>
            <span class="metric-item-value">${item.value}</span>
        </div>
    `).join('');
}

function getCholesterolLabel(level) {
    const labels = {1: 'Normal', 2: 'Above Normal', 3: 'High'};
    return labels[level] || 'Unknown';
}

function generateClinicalInsight(data) {
    const container = document.getElementById('clinicalInsight');
    const prob = Math.round(data.probability.cvd * 100);
    const level = data.risk_level.level;
    
    // Get top risk factors
    const riskFactors = data.shap_explanation.contributions
        .filter(c => c.contribution > 0)
        .slice(0, 3);
    
    const protectiveFactors = data.shap_explanation.contributions
        .filter(c => c.contribution < 0)
        .slice(0, 2);
    
    let html = `
        <p style="font-size: 1.1rem; line-height: 1.8; color: var(--text-secondary);">
            Based on your health profile, the AI model predicts a <strong style="color: var(--text-primary);">${level} cardiovascular risk</strong> of <strong style="color: var(--text-primary);">${prob}%</strong>.
        </p>
        <p style="margin-top: 16px; color: var(--text-secondary);">
    `;
    
    if (riskFactors.length > 0) {
        html += `The most influential factors <span style="color: var(--risk-high); font-weight: 600;">increasing</span> your risk include ${riskFactors.map(f => f.feature).join(', ')}. `;
    }
    
    if (protectiveFactors.length > 0) {
        html += `Positive indicators include ${protectiveFactors.map(f => f.feature).join(', ')} which help reduce risk.`;
    }
    
    html += '</p>';
    
    // Add recommendation preview
    html += `
        <div style="margin-top: 20px; padding: 16px; background: rgba(59, 130, 246, 0.1); border-radius: 12px; border-left: 4px solid var(--primary-electric);">
            <p style="color: var(--text-secondary); font-size: 0.95rem;">
                💡 <strong>AI Tip:</strong> Improving ${riskFactors[0]?.feature || 'key risk factors'} could significantly reduce your cardiovascular risk.
            </p>
        </div>
    `;
    
    container.innerHTML = html;
}

function initializeWhatIfSliders(inputData) {
    document.getElementById('whatifCholesterol').value = inputData.cholesterol;
    document.getElementById('whatifSystolic').value = inputData.ap_hi;
    document.getElementById('whatifDiastolic').value = inputData.ap_lo;
    document.getElementById('whatifWeight').value = inputData.weight;
    document.getElementById('whatifActive').checked = inputData.active === 1;
    document.getElementById('whatifSmoke').checked = inputData.smoke === 1;
    
    // Update displays
    updateSliderDisplay('whatifCholesterol', inputData.cholesterol);
    updateSliderDisplay('whatifSystolic', inputData.ap_hi);
    updateSliderDisplay('whatifDiastolic', inputData.ap_lo);
    updateSliderDisplay('whatifWeight', inputData.weight);
    
    // Initial calculation
    runRealTimeWhatIf();
}

function renderSHAP(shapData) {
    const features = shapData.contributions.slice(0, 8);
    
    const trace = {
        x: features.map(f => f.contribution),
        y: features.map(f => f.feature),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: features.map(f => f.contribution > 0 ? '#ef4444' : '#3b82f6'),
            line: {
                color: features.map(f => f.contribution > 0 ? '#dc2626' : '#2563eb'),
                width: 1
            }
        },
        text: features.map(f => {
            const val = f.contribution > 0 ? `+${f.contribution.toFixed(3)}` : f.contribution.toFixed(3);
            return val;
        }),
        textposition: 'outside',
        hovertemplate: '<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>'
    };
    
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', family: 'Inter, sans-serif' },
        xaxis: { 
            title: { text: 'SHAP Value (impact on risk)', font: { color: '#64748b' } },
            zeroline: true,
            zerolinecolor: '#475569',
            zerolinewidth: 2,
            gridcolor: 'rgba(255,255,255,0.05)'
        },
        yaxis: { 
            automargin: true,
            gridcolor: 'rgba(255,255,255,0.05)'
        },
        margin: { l: 120, r: 80, t: 30, b: 50 },
        showlegend: false
    };
    
    Plotly.newPlot('shapPlot', [trace], layout, {responsive: true, displayModeBar: false});
    
    // Ranking list
    const listContainer = document.getElementById('shapRankingList');
    listContainer.innerHTML = features.slice(0, 5).map((f, i) => `
        <div class="ranking-item" style="animation: fadeInUp 0.4s ease ${i * 0.1}s both;">
            <div class="ranking-number">${i + 1}</div>
            <div class="ranking-info">
                <div class="ranking-name">${f.feature}</div>
                <div class="ranking-desc">${f.description}: ${f.value}</div>
            </div>
            <div class="ranking-impact ${f.contribution > 0 ? 'positive' : 'negative'}">
                ${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(3)}
            </div>
        </div>
    `).join('');
}

function renderLIME(limeData) {
    const container = document.getElementById('limePlot');
    
    if (!limeData || limeData.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 60px 20px; color: var(--text-muted);">
                <div style="font-size: 3rem; margin-bottom: 16px;">🔍</div>
                <p>LIME explanation temporarily unavailable</p>
                <p style="font-size: 0.9rem; margin-top: 8px;">SHAP explanations provide comprehensive feature analysis</p>
            </div>
        `;
        document.getElementById('limeFeatures').innerHTML = '';
        return;
    }
    
    const trace = {
        x: limeData.map(l => Math.abs(l.weight)),
        y: limeData.map((l, i) => `Factor ${i + 1}`),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: limeData.map(l => l.weight > 0 ? '#ef4444' : '#10b981'),
            opacity: 0.9
        },
        text: limeData.map(l => l.direction),
        textposition: 'outside',
        hovertemplate: '<b>%{text}</b><br>Weight: %{x:.4f}<extra></extra>'
    };
    
    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', family: 'Inter, sans-serif' },
        xaxis: { 
            title: { text: 'Importance Weight', font: { color: '#64748b' } },
            gridcolor: 'rgba(255,255,255,0.05)'
        },
        yaxis: { 
            automargin: true,
            gridcolor: 'rgba(255,255,255,0.05)'
        },
        margin: { l: 100, r: 150, t: 30, b: 50 }
    };
    
    Plotly.newPlot('limePlot', [trace], layout, {responsive: true, displayModeBar: false});
}

async function runRealTimeWhatIf() {
    if (!currentInputData) return;
    
    const modifiedData = {
        ...currentInputData,
        cholesterol: parseInt(document.getElementById('whatifCholesterol').value),
        ap_hi: parseInt(document.getElementById('whatifSystolic').value),
        ap_lo: parseInt(document.getElementById('whatifDiastolic').value),
        weight: parseFloat(document.getElementById('whatifWeight').value),
        active: document.getElementById('whatifActive').checked ? 1 : 0,
        smoke: document.getElementById('whatifSmoke').checked ? 1 : 0
    };
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(modifiedData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayWhatIfResult(result.probability.cvd, modifiedData);
        }
    } catch (error) {
        console.error('What-if error:', error);
    }
}

function displayWhatIfResult(newProbability, modifiedData) {
    const origPercent = Math.round(originalProbability * 100);
    const newPercent = Math.round(newProbability * 100);
    const diff = newPercent - origPercent;
    
    // Update bars
    document.getElementById('originalBar').style.width = origPercent + '%';
    document.getElementById('modifiedBar').style.width = newPercent + '%';
    document.getElementById('originalRiskValue').textContent = origPercent + '%';
    document.getElementById('modifiedRiskValue').textContent = newPercent + '%';
    
    // Color coding
    const modBar = document.getElementById('modifiedBar');
    modBar.className = 'comp-bar-fill modified ' + (newPercent > 70 ? 'high' : newPercent > 30 ? 'moderate' : 'low');
    
    // Difference
    const diffEl = document.getElementById('diffValue');
    diffEl.textContent = (diff > 0 ? '+' : '') + diff + '%';
    diffEl.className = 'diff-value ' + (diff < 0 ? 'positive' : diff > 0 ? 'negative' : '');
    
    // Explanation
    const explanation = generateWhatIfExplanation(diff, modifiedData, currentInputData);
    document.getElementById('simulatorExplanation').innerHTML = `
        <p style="color: var(--text-secondary); line-height: 1.7;">${explanation}</p>
    `;
}

function generateWhatIfExplanation(change, modified, original) {
    const explanations = [];
    
    if (modified.cholesterol !== original.cholesterol) {
        const levels = {1: 'Normal', 2: 'Above Normal', 3: 'High'};
        const change = modified.cholesterol < original.cholesterol ? 'reducing' : 'increasing';
        explanations.push(`${change} cholesterol to ${levels[modified.cholesterol]}`);
    }
    
    if (modified.ap_hi !== original.ap_hi) {
        const diff = modified.ap_hi - original.ap_hi;
        const direction = diff < 0 ? 'lowering' : 'raising';
        explanations.push(`${direction} systolic BP by ${Math.abs(diff)} mmHg`);
    }
    
    if (modified.weight !== original.weight) {
        const diff = modified.weight - original.weight;
        const direction = diff < 0 ? 'losing' : 'gaining';
        explanations.push(`${direction} ${Math.abs(diff)} kg`);
    }
    
    if (modified.smoke !== original.smoke) {
        explanations.push(modified.smoke ? 'starting smoking' : 'quitting smoking');
    }
    
    if (modified.active !== original.active) {
        explanations.push(modified.active ? 'becoming physically active' : 'becoming sedentary');
    }
    
    if (explanations.length === 0) return 'Adjust sliders to see how lifestyle changes affect your risk.';
    
    const impact = change < 0 ? 'reduces' : change > 0 ? 'increases' : 'maintains';
    const impactText = change < 0 ? '✅ Risk reduction achieved!' : change > 0 ? '⚠️ Risk increases' : 'ℹ️ No significant change';
    
    return `${impactText} ${explanations.join(', ')} ${impact} your cardiovascular risk by ${Math.abs(change)}%.`;
}

async function showCounterfactuals(silent = false) {
    const modal = document.getElementById('cfModal');
    const loadingDiv = document.getElementById('cfLoading');
    const resultsDiv = document.getElementById('cfResults');
    
    if (!silent) {
        modal.classList.add('active');
    }
    
    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    
    // Animate stages
    const stages = ['stage1', 'stage2', 'stage3'];
    stages.forEach((id, i) => {
        setTimeout(() => {
            document.querySelectorAll('.stage').forEach(s => s.classList.remove('active'));
            document.getElementById(id).classList.add('active');
        }, i * 800);
    });
    
    try {
        const response = await fetch(`${API_URL}/counterfactual`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentInputData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            setTimeout(() => {
                displayCounterfactuals(result);
            }, 2500);
        } else {
            loadingDiv.innerHTML = `<p style="color: var(--risk-high);">${result.error}</p>`;
        }
    } catch (error) {
        loadingDiv.innerHTML = `<p style="color: var(--risk-high);">Failed to generate recommendations</p>`;
    }
}

function displayCounterfactuals(data) {
    document.getElementById('cfLoading').style.display = 'none';
    document.getElementById('cfResults').style.display = 'block';
    
    const bestCf = data.counterfactuals[0];
    const origRisk = Math.round(bestCf.original_risk * 100);
    const newRisk = Math.round(bestCf.new_risk * 100);
    const reduction = Math.round(bestCf.risk_reduction * 100);
    
    // Main recommendation
    const mainRecHtml = `
        <div class="cf-risk-compare">
            <div class="cf-risk-box">
                <span class="label">Current Risk</span>
                <span class="value">${origRisk}%</span>
            </div>
            <div class="cf-arrow">→</div>
            <div class="cf-risk-box">
                <span class="label">Target Risk</span>
                <span class="value improved">${newRisk}%</span>
            </div>
        </div>
        <div class="cf-impact">
            <span class="cf-impact-text">🎯 Risk Reduction: ${reduction} percentage points</span>
        </div>
        <div class="cf-actions-list">
            ${bestCf.changes.map(c => `
                <div class="cf-action">
                    <div class="cf-action-icon">✓</div>
                    <div class="cf-action-text">${c.description}</div>
                </div>
            `).join('')}
        </div>
    `;
    
    document.getElementById('cfMainRec').innerHTML = mainRecHtml;
    
    // Recommendations grid in main dashboard
    const recGrid = document.getElementById('recommendationsGrid');
    recGrid.innerHTML = bestCf.changes.map((c, i) => `
        <div class="recommendation-card" style="animation: fadeInUp 0.4s ease ${i * 0.1}s both;">
            <div class="rec-header">
                <span class="rec-type">Recommendation ${i + 1}</span>
                <span class="rec-impact">-${Math.round(bestCf.risk_reduction * 100 / bestCf.changes.length)}%</span>
            </div>
            <div class="rec-title-text">${c.description}</div>
            <div class="rec-description">This change significantly improves your cardiovascular health profile.</div>
        </div>
    `).join('');
    
    // Alternatives
    const altSection = document.getElementById('cfAlternativesSection');
    if (data.counterfactuals.length > 1) {
        altSection.innerHTML = `
            <h3 style="margin-bottom: 16px; color: var(--text-primary);">Alternative Approaches</h3>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                ${data.counterfactuals.slice(1).map((cf, idx) => `
                    <div class="cf-alternative-card" onclick="showAltDetails(${idx + 1})">
                        <div class="cf-alt-header">
                            <span class="cf-alt-title">Alternative Plan ${idx + 2}</span>
                            <span class="cf-alt-impact">↓ ${Math.round(cf.risk_reduction * 100)}%</span>
                        </div>
                        <div class="cf-alt-details">
                            <span>New risk: ${Math.round(cf.new_risk * 100)}%</span>
                            <span>•</span>
                            <span>${cf.changes.length} changes</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    } else {
        altSection.innerHTML = '';
    }
}

function closeCfModal() {
    const modal = document.getElementById('cfModal');
    modal.classList.remove('active');
}

async function generatePDF() {
    showNotification('Generating AI report...', 'info');
    
    try {
        const reportData = {
            patient_data: currentInputData,
            prediction: currentPredictionData,
            patient_metrics: currentPredictionData.patient_metrics,
            shap_explanation: currentPredictionData.shap_explanation,
            counterfactuals: { summary: generateClientCfSummary() }
        };
        
        const response = await fetch(`${API_URL}/generate-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reportData)
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `CardioXAI_Report_${new Date().toISOString().split('T')[0]}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            showNotification('Report downloaded successfully!', 'success');
        } else {
            throw new Error('Server PDF generation failed');
        }
    } catch (error) {
        console.error('PDF error:', error);
        await generateClientSidePDF();
    }
}

async function generateClientSidePDF() {
    const element = document.createElement('div');
    element.innerHTML = createPDFContent();
    element.style.cssText = 'position: absolute; left: -9999px;';
    document.body.appendChild(element);
    
    const opt = {
        margin: 10,
        filename: `CardioXAI_Report_${new Date().toISOString().split('T')[0]}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, backgroundColor: '#0f172a' },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };
    
    try {
        await html2pdf().set(opt).from(element).save();
        showNotification('Report downloaded!', 'success');
    } catch (error) {
        showNotification('PDF generation failed', 'error');
    } finally {
        document.body.removeChild(element);
    }
}

function createPDFContent() {
    const prob = currentPredictionData.probability.cvd;
    const riskLevel = currentPredictionData.risk_level;
    const metrics = currentPredictionData.patient_metrics;
    const healthScore = 100 - Math.round(prob * 100);
    
    return `
        <div style="font-family: Inter, Arial, sans-serif; padding: 40px; color: #f8fafc; background: #0f172a; max-width: 800px;">
            <div style="text-align: center; border-bottom: 3px solid #3b82f6; padding-bottom: 30px; margin-bottom: 40px;">
                <div style="font-size: 64px; margin-bottom: 16px;">🫀</div>
                <h1 style="color: #f8fafc; margin: 0; font-size: 32px; font-weight: 800;">CardioXAI</h1>
                <p style="color: #94a3b8; margin-top: 8px; font-size: 14px; letter-spacing: 0.1em;">AI-POWERED CARDIOVASCULAR RISK ASSESSMENT</p>
            </div>
            
            <div style="background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(6,182,212,0.2)); border-radius: 20px; padding: 40px; text-align: center; margin: 30px 0; border: 1px solid rgba(59,130,246,0.3);">
                <h3 style="margin-top: 0; color: #f8fafc; font-size: 18px; font-weight: 600;">Risk Assessment</h3>
                <div style="font-size: 72px; font-weight: 800; color: ${riskLevel.color === 'red' ? '#ef4444' : riskLevel.color === 'orange' ? '#f59e0b' : '#10b981'}; margin: 20px 0; text-shadow: 0 0 40px currentColor;">
                    ${Math.round(prob * 100)}%
                </div>
                <p style="font-size: 20px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #f8fafc;">${riskLevel.level} Risk</p>
                <p style="color: #94a3b8; max-width: 400px; margin: 16px auto 0;">${riskLevel.message}</p>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0;">
                <div style="background: rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 36px; font-weight: 800; color: #10b981;">${healthScore}</div>
                    <div style="color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 8px;">Health Score</div>
                </div>
                <div style="background: rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 36px; font-weight: 800; color: #3b82f6;">${metrics.bmi}</div>
                    <div style="color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 8px;">BMI</div>
                </div>
            </div>
            
            <h2 style="color: #3b82f6; font-size: 20px; margin: 40px 0 20px; font-weight: 700;">Key Risk Factors</h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="background: rgba(59,130,246,0.2);">
                        <th style="padding: 16px; text-align: left; color: #f8fafc; font-weight: 600;">Feature</th>
                        <th style="padding: 16px; text-align: left; color: #f8fafc; font-weight: 600;">Value</th>
                        <th style="padding: 16px; text-align: right; color: #f8fafc; font-weight: 600;">Impact</th>
                    </tr>
                </thead>
                <tbody>
                    ${currentPredictionData.shap_explanation.contributions.slice(0, 5).map(f => `
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <td style="padding: 16px; color: #f8fafc; font-weight: 500;">${f.feature}</td>
                            <td style="padding: 16px; color: #94a3b8;">${f.value}</td>
                            <td style="padding: 16px; text-align: right; color: ${f.contribution > 0 ? '#ef4444' : '#10b981'}; font-weight: 700;">
                                ${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(3)}
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
            
            <div style="background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(251,191,36,0.2)); border: 2px solid #f59e0b; border-radius: 16px; padding: 30px; margin-top: 40px;">
                <h3 style="color: #fbbf24; margin-top: 0; font-size: 18px; font-weight: 700;">💡 AI Recommendation</h3>
                <p style="color: #f8fafc; font-size: 15px; line-height: 1.7; margin: 0;">
                    ${generateClientCfSummary().outcome || 'Consult with a healthcare provider for personalized recommendations.'}
                </p>
            </div>
            
            <div style="margin-top: 50px; padding-top: 30px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 11px; color: #64748b; text-align: center; line-height: 1.8;">
                <p><strong>Medical Disclaimer:</strong> This report is generated by AI for educational purposes only and does not constitute medical advice.</p>
                <p>Always consult with a qualified healthcare provider before making health decisions.</p>
                <p style="margin-top: 20px; color: #475569;">© ${new Date().getFullYear()} CardioXAI. All rights reserved.</p>
            </div>
        </div>
    `;
}

function generateClientCfSummary() {
    const topFeatures = currentPredictionData.shap_explanation.contributions
        .filter(f => f.contribution > 0)
        .slice(0, 2);
    
    if (topFeatures.length === 0) return { outcome: '' };
    
    const actions = topFeatures.map(f => {
        const map = {
            'cholesterol': 'reduce cholesterol levels',
            'ap_hi': 'lower systolic blood pressure',
            'ap_lo': 'lower diastolic blood pressure',
            'weight': 'reduce weight',
            'smoke': 'quit smoking',
            'age': 'maintain regular checkups'
        };
        return map[f.feature] || `improve ${f.feature}`;
    });
    
    return {
        outcome: `Based on your AI analysis, we recommend ${actions.join(' and ')} to significantly reduce your cardiovascular risk.`
    };
}

function switchTab(tabName) {
    document.querySelectorAll('.xai-tab').forEach(btn => {
        btn.classList.remove('active');
        if(btn.getAttribute('data-tab') === tabName) {
            btn.classList.add('active');
        }
    });
    
    document.querySelectorAll('.xai-tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName + 'Tab').classList.add('active');
}

function showInput() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.opacity = '0';
    resultsSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultsSection.style.display = 'none';
        const inputSection = document.getElementById('inputSection');
        inputSection.style.display = 'block';
        setTimeout(() => {
            inputSection.style.opacity = '1';
            inputSection.style.transform = 'translateY(0)';
        }, 50);
    }, 300);
}

// Close modal on outside click
window.onclick = function(event) {
    const modal = document.getElementById('cfModal');
    if (event.target === modal) {
        closeCfModal();
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeCfModal();
    }
});