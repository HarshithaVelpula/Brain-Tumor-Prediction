/**
 * Brain Tumor Detection - Frontend JavaScript
 * Handles file upload, API calls, and UI interactions
 */

// ==================== CONSTANTS ====================
const API_BASE = '/api';
const UPLOAD_AREA_ID = 'uploadArea';
const FILE_INPUT_ID = 'fileInput';
const PREDICT_BTN_ID = 'predictBtn';
const CLEAR_BTN_ID = 'clearBtn';
const FILE_INFO_ID = 'fileInfo';
const FILE_NAME_ID = 'fileName';

// UI State
let selectedFile = null;
let lastPrediction = null;

// ==================== DOM ELEMENTS ====================
const uploadArea = document.getElementById(UPLOAD_AREA_ID);
const fileInput = document.getElementById(FILE_INPUT_ID);
const predictBtn = document.getElementById(PREDICT_BTN_ID);
const clearBtn = document.getElementById(CLEAR_BTN_ID);
const fileInfo = document.getElementById(FILE_INFO_ID);
const fileNameSpan = document.getElementById(FILE_NAME_ID);

const resultsContainer = document.getElementById('resultsContainer');
const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const errorState = document.getElementById('errorState');

const toast = document.getElementById('toast');

// ==================== EVENT LISTENERS ====================

/**
 * Upload area drag and drop handlers
 */
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

/**
 * Predict button click handler
 */
predictBtn.addEventListener('click', () => {
    if (selectedFile) {
        predictTumor();
    }
});

/**
 * Clear button click handler
 */
clearBtn.addEventListener('click', () => {
    clearSelection();
});

/**
 * New analysis button
 */
document.getElementById('newAnalysisBtn')?.addEventListener('click', () => {
    clearSelection();
});

/**
 * Download report button
 */
document.getElementById('downloadResultBtn')?.addEventListener('click', () => {
    downloadReport();
});

/**
 * Dismiss error button
 */
document.getElementById('dismissErrorBtn')?.addEventListener('click', () => {
    hideError();
});

// ==================== FILE HANDLING ====================

/**
 * Handle file selection
 */
function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image.');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16 MB.');
        return;
    }

    selectedFile = file;
    
    // Update UI
    fileNameSpan.textContent = file.name;
    fileInfo.classList.remove('hidden');
    predictBtn.disabled = false;
    
    showToast(`✓ File selected: ${file.name}`, 'success');
}

/**
 * Clear file selection
 */
function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    fileNameSpan.textContent = '';
    predictBtn.disabled = true;
    
    // Hide results
    resultsContainer.classList.add('hidden');
    emptyState.classList.remove('hidden');
    errorState.classList.add('hidden');
    
    showToast('Selection cleared', 'success');
}

// ==================== API CALLS ====================

/**
 * Predict tumor class using uploaded image
 */
async function predictTumor() {
    if (!selectedFile) {
        showError('No file selected');
        return;
    }

    // Show loading state
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    errorState.classList.add('hidden');
    loadingState.classList.remove('hidden');
    predictBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
        lastPrediction = result;
        showToast('✓ Analysis complete', 'success');

    } catch (error) {
        console.error('Prediction error:', error);
        showError(`Analysis failed: ${error.message}`);
    } finally {
        loadingState.classList.add('hidden');
        predictBtn.disabled = false;
    }
}

// ==================== RESULT DISPLAY ====================

/**
 * Display prediction results
 */
function displayResults(result) {
    // Hide loading, show results
    loadingState.classList.add('hidden');
    emptyState.classList.add('hidden');
    errorState.classList.add('hidden');
    resultsContainer.classList.remove('hidden');

    // Update predicted class
    const classElement = document.getElementById('predictedClass');
    classElement.textContent = formatClassName(result.predicted_class);
    classElement.className = `class-badge ${result.predicted_class}`;

    // Update confidence score
    const confidenceScore = document.getElementById('confidenceScore');
    confidenceScore.textContent = `${result.confidence}%`;

    // Update progress bar with animation
    const progressFill = document.getElementById('progressFill');
    progressFill.style.width = '0%';
    setTimeout(() => {
        progressFill.style.width = `${result.confidence}%`;
    }, 100);

    // Update timestamp
    const resultTimestamp = document.getElementById('resultTimestamp');
    const date = new Date(result.timestamp);
    resultTimestamp.textContent = date.toLocaleString();

    // Update confidence scores for all classes
    displayConfidenceScores(result.all_predictions);

    // Display Grad-CAM image
    const gradcamImage = document.getElementById('gradcamImage');
    gradcamImage.src = result.gradcam;
    gradcamImage.alt = `Grad-CAM for ${result.predicted_class}`;

    // Fetch and display tumor information
    displayTumorInfo(result.predicted_class);
}

/**
 * Display confidence scores for all classes
 */
function displayConfidenceScores(scores) {
    const scoresContainer = document.getElementById('confidenceScores');
    scoresContainer.innerHTML = '';

    // Sort by confidence descending
    const sortedScores = Object.entries(scores)
        .sort(([, a], [, b]) => b - a);

    sortedScores.forEach(([className, score]) => {
        const scoreItem = document.createElement('div');
        scoreItem.className = 'score-item';
        
        const label = document.createElement('span');
        label.className = 'score-item-label';
        label.textContent = formatClassName(className);

        const value = document.createElement('span');
        value.className = 'score-item-value';
        value.textContent = `${score.toFixed(2)}%`;

        scoreItem.appendChild(label);
        scoreItem.appendChild(value);
        scoresContainer.appendChild(scoreItem);
    });
}

/**
 * Fetch and display comprehensive tumor information
 */
async function displayTumorInfo(tumorType) {
    try {
        const tumorInfoContainer = document.getElementById('tumorInfo');
        if (!tumorInfoContainer) return;

        // Show loading state
        tumorInfoContainer.innerHTML = '<div class="loading-spinner">Loading tumor information...</div>';

        const response = await fetch(`${API_BASE}/tumor-info/${tumorType}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch tumor information: ${response.statusText}`);
        }

        const data = await response.json();
        renderTumorInfo(tumorInfoContainer, data);
    } catch (error) {
        console.error('Error fetching tumor information:', error);
        const tumorInfoContainer = document.getElementById('tumorInfo');
        if (tumorInfoContainer) {
            tumorInfoContainer.innerHTML = '<div class="error-message">Unable to load tumor information.</div>';
        }
    }
}

/**
 * Render tumor information in the container
 */
function renderTumorInfo(container, tumorData) {
    container.innerHTML = '';

    // Severity badge
    const severityBadge = document.createElement('div');
    severityBadge.className = `severity-badge ${getSeverityClass(tumorData.severity)}`;
    severityBadge.textContent = `Severity: ${tumorData.severity}`;
    container.appendChild(severityBadge);

    // Tumor description
    const description = document.createElement('div');
    description.className = 'tumor-description';
    description.innerHTML = `
        <strong>${tumorData.name}</strong><br><br>
        ${tumorData.detailed_description}
    `;
    container.appendChild(description);

    // Medical information sections
    const infoSections = [
        { title: '⚕️ Symptoms', data: tumorData.symptoms, type: 'list' },
        { title: '🔍 Root Causes', data: tumorData.causes, type: 'list' },
        { title: '💔 Effects on Human Health', data: tumorData.effects_on_health, type: 'list' },
        { title: '💊 Treatment Options', data: tumorData.treatment_options, type: 'list' }
    ];

    infoSections.forEach(section => {
        if (section.data && section.data.length > 0) {
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'tumor-info-section';
            
            const title = document.createElement('h4');
            title.textContent = section.title;
            sectionDiv.appendChild(title);

            if (section.type === 'list') {
                const list = document.createElement('ul');
                section.data.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = item;
                    list.appendChild(li);
                });
                sectionDiv.appendChild(list);
            }

            container.appendChild(sectionDiv);
        }
    });

    // Prognosis section
    if (tumorData.prognosis) {
        const prognosisDiv = document.createElement('div');
        prognosisDiv.className = 'tumor-info-section';
        prognosisDiv.innerHTML = `
            <h4>📋 Prognosis</h4>
            <div class="prognosis-section">${tumorData.prognosis}</div>
        `;
        container.appendChild(prognosisDiv);
    }

    // Urgency indicator
    if (tumorData.urgency) {
        const urgencyDiv = document.createElement('div');
        urgencyDiv.className = `urgency-badge ${getUrgencyClass(tumorData.urgency)}`;
        urgencyDiv.innerHTML = `<strong>⚠️ URGENCY LEVEL: ${tumorData.urgency}</strong>`;
        container.appendChild(urgencyDiv);
    }
}

/**
 * Get severity badge class based on severity level
 */
function getSeverityClass(severity) {
    const severityMap = {
        'High': 'severity-high',
        'Medium-High': 'severity-high',
        'Medium': 'severity-medium',
        'Low': 'severity-low',
        'None': 'severity-none'
    };
    return severityMap[severity] || 'severity-low';
}

/**
 * Get urgency badge class based on urgency level
 */
function getUrgencyClass(urgency) {
    const urgencyMap = {
        'IMMEDIATE': 'urgency-immediate',
        'URGENT': 'urgency-urgent',
        'NONE': 'urgency-none'
    };
    return urgencyMap[urgency] || 'urgency-none';
}

// ==================== UTILITY FUNCTIONS ====================

/**
 * Format class name for display
 */
function formatClassName(className) {
    return className
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

/**
 * Show error message
 */
function showError(message) {
    const errorState = document.getElementById('errorState');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    loadingState.classList.add('hidden');
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    errorState.classList.remove('hidden');
    
    console.error('Error:', message);
}

/**
 * Hide error message
 */
function hideError() {
    const errorState = document.getElementById('errorState');
    errorState.classList.add('hidden');
    emptyState.classList.remove('hidden');
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

/**
 * Download analysis report as PDF
 */
function downloadReport() {
    if (!lastPrediction) {
        showError('No prediction data available');
        return;
    }

    // Show loading state
    const btn = document.getElementById('downloadResultBtn');
    const originalText = btn.textContent;
    btn.textContent = '⏳ Generating PDF...';
    btn.disabled = true;

    // Prepare data for PDF generation
    const reportData = {
        predicted_class: lastPrediction.predicted_class,
        confidence: lastPrediction.confidence,
        all_predictions: lastPrediction.all_predictions,
        gradcam: lastPrediction.gradcam
    };

    // Send to backend for PDF generation
    fetch(`${API_BASE}/generate-pdf-report`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(reportData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to generate PDF: ${response.statusText}`);
        }
        return response.blob();
    })
    .then(blob => {
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `brain_tumor_report_${Date.now()}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        showToast('✓ PDF Report downloaded successfully', 'success');
    })
    .catch(error => {
        console.error('Error downloading PDF:', error);
        showError(`Failed to generate PDF: ${error.message}`);
    })
    .finally(() => {
        // Restore button
        btn.textContent = originalText;
        btn.disabled = false;
    });
}

// ==================== INITIALIZATION ====================

/**
 * Initialize application
 */
async function initializeApp() {
    try {
        // Check API health
        const healthResponse = await fetch(`${API_BASE}/health`);
        if (healthResponse.ok) {
            const health = await healthResponse.json();
            console.log('✓ API is healthy', health);
            
            if (health.model_loaded) {
                showToast('✓ Model loaded successfully', 'success');
            } else {
                showError('Model is not loaded');
            }
        }
    } catch (error) {
        console.error('Failed to connect to API:', error);
        showError('Failed to connect to server. Please check if the backend is running.');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);

// ==================== KEYBOARD SHORTCUTS ====================

/**
 * Keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Press 'U' to focus upload area
    if (e.key === 'u' && e.ctrlKey) {
        e.preventDefault();
        fileInput.click();
    }

    // Press 'Enter' to predict (if file is selected)
    if (e.key === 'Enter' && selectedFile && !predictBtn.disabled) {
        e.preventDefault();
        predictTumor();
    }
});

// ==================== PROGRESSIVE ENHANCEMENT ====================

/**
 * Check if browser supports required features
 */
if (!window.FormData) {
    showError('Your browser does not support the required features. Please use a modern browser.');
}

console.log('Brain Tumor Detection System initialized');
