// Theme Management
const themeToggle = document.getElementById('themeToggle');
const html = document.documentElement;

// Load saved theme
const savedTheme = localStorage.getItem('theme') || 'light';
html.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
});

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('.icon');
    icon.textContent = theme === 'dark' ? '☀️' : '🌙';
}

// Mode Selection
const modeRadios = document.querySelectorAll('input[name="mode"]');
const modelSection = document.getElementById('modelSection');

modeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.value === 'offline') {
            modelSection.style.display = 'block';
            modelSection.style.animation = 'fadeIn 0.5s ease';
        } else {
            modelSection.style.display = 'none';
        }
    });
});

// Collapsible Sections
function toggleSection(sectionId) {
    const content = document.getElementById(sectionId);
    const header = content.previousElementSibling;
    const icon = header.querySelector('.toggle-icon');

    content.classList.toggle('collapsed');
    icon.classList.toggle('rotated');
}

// Form Reset
document.getElementById('resetBtn').addEventListener('click', () => {
    if (confirm('Voulez-vous vraiment réinitialiser tous les champs ?')) {
        // Reset form fields
        document.getElementById('file1').value = '';
        document.getElementById('file2').value = '';
        document.getElementById('sheet1').value = 'Feuil1';
        document.getElementById('sheet2').value = 'Feuil1';
        document.getElementById('col1').value = '1';
        document.getElementById('col2').value = '1';
        document.getElementById('threshold').value = '0.78';
        document.getElementById('batchSize').value = '16';
        document.getElementById('limit').value = '';

        // Reset checkboxes
        document.getElementById('llmEquivalent').checked = false;
        document.getElementById('forceCorr').checked = false;
        document.getElementById('assistCorr').checked = false;
        document.getElementById('assistPhrases').checked = false;

        // Reset mode
        document.querySelector('input[name="mode"][value="online"]').checked = true;
        modelSection.style.display = 'none';

        // Hide results
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
    }
});

// Run Comparison
document.getElementById('runBtn').addEventListener('click', async () => {
    const runBtn = document.getElementById('runBtn');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');

    // Validate inputs
    const file1 = document.getElementById('file1').value.trim();
    const file2 = document.getElementById('file2').value.trim();

    if (!file1 || !file2) {
        alert('Veuillez renseigner les deux fichiers à comparer.');
        return;
    }

    // Collect form data
    const formData = {
        mode: document.querySelector('input[name="mode"]:checked').value,
        file1: file1,
        sheet1: document.getElementById('sheet1').value,
        col1: parseInt(document.getElementById('col1').value),
        file2: file2,
        sheet2: document.getElementById('sheet2').value,
        col2: parseInt(document.getElementById('col2').value),
        threshold: parseFloat(document.getElementById('threshold').value),
        batch_size: parseInt(document.getElementById('batchSize').value),
        limit: document.getElementById('limit').value ? parseInt(document.getElementById('limit').value) : null,
        llm_equivalent: document.getElementById('llmEquivalent').checked,
        force_corr: document.getElementById('forceCorr').checked,
        assist_corr: document.getElementById('assistCorr').checked,
        assist_phrases: document.getElementById('assistPhrases').checked,
        match_mode: document.getElementById('matchMode').value,
        topk: parseInt(document.getElementById('topk').value),
    };

    // Add offline model config if needed
    if (formData.mode === 'offline') {
        formData.llm_model = document.getElementById('llmModel').value;
        formData.embedding_model = document.getElementById('embeddingModel').value;
    }

    // Disable button and show progress
    runBtn.disabled = true;
    runBtn.innerHTML = '<span class="btn-icon">⏳</span> Traitement en cours...';
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';

    updateProgress(10, 'Initialisation...');

    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Erreur lors de la comparaison');
        }

        updateProgress(90, 'Finalisation...');

        const result = await response.json();

        updateProgress(100, 'Terminé !');

        // Show results
        setTimeout(() => {
            displayResults(result);
            progressSection.style.display = 'none';
        }, 500);

    } catch (error) {
        console.error('Error:', error);
        alert(`Erreur: ${error.message}`);
        progressSection.style.display = 'none';
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = '<span class="btn-icon">▶️</span> Lancer la comparaison';
    }
});

function updateProgress(percent, text) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
}

function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    const matchesCount = document.getElementById('matchesCount');
    const underCount = document.getElementById('underCount');
    const matchRate = document.getElementById('matchRate');
    const downloadMatches = document.getElementById('downloadMatches');
    const downloadUnder = document.getElementById('downloadUnder');

    // Update counts
    matchesCount.textContent = result.matches_count;
    underCount.textContent = result.under_count;

    // Calculate match rate
    const total = result.matches_count + result.under_count;
    const rate = total > 0 ? ((result.matches_count / total) * 100).toFixed(1) : 0;
    matchRate.textContent = `${rate}%`;

    // Update download links
    downloadMatches.href = result.matches_file;
    downloadUnder.href = result.under_file;

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Auto-save form data to localStorage
const formFields = [
    'file1', 'sheet1', 'col1',
    'file2', 'sheet2', 'col2',
    'threshold', 'batchSize', 'limit'
];

formFields.forEach(fieldId => {
    const field = document.getElementById(fieldId);

    // Load saved value
    const savedValue = localStorage.getItem(`field_${fieldId}`);
    if (savedValue && fieldId !== 'file1' && fieldId !== 'file2') {
        field.value = savedValue;
    }

    // Save on change
    field.addEventListener('change', () => {
        localStorage.setItem(`field_${fieldId}`, field.value);
    });
});

// Enable/disable topk based on match mode
document.getElementById('matchMode').addEventListener('change', (e) => {
    const topkField = document.getElementById('topk');
    topkField.disabled = e.target.value === 'full';
    topkField.style.opacity = e.target.value === 'full' ? '0.5' : '1';
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('CompareDB Interface loaded');

    // Set initial topk state
    const matchMode = document.getElementById('matchMode').value;
    const topkField = document.getElementById('topk');
    topkField.disabled = matchMode === 'full';
    topkField.style.opacity = matchMode === 'full' ? '0.5' : '1';
});
