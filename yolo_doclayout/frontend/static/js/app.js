/**
 * Historical Manuscript Layout Analysis Engine - Frontend Controller
 * Handles async REST API polling, image rendering, interactive filtering, and telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const selectedBanner = document.getElementById('selected-file-banner');
    const selectedFilename = document.getElementById('selected-filename');
    const clearFileBtn = document.getElementById('clear-file-btn');
    const samplesContainer = document.getElementById('samples-container');
    const runBtn = document.getElementById('run-btn');
    
    // Progress Elements
    const progressBox = document.getElementById('progress-box');
    const progressBar = document.getElementById('progress-bar');
    const stageText = document.getElementById('stage-text');
    const stagePct = document.getElementById('stage-pct');
    const stageNodes = {
        loading: document.getElementById('node-loading'),
        preproc: document.getElementById('node-preproc'),
        detect: document.getElementById('node-detect'),
        classify: document.getElementById('node-classify'),
        complete: document.getElementById('node-complete')
    };

    // Results Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const emptyState = document.getElementById('empty-state');
    const imageViewWrapper = document.getElementById('image-view-wrapper');
    const displayAnnotated = document.getElementById('display-annotated');
    const displayOriginal = document.getElementById('display-original');
    const telemetryStrip = document.getElementById('telemetry-strip');
    const totalRegionsVal = document.getElementById('total-regions-val');
    const substratePill = document.getElementById('substrate-pill');
    const regionInspector = document.getElementById('region-inspector');
    const regionsTableBody = document.getElementById('regions-table-body');
    const classFilterChips = document.getElementById('class-filter-chips');
    const resultsFooter = document.getElementById('results-footer');
    const downloadOverlayBtn = document.getElementById('download-overlay-btn');

    // State Variables
    let selectedFile = null;
    let selectedSampleName = null;
    let currentPredictionData = null;
    let activeFilterClass = 'all';
    let pollingTimer = null;

    // 1. Initialize & Fetch Samples
    loadSampleManuscripts();

    // 2. File Input & Dropzone Handlers
    browseBtn.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('click', (e) => {
        if (e.target !== browseBtn && e.target !== clearFileBtn && !selectedFile) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleLocalFileSelection(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleLocalFileSelection(e.dataTransfer.files[0]);
        }
    });

    clearFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });

    function handleLocalFileSelection(file) {
        selectedFile = file;
        selectedSampleName = null;
        selectedFilename.textContent = file.name;
        selectedBanner.style.display = 'flex';
        document.querySelector('.dropzone-content').style.display = 'none';

        // Clear active class from sample cards
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
    }

    function clearSelectedFile() {
        selectedFile = null;
        fileInput.value = '';
        selectedBanner.style.display = 'none';
        document.querySelector('.dropzone-content').style.display = 'block';
    }

    // 3. Load Pre-loaded Sample Manuscripts from API
    async function loadSampleManuscripts() {
        try {
            const resp = await fetch('/api/samples');
            const data = await resp.json();
            samplesContainer.innerHTML = '';

            if (data.samples && data.samples.length > 0) {
                data.samples.forEach((sample, idx) => {
                    const card = document.createElement('div');
                    card.className = 'sample-card';
                    if (idx === 0) {
                        // Set first sample by default
                        card.classList.add('active');
                        selectedSampleName = sample.filename;
                    }

                    card.innerHTML = `
                        <img src="${sample.preview_url}" class="sample-thumbnail" alt="${sample.filename}">
                        <span class="sample-title" title="${sample.filename}">${sample.filename}</span>
                        <span class="sample-substrate-badge">${sample.substrate_hint.split(' ')[0]}</span>
                    `;

                    card.addEventListener('click', () => {
                        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
                        card.classList.add('active');
                        selectedSampleName = sample.filename;
                        clearSelectedFile();
                    });

                    samplesContainer.appendChild(card);
                });
            } else {
                samplesContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem;">No test images found.</div>';
            }
        } catch (err) {
            console.error('Failed to load sample manuscripts:', err);
            samplesContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem;">Samples offline.</div>';
        }
    }

    // 4. Tab Switching (Annotated Overlay vs. Original Scan)
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tabName = btn.dataset.tab;

            if (tabName === 'annotated') {
                imageViewWrapper.style.display = 'flex';
                displayAnnotated.classList.add('active');
                displayOriginal.classList.remove('active');
            } else if (tabName === 'original') {
                imageViewWrapper.style.display = 'flex';
                displayOriginal.classList.add('active');
                displayAnnotated.classList.remove('active');
            }
        });
    });

    // 5. Execute Layout Detection
    runBtn.addEventListener('click', async () => {
        if (!selectedFile && !selectedSampleName) {
            alert('Please select a sample manuscript or upload an image first.');
            return;
        }

        const formData = new FormData();
        formData.append('conf', 0.40);

        if (selectedFile) {
            formData.append('file', selectedFile);
        } else if (selectedSampleName) {
            formData.append('sample_name', selectedSampleName);
        }

        // UI Loading State
        runBtn.disabled = true;
        progressBox.style.display = 'flex';
        updateProgressUI(10, 'Submitting job to pipeline worker...', 'loading');

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();

            if (data.job_id) {
                pollJobStatus(data.job_id);
            } else {
                throw new Error(data.error || 'Failed to initialize background job.');
            }
        } catch (err) {
            alert('Error running pipeline: ' + err.message);
            resetProgressUI();
            runBtn.disabled = false;
        }
    });

    // 6. Polling Function for Background Worker Thread Execution
    function pollJobStatus(jobId) {
        if (pollingTimer) clearInterval(pollingTimer);

        pollingTimer = setInterval(async () => {
            try {
                const resp = await fetch(`/api/status/${jobId}`);
                const job = await resp.json();

                if (job.status === 'RUNNING' || job.status === 'PENDING') {
                    const stage = job.stage || 'PROCESSING';
                    const pct = job.progress_pct || 30;
                    let nodeKey = 'loading';
                    if (stage === 'PREPROCESSING') nodeKey = 'preproc';
                    else if (stage === 'DETECTION') nodeKey = 'detect';
                    else if (stage === 'CLASSIFICATION') nodeKey = 'classify';
                    else if (stage === 'SERIALIZATION') nodeKey = 'complete';

                    updateProgressUI(pct, `Executing: ${stage}`, nodeKey);
                } else if (job.status === 'COMPLETED') {
                    clearInterval(pollingTimer);
                    updateProgressUI(100, 'Layout Analysis Complete!', 'complete');
                    setTimeout(() => {
                        progressBox.style.display = 'none';
                        runBtn.disabled = false;
                        renderResults(job.result);
                    }, 400);
                } else if (job.status === 'FAILED') {
                    clearInterval(pollingTimer);
                    alert('Job failed: ' + (job.error || 'Unknown error'));
                    resetProgressUI();
                    runBtn.disabled = false;
                }
            } catch (err) {
                console.error('Polling error:', err);
            }
        }, 150);
    }

    function updateProgressUI(pct, text, activeNode) {
        progressBar.style.width = `${pct}%`;
        stagePct.textContent = `${pct}%`;
        stageText.textContent = text;

        Object.keys(stageNodes).forEach(k => {
            stageNodes[k].classList.remove('active', 'done');
        });

        if (activeNode && stageNodes[activeNode]) {
            stageNodes[activeNode].classList.add('active');
        }
    }

    function resetProgressUI() {
        progressBox.style.display = 'none';
        progressBar.style.width = '0%';
        stagePct.textContent = '0%';
    }

    // 7. Render Analysis Results
    function renderResults(result) {
        currentPredictionData = result;

        // Hide Empty State, Show Views
        emptyState.style.display = 'none';
        imageViewWrapper.style.display = 'flex';
        telemetryStrip.style.display = 'grid';
        regionInspector.style.display = 'flex';
        resultsFooter.style.display = 'flex';

        // Set Image URLs
        const cacheBuster = `?t=${Date.now()}`;
        displayAnnotated.src = result.annotated_image_url + cacheBuster;
        displayOriginal.src = (result.original_image_url || result.annotated_image_url) + cacheBuster;
        displayAnnotated.classList.add('active');
        displayOriginal.classList.remove('active');

        // Set Tab to Annotated Overlay
        tabBtns.forEach(b => b.classList.remove('active'));
        tabBtns[0].classList.add('active');

        // Telemetry Summary
        const meta = result.image_metadata || {};
        totalRegionsVal.textContent = result.summary ? result.summary.total_regions_detected : (result.regions ? result.regions.length : 0);

        // Substrate Pill
        substratePill.style.display = 'inline-block';
        if (meta.substrate_type === 'palm_leaf') {
            substratePill.textContent = 'Palm-Leaf (Tala-patra)';
            substratePill.style.color = '#e3b341';
            substratePill.style.background = 'rgba(227, 179, 65, 0.15)';
            substratePill.style.borderColor = 'rgba(227, 179, 65, 0.3)';
        } else {
            substratePill.textContent = 'Handmade Paper (Kaghaz)';
            substratePill.style.color = '#58a6ff';
            substratePill.style.background = 'rgba(88, 166, 255, 0.15)';
            substratePill.style.borderColor = 'rgba(88, 166, 255, 0.3)';
        }

        // Download Overlay Link
        downloadOverlayBtn.href = result.annotated_image_url;
        downloadOverlayBtn.download = result.annotated_image_filename || 'manuscript_annotated.png';

        // Render Filter Chips & Regions Table
        renderFilterChips(result.summary ? result.summary.class_distribution : {});
        renderRegionsTable(result.regions || []);
    }

    function renderFilterChips(distribution) {
        classFilterChips.innerHTML = '';
        const allChip = document.createElement('span');
        allChip.className = 'filter-chip active';
        allChip.textContent = 'All Classes';
        allChip.addEventListener('click', () => {
            document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
            allChip.classList.add('active');
            activeFilterClass = 'all';
            renderRegionsTable(currentPredictionData.regions || []);
        });
        classFilterChips.appendChild(allChip);

        const classes = ['main_text', 'header', 'footer', 'side_text', 'filler'];
        classes.forEach(cls => {
            const count = distribution[cls] || 0;
            if (count > 0) {
                const chip = document.createElement('span');
                chip.className = 'filter-chip';
                chip.textContent = `${cls} (${count})`;
                chip.addEventListener('click', () => {
                    document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    activeFilterClass = cls;
                    renderRegionsTable(currentPredictionData.regions || []);
                });
                classFilterChips.appendChild(chip);
            }
        });
    }

    function renderRegionsTable(regions) {
        regionsTableBody.innerHTML = '';
        const filtered = activeFilterClass === 'all' 
            ? regions 
            : regions.filter(r => r.class === activeFilterClass);

        if (filtered.length === 0) {
            regionsTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No regions found for this class.</td></tr>';
            return;
        }

        filtered.forEach(reg => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-family: var(--font-mono); font-weight: 600;">#${reg.id}</td>
                <td><span class="badge badge-${reg.class}">${reg.class}</span></td>
                <td style="font-family: var(--font-mono);">${(reg.confidence * 100).toFixed(0)}%</td>
                <td style="font-family: var(--font-mono); font-size: 0.75rem;">[${reg.bbox.join(', ')}]</td>
                <td style="font-family: var(--font-mono);">${reg.area_px.toLocaleString()}</td>
                <td style="color: var(--text-secondary);">${reg.description || '-'}</td>
            `;
            regionsTableBody.appendChild(tr);
        });
    }
});
