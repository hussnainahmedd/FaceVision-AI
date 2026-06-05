/* ============================================================
   FaceVision AI — Frontend Interactivity
   Vanilla JS · No frameworks
   ============================================================ */

(() => {
  'use strict';

  // ── State ──────────────────────────────────────────────────
  let previousFaces = [];
  let statsInterval = null;
  let toastCount = 0;

  // ── DOM References ─────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    faceCount:        $('#face-count'),
    faceCountBadge:   $('#face-count-badge'),
    detectedList:     $('#detected-faces-list'),
    emptyState:       $('#empty-state'),
    registerToggle:   $('#register-toggle'),
    toggleIcon:       $('#toggle-icon'),
    formContainer:    $('#register-form-container'),
    registerForm:     $('#register-form'),
    uploadZone:       $('#upload-zone'),
    fileInput:        $('#face-image-input'),
    previewContainer: $('#image-preview-container'),
    imagePreview:     $('#image-preview'),
    removePreview:    $('#remove-preview'),
    nameInput:        $('#face-name-input'),
    registerBtn:      $('#register-btn'),
    facesGallery:     $('#faces-gallery'),
    toastContainer:   $('#toast-container'),
  };

  // ============================================================
  //  1. Stats Polling — every 500ms
  // ============================================================
  function startStatsPolling() {
    fetchStats();
    statsInterval = setInterval(fetchStats, 500);
  }

  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) return;
      const data = await res.json();
      updateDetectionPanel(data);
    } catch {
      // Silently ignore network errors
    }
  }

  function updateDetectionPanel(data) {
    const faces = data.faces || [];
    const count = faces.length;

    // Update count with bounce animation
    const prevCount = parseInt(dom.faceCount.textContent, 10) || 0;
    if (prevCount !== count) {
      dom.faceCount.textContent = count;
      dom.faceCount.classList.add('bounce');
      setTimeout(() => dom.faceCount.classList.remove('bounce'), 250);
    }

    // Badge
    dom.faceCountBadge.textContent = `${count} face${count !== 1 ? 's' : ''}`;

    // Empty state
    if (count === 0) {
      dom.emptyState.style.display = '';
      // Remove remaining face cards with exit animation
      const oldCards = dom.detectedList.querySelectorAll('.face-card');
      oldCards.forEach((card) => {
        card.classList.add('removing');
        card.addEventListener('animationend', () => card.remove(), { once: true });
      });
      previousFaces = [];
      return;
    }

    dom.emptyState.style.display = 'none';

    // Build a map of current face cards by name
    const currentCardMap = {};
    dom.detectedList.querySelectorAll('.face-card').forEach((card) => {
      currentCardMap[card.dataset.name] = card;
    });

    const newFaceNames = new Set(faces.map((f) => f.name || 'Unknown'));

    // Remove faces no longer present
    Object.keys(currentCardMap).forEach((name) => {
      if (!newFaceNames.has(name)) {
        const card = currentCardMap[name];
        card.classList.add('removing');
        card.addEventListener('animationend', () => card.remove(), { once: true });
      }
    });

    // Add or update face cards
    faces.forEach((face, index) => {
      const name = face.name || 'Unknown';
      const confidence = Math.round((face.confidence || 0) * 100);
      const age = face.age ? Math.round(face.age) : null;
      const gender = face.gender || null;

      const existing = currentCardMap[name];
      if (existing) {
        // Update existing card
        updateFaceCard(existing, name, confidence, age, gender);
      } else {
        // Create new card
        const card = createFaceCard(name, confidence, age, gender, index);
        dom.detectedList.appendChild(card);
      }
    });

    previousFaces = faces;
  }

  function createFaceCard(name, confidence, age, gender, index) {
    const card = document.createElement('div');
    card.className = 'face-card';
    card.dataset.name = name;
    card.style.animationDelay = `${index * 0.06}s`;

    card.innerHTML = buildFaceCardHTML(name, confidence, age, gender);
    return card;
  }

  function updateFaceCard(card, name, confidence, age, gender) {
    // Update confidence text
    const confEl = card.querySelector('.face-card-confidence');
    if (confEl) confEl.textContent = `${confidence}%`;

    // Update confidence bar smoothly via CSS transition
    const bar = card.querySelector('.confidence-bar-fill');
    if (bar) bar.style.width = `${confidence}%`;

    // Update meta
    const meta = card.querySelector('.face-card-meta');
    if (meta) meta.innerHTML = buildMetaHTML(age, gender);
  }

  function buildFaceCardHTML(name, confidence, age, gender) {
    const isUnknown = name === 'Unknown';
    return `
      <div class="face-card-header">
        <span class="face-card-name ${isUnknown ? 'unknown' : ''}">${escapeHtml(name)}</span>
        <span class="face-card-confidence">${confidence}%</span>
      </div>
      <div class="confidence-bar-track">
        <div class="confidence-bar-fill" style="width: ${confidence}%"></div>
      </div>
      <div class="face-card-meta">
        ${buildMetaHTML(age, gender)}
      </div>
    `;
  }

  function buildMetaHTML(age, gender) {
    let parts = [];
    if (age !== null && age !== undefined) {
      parts.push(`<span>🎂 ${age} yrs</span>`);
    }
    if (gender) {
      const icon = gender.toLowerCase() === 'male' ? '♂' : gender.toLowerCase() === 'female' ? '♀' : '⚧';
      parts.push(`<span>${icon} ${capitalize(gender)}</span>`);
    }
    return parts.join('');
  }

  // ============================================================
  //  2. Face Registration
  // ============================================================

  // --- Drag & Drop ---
  function setupDragAndDrop() {
    const zone = dom.uploadZone;

    zone.addEventListener('click', () => dom.fileInput.click());

    zone.addEventListener('dragenter', (e) => {
      e.preventDefault();
      zone.classList.add('drag-over');
    });

    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].type.startsWith('image/')) {
        dom.fileInput.files = files;
        showImagePreview(files[0]);
      }
    });

    dom.fileInput.addEventListener('change', () => {
      const file = dom.fileInput.files[0];
      if (file) showImagePreview(file);
    });

    dom.removePreview.addEventListener('click', clearPreview);
  }

  function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      dom.imagePreview.src = e.target.result;
      dom.previewContainer.style.display = '';
      dom.uploadZone.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  function clearPreview() {
    dom.imagePreview.src = '';
    dom.previewContainer.style.display = 'none';
    dom.uploadZone.style.display = '';
    dom.fileInput.value = '';
  }

  // --- Form Submission ---
  function setupRegistrationForm() {
    dom.registerForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      const file = dom.fileInput.files[0];
      const name = dom.nameInput.value.trim();

      if (!file) {
        showToast('Please select an image first.', 'error');
        return;
      }
      if (!name) {
        showToast('Please enter a name.', 'error');
        return;
      }

      dom.registerBtn.disabled = true;
      dom.registerBtn.textContent = 'Registering...';

      try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('name', name);

        const res = await fetch('/register_face', {
          method: 'POST',
          body: formData,
        });

        const data = await res.json();

        if (res.ok && data.success) {
          showToast(`"${name}" registered successfully!`, 'success');
          clearPreview();
          dom.nameInput.value = '';
          loadGallery();
        } else {
          showToast(data.message || 'Registration failed.', 'error');
        }
      } catch (err) {
        showToast('Network error. Please try again.', 'error');
      } finally {
        dom.registerBtn.disabled = false;
        dom.registerBtn.textContent = 'Register Face';
      }
    });
  }

  // ============================================================
  //  3. Registered Faces Gallery
  // ============================================================
  async function loadGallery() {
    try {
      const res = await fetch('/api/faces');
      if (!res.ok) return;
      const data = await res.json();
      renderGallery(data.faces || data || []);
    } catch {
      // Silently ignore
    }
  }

  function renderGallery(faces) {
    if (!faces.length) {
      dom.facesGallery.innerHTML = '<p class="empty-state">No faces registered yet</p>';
      return;
    }

    dom.facesGallery.innerHTML = '';
    faces.forEach((face, i) => {
      const card = document.createElement('div');
      card.className = 'gallery-face-card';
      card.style.animationDelay = `${i * 0.06}s`;

      const name = face.name || face;
      const thumbSrc = face.image
        ? face.image
        : `/api/faces/${encodeURIComponent(name)}/thumbnail`;

      card.innerHTML = `
        <img class="face-thumb" src="${thumbSrc}" alt="${escapeHtml(name)}" onerror="this.style.display='none'">
        <div class="face-info">
          <span class="face-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
          <button class="delete-btn" data-name="${escapeHtml(name)}" title="Delete">🗑</button>
        </div>
      `;

      card.querySelector('.delete-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteFace(name);
      });

      dom.facesGallery.appendChild(card);
    });
  }

  async function deleteFace(name) {
    if (!confirm(`Delete "${name}" from registered faces?`)) return;

    try {
      const res = await fetch(`/api/faces/${encodeURIComponent(name)}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        showToast(`"${name}" deleted.`, 'info');
        loadGallery();
      } else {
        const data = await res.json().catch(() => ({}));
        showToast(data.message || 'Failed to delete.', 'error');
      }
    } catch {
      showToast('Network error.', 'error');
    }
  }

  // ============================================================
  //  4. Toast Notification System
  // ============================================================
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toastCount++;

    const icons = {
      success: '✅',
      error: '❌',
      info: 'ℹ️',
    };

    toast.innerHTML = `
      <span class="toast-icon">${icons[type] || icons.info}</span>
      <span>${escapeHtml(message)}</span>
    `;

    dom.toastContainer.appendChild(toast);

    // Auto-dismiss after 3s
    setTimeout(() => {
      toast.classList.add('removing');
      toast.addEventListener('animationend', () => toast.remove(), { once: true });
    }, 3000);
  }

  // ============================================================
  //  5. Registration Panel Toggle
  // ============================================================
  function setupPanelToggle() {
    dom.registerToggle.addEventListener('click', () => {
      const isOpen = dom.formContainer.classList.toggle('open');
      dom.toggleIcon.classList.toggle('open', isOpen);
    });
  }

  // ============================================================
  //  Utilities
  // ============================================================
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  }

  // ============================================================
  //  6. Webcam Snapshot Registration
  // ============================================================
  function setupWebcamRegister() {
    const btn = document.getElementById('webcam-register-btn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
      const name = dom.nameInput.value.trim();
      if (!name) {
        showToast('Please enter a name first.', 'error');
        dom.nameInput.focus();
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Capturing...';

      try {
        const res = await fetch('/register_webcam', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });

        const data = await res.json();

        if (res.ok && data.success) {
          showToast(`"${name}" registered from webcam!`, 'success');
          dom.nameInput.value = '';
          clearPreview();
          loadGallery();
        } else {
          showToast(data.message || 'Registration failed.', 'error');
        }
      } catch {
        showToast('Network error.', 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Capture from Webcam';
      }
    });
  }

  // ============================================================
  //  7. Initialize on DOMContentLoaded
  // ============================================================
  document.addEventListener('DOMContentLoaded', () => {
    loadGallery();
    startStatsPolling();
    setupDragAndDrop();
    setupRegistrationForm();
    setupPanelToggle();
    setupWebcamRegister();
  });
})();
