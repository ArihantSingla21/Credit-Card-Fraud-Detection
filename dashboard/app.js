const evidenceButtons = document.querySelectorAll('[data-evidence]');
const evidencePanels = document.querySelectorAll('[data-evidence-panel]');
const evidenceNote = document.querySelector('#evidenceNote');

const evidenceNotes = {
  notebook: 'These are findings as stated in the notebook narrative, not freshly recomputed results. The CSV is not included in this project folder.',
  audit: 'Review lens highlights interpretation and evaluation risks visible in the notebook code. These are methodology observations, not new model measurements.'
};

for (const button of evidenceButtons) {
  button.addEventListener('click', () => {
    const selectedEvidence = button.dataset.evidence;

    for (const candidate of evidenceButtons) {
      const isSelected = candidate === button;
      candidate.classList.toggle('is-selected', isSelected);
      candidate.setAttribute('aria-pressed', String(isSelected));
    }

    for (const panel of evidencePanels) {
      panel.hidden = panel.dataset.evidencePanel !== selectedEvidence;
    }

    evidenceNote.textContent = evidenceNotes[selectedEvidence];
  });
}

document.querySelector('#printBrief').addEventListener('click', () => window.print());

const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('main section[id], main[id]');

const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;

    for (const link of navLinks) {
      const isCurrent = link.hash === `#${entry.target.id}`;
      link.classList.toggle('is-active', isCurrent);
      if (isCurrent) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }
}, { rootMargin: '-20% 0px -65% 0px' });

for (const section of sections) observer.observe(section);

const chartCanvas = document.querySelector('#classBalanceChart');
const chartFallback = document.querySelector('#chartFallback');

if (window.Chart && chartCanvas) {
  new Chart(chartCanvas, {
    type: 'bar',
    data: {
      labels: ['Legitimate', 'Fraud'],
      datasets: [{ data: [284315, 492], backgroundColor: ['#385d25', '#e45b49'], borderRadius: 2, barThickness: 25 }]
    },
    options: {
      indexAxis: 'y',
      maintainAspectRatio: false,
      animation: { duration: 650 },
      layout: { padding: { left: 12 } },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (context) => `${context.raw.toLocaleString()} transactions` } }
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 300000,
          grid: { color: '#e8ede8' },
          border: { display: false },
          ticks: { color: '#758078', font: { family: 'IBM Plex Mono', size: 9 }, callback: (value) => value === 0 ? '0' : `${value / 1000}k` }
        },
        y: { grid: { display: false }, border: { display: false }, ticks: { padding: 7, color: '#536058', font: { family: 'Space Grotesk', size: 10 } } }
      }
    }
  });
} else if (chartCanvas && chartFallback) {
  chartCanvas.hidden = true;
  chartFallback.hidden = false;
}