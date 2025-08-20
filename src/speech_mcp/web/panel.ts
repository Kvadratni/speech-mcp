function postIntent(intent: string, params: Record<string, unknown> = {}) {
  const status = document.getElementById('uiStatus');
  if (status) status.textContent = `Intent: ${intent} — ${JSON.stringify(params)}`;
  if (window.parent) {
    window.parent.postMessage({ type: 'intent', payload: { intent, params } }, '*');
  }
}

// Expose to inline HTML
// @ts-ignore
window.sendIntent = (intent: string, params: Record<string, unknown> = {}) => {
  postIntent(intent, params);
  if (intent === 'start_listening') { setListening(true); setSpeaking(false); }
  if (intent === 'stop') { setListening(false); setSpeaking(false); }
};
// @ts-ignore
window.onVoiceChange = (voice: string) => {
  postIntent('set_voice', { voice });
  const k = document.getElementById('kpiVoice'); if (k) k.textContent = voice;
};
// @ts-ignore
window.onThemeChange = (theme: string) => {
  applyTheme(theme);
};
// @ts-ignore
window.promptSpeak = () => {
  const text = window.prompt('Text to speak');
  if (text && text.trim()) {
    setSpeaking(true); setListening(false);
    const lr = document.getElementById('lastResponse'); if (lr) lr.textContent = text.trim();
    postIntent('speak', { text: text.trim() });
    setTimeout(() => setSpeaking(false), 1600);
  }
};

function seedWaves(el: HTMLElement | null) {
  if (!el) return;
  el.innerHTML = '';
  for (let i = 0; i < 12; i++) {
    const d = document.createElement('div');
    d.className = 'bar';
    el.appendChild(d);
  }
}

// @ts-ignore
window.setListening = (on: boolean) => {
  const k = document.getElementById('kpiListening'); if (k) k.textContent = on ? 'true' : 'false';
  const lw = document.getElementById('listenWaves') as HTMLElement | null; if (lw) lw.style.opacity = on ? '1' : '.35';
};
// @ts-ignore
window.setSpeaking = (on: boolean) => {
  const k = document.getElementById('kpiSpeaking'); if (k) k.textContent = on ? 'true' : 'false';
  const sw = document.getElementById('speakWaves') as HTMLElement | null; if (sw) sw.style.opacity = on ? '1' : '.35';
};

// Initialize waves
seedWaves(document.getElementById('listenWaves'));
seedWaves(document.getElementById('speakWaves'));

// Initialize KPIs from server-injected values already in DOM
// @ts-ignore
const initL = (document.getElementById('kpiListening')?.textContent || 'false') === 'true';
// @ts-ignore
const initS = (document.getElementById('kpiSpeaking')?.textContent || 'false') === 'true';
// @ts-ignore
window.setListening(initL);
// @ts-ignore
window.setSpeaking(initS);

function applyTheme(theme: string) {
  const root = document.documentElement;
  const themes: Record<string, Record<string, string>> = {
    midnight: {
      '--bg': '#0a0f1e', '--panel': '#111729', '--text': '#e8ecf1', '--muted': '#99a3b3', '--border': '#20283b',
      '--accent': '#6ea8fe', '--ok': '#38e1a8', '--dim': '#56607a'
    },
    glacier: {
      '--bg': '#f6f7fb', '--panel': '#ffffff', '--text': '#0d1220', '--muted': '#5c6785', '--border': '#e4e8f0',
      '--accent': '#3b82f6', '--ok': '#10b981', '--dim': '#9aa3b2'
    },
    emerald: {
      '--bg': '#0c1210', '--panel': '#111814', '--text': '#e7f0eb', '--muted': '#97a39c', '--border': '#1d2a23',
      '--accent': '#34d399', '--ok': '#22c55e', '--dim': '#5a6a61'
    }
  };
  const t = themes[theme] || themes.midnight;
  Object.entries(t).forEach(([k, v]) => root.style.setProperty(k, v));
}

// pick a default theme
detectAndApplySystemTheme();

function detectAndApplySystemTheme() {
  const prefersDark = typeof window.matchMedia === 'function' && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = prefersDark ? 'midnight' : 'glacier';
  applyTheme(theme);
  const sel = document.getElementById('themeSelect') as HTMLSelectElement | null;
  if (sel) sel.value = theme;
  // Watch for changes
  if (typeof window.matchMedia === 'function') {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    if ('addEventListener' in mq) {
      mq.addEventListener('change', ev => {
        const t = ev.matches ? 'midnight' : 'glacier';
        applyTheme(t);
        const s = document.getElementById('themeSelect') as HTMLSelectElement | null; if (s) s.value = t;
      });
    } else if ('addListener' in mq) {
      // Safari
      // @ts-ignore
      mq.addListener((ev: MediaQueryListEvent) => {
        const t = ev.matches ? 'midnight' : 'glacier';
        applyTheme(t);
        const s = document.getElementById('themeSelect') as HTMLSelectElement | null; if (s) s.value = t;
      });
    }
  }
}


