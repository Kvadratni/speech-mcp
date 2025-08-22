function netLog(message: string) {
  const el = document.getElementById('netLog'); if (!el) return;
  el.textContent = `${new Date().toISOString()} ${message}\n` + el.textContent;
}

// Send a prompt to the host (Goose will route it to the agent)
function postPrompt(text: string) {
  netLog(`PROMPT ${text}`);
  const status = document.getElementById('uiStatus');
  if (status) status.textContent = `Prompt: ${text}`;
  try {
    const parent = (window as any).parent;
    if (!parent) return;
    // Primary (mcp-ui): payload.prompt
    parent.postMessage({ type: 'prompt', payload: { prompt: text } }, '*');
  } catch {}
}


// Expose to inline HTML
// @ts-ignore
window.startListening = async () => {
  (window as any).setListening(true); (window as any).setSpeaking(false);
  try { postPrompt('listen'); } catch {}
};
// @ts-ignore
window.stopAll = async () => { try { postPrompt('stop'); } catch {} };
// @ts-ignore
window.onVoiceChange = async (voice: string) => {
  const k = document.getElementById('kpiVoice'); if (k) k.textContent = voice;
  try { postPrompt(`set_voice ${voice}`); } catch {}
};
// @ts-ignore
window.onThemeChange = (theme: string) => {
  applyTheme(theme);
};
// @ts-ignore
window.promptSpeak = () => {
  const text = window.prompt('Text to speak');
  if (text && text.trim()) {
    (window as any).setSpeaking(true); (window as any).setListening(false);
    const lr = document.getElementById('lastResponse'); if (lr) lr.textContent = text.trim();
    postPrompt(`speak ${text.trim()}`);
  }
};

// Say from inline input
// @ts-ignore
window.sayFromInput = () => {
  const input = document.getElementById('sayInput') as HTMLInputElement | null;
  const text = (input?.value || '').trim();
  if (!text) return;
  (window as any).setSpeaking(true); (window as any).setListening(false);
  const lr = document.getElementById('lastResponse'); if (lr) lr.textContent = text;
  postPrompt(`speak ${text}`);
};

// Start/Stop conversation convenience (enable loop and listen immediately)
// @ts-ignore
window.startConversation = async () => {
  const btn = document.getElementById('convBtn') as HTMLButtonElement | null;
  postPrompt('listen');
  if (btn) { btn.textContent = 'Listening…'; setTimeout(() => { if (btn) btn.textContent = 'Start Conversation'; }, 1000); }
};

// Boot log for BASE/TOKEN visibility
try {
  const base = (window as any).SPEECH_BASE;
  const token = (window as any).SPEECH_TOKEN;
  if (base) netLog(`BOOT BASE=${base}`);
  if (token) netLog(`BOOT TOKEN=${token}`);
} catch {}

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
    const mq: any = window.matchMedia('(prefers-color-scheme: dark)');
    if ((mq as any) && 'addEventListener' in (mq as any)) {
      mq.addEventListener('change', (ev: any) => {
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

// Show a simple UI version so you can verify reloads
try {
  const verEl = document.getElementById('uiVer');
  if (verEl) {
    const ts = new Date().toISOString().replace('T',' ').replace('Z','');
    verEl.textContent = `UI v0.35 (${ts})`;
  }
} catch {}


// MCP-UI: report size changes to host (ui-size-change with size-change fallback)
function postSizeToHost(): void {
  const height = document.documentElement.scrollHeight;
  const width = document.documentElement.scrollWidth;
  const payload = { height, width };
  if (window.parent) {
    // Use the canonical mcp-ui size message; avoid legacy 'size-change' to prevent UNKNOWN_ACTION
    window.parent.postMessage({ type: 'ui-size-change', payload }, '*');
  }
}

(function initSizeObserver() {
  let rafScheduled = false;
  const schedulePost = () => {
    if (rafScheduled) return;
    rafScheduled = true;
    requestAnimationFrame(() => {
      rafScheduled = false;
      postSizeToHost();
    });
  };

  if ('ResizeObserver' in window) {
    const ro = new ResizeObserver(() => schedulePost());
    ro.observe(document.documentElement);
    ro.observe(document.body);
  } else {
    (window as any).addEventListener('resize', schedulePost);
  }

  document.addEventListener('DOMContentLoaded', schedulePost);
  window.addEventListener('load', schedulePost);
  setTimeout(schedulePost, 0);
})();

