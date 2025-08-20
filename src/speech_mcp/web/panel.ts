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


