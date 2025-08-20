function postIntent(intent, params) {
  var status = document.getElementById('uiStatus');
  if (status) status.textContent = 'Intent: ' + intent + ' — ' + JSON.stringify(params || {});
  if (window.parent) {
    window.parent.postMessage({ type: 'intent', payload: { intent: intent, params: params || {} } }, '*');
  }
}
window.sendIntent = function(intent, params) {
  postIntent(intent, params || {});
  if (intent === 'start_listening') { setListening(true); setSpeaking(false); }
  if (intent === 'stop') { setListening(false); setSpeaking(false); }
};
window.onVoiceChange = function(voice) {
  postIntent('set_voice', { voice: voice });
  var k = document.getElementById('kpiVoice'); if (k) k.textContent = voice;
};
window.promptSpeak = function() {
  var text = window.prompt('Text to speak');
  if (text && text.trim()) {
    setSpeaking(true); setListening(false);
    var lr = document.getElementById('lastResponse'); if (lr) lr.textContent = text.trim();
    postIntent('speak', { text: text.trim() });
    setTimeout(function(){ setSpeaking(false); }, 1600);
  }
};
function seedWaves(el) {
  if (!el) return; el.innerHTML = '';
  for (var i=0; i<12; i++) { var d = document.createElement('div'); d.className='bar'; el.appendChild(d); }
}
window.setListening = function(on) {
  var k = document.getElementById('kpiListening'); if (k) k.textContent = on ? 'true' : 'false';
  var lw = document.getElementById('listenWaves'); if (lw) lw.style.opacity = on ? '1' : '.35';
};
window.setSpeaking = function(on) {
  var k = document.getElementById('kpiSpeaking'); if (k) k.textContent = on ? 'true' : 'false';
  var sw = document.getElementById('speakWaves'); if (sw) sw.style.opacity = on ? '1' : '.35';
};
seedWaves(document.getElementById('listenWaves'));
seedWaves(document.getElementById('speakWaves'));
var initL = (document.getElementById('kpiListening') && document.getElementById('kpiListening').textContent || 'false') === 'true';
var initS = (document.getElementById('kpiSpeaking') && document.getElementById('kpiSpeaking').textContent || 'false') === 'true';
setListening(initL); setSpeaking(initS);

