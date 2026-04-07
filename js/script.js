// Language detection and switching
function detectLanguage() {
  var stored = localStorage.getItem('lang');
  if (stored) return stored;
  return 'en';
}

function setLanguage(lang) {
  document.body.classList.toggle('lang-zh', lang === 'zh');
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  localStorage.setItem('lang', lang);
}

setLanguage(detectLanguage());

document.getElementById('langToggle').addEventListener('click', function () {
  var isZh = document.body.classList.contains('lang-zh');
  setLanguage(isZh ? 'en' : 'zh');
});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    // Deactivate all tabs and panels
    document.querySelectorAll('.tab-btn').forEach(function (b) { b.classList.remove('active'); });
    document.querySelectorAll('.tab-panel').forEach(function (p) { p.classList.remove('active'); });
    // Activate clicked tab and its panel
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});
