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

// Top navigation page switching
document.querySelectorAll('.nav-tab').forEach(function (tab) {
  tab.addEventListener('click', function (e) {
    e.preventDefault();
    // Deactivate all nav tabs and pages
    document.querySelectorAll('.nav-tab').forEach(function (t) { t.classList.remove('active'); });
    document.querySelectorAll('.page').forEach(function (p) { p.classList.remove('active'); });
    // Activate clicked tab and its page
    tab.classList.add('active');
    var pageId = 'page-' + tab.dataset.page;
    document.getElementById(pageId).classList.add('active');
    // Scroll to top
    window.scrollTo(0, 0);
  });
});

// Project sub-tab switching
document.querySelectorAll('.tab-btn').forEach(function (btn) {
  btn.addEventListener('click', function () {
    document.querySelectorAll('.tab-btn').forEach(function (b) { b.classList.remove('active'); });
    document.querySelectorAll('.tab-panel').forEach(function (p) { p.classList.remove('active'); });
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// Nav name click goes to About page
document.querySelector('.nav-name').addEventListener('click', function (e) {
  e.preventDefault();
  document.querySelectorAll('.nav-tab').forEach(function (t) { t.classList.remove('active'); });
  document.querySelectorAll('.page').forEach(function (p) { p.classList.remove('active'); });
  document.querySelector('[data-page="about"]').classList.add('active');
  document.getElementById('page-about').classList.add('active');
  window.scrollTo(0, 0);
});
