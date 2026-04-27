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
  // If on project detail view, re-render for new language
  if (projectDetailState.active && projectDetailState.rawMarkdown) {
    renderProjectDetail(projectDetailState.rawMarkdown);
  }
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
    // If navigating to projects, reset to list view
    if (tab.dataset.page === 'projects') {
      hideProjectDetail();
    }
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

// ============================================================
// Project Detail View
// ============================================================

var projectDetailState = {
  active: false,
  rawMarkdown: null,
  currentSlug: null
};

// Elements
var projectDetail = document.getElementById('project-detail');
var projectDetailContent = document.getElementById('project-detail-content');
var projectBackBtn = document.getElementById('project-back');
var projectListElements = {
  tabs: document.querySelector('#page-projects > .tabs'),
  panels: document.querySelectorAll('#page-projects > .tab-panel'),
  notes: document.querySelectorAll('#page-projects > .tab-panel .tab-note'),
  title: document.querySelector('#page-projects > h2')
};

// Extract bilingual content from markdown
function extractLanguageContent(markdown, lang) {
  var zhMatch = markdown.match(/<!--\s*zh\s*-->([\s\S]*?)(?=<!--\s*en\s*-->|$)/);
  var enMatch = markdown.match(/<!--\s*en\s*-->([\s\S]*?)(?=<!--\s*zh\s*-->|$)/);

  if (lang === 'zh' && zhMatch) {
    return zhMatch[1].trim();
  }
  if (lang === 'en' && enMatch) {
    return enMatch[1].trim();
  }
  // Fallback: if no language markers, return full content
  return markdown.trim();
}

// Render KaTeX formulas in the rendered HTML
function renderKaTeX(container) {
  if (typeof katex === 'undefined') return;

  // Process block math: $$...$$
  // They appear as <p>$$...$$</p> after marked renders
  var html = container.innerHTML;

  // Block math: $$...$$ (may span multiple lines, may be inside <p> tags)
  html = html.replace(/\$\$([\s\S]*?)\$\$/g, function (match, formula) {
    try {
      return katex.renderToString(formula.trim(), {
        displayMode: true,
        throwOnError: false
      });
    } catch (e) {
      return match;
    }
  });

  // Inline math: $...$ (single line, not preceded/followed by $)
  html = html.replace(/(?<!\$)\$(?!\$)((?:[^$\\]|\\.)+?)\$(?!\$)/g, function (match, formula) {
    try {
      return katex.renderToString(formula.trim(), {
        displayMode: false,
        throwOnError: false
      });
    } catch (e) {
      return match;
    }
  });

  container.innerHTML = html;
}

// Render Mermaid diagrams: marked outputs <pre><code class="language-mermaid">...
// Mermaid expects <div class="mermaid">..., so we convert and trigger mermaid.run()
function renderMermaid(container) {
  if (typeof mermaid === 'undefined') return;

  var mermaidBlocks = container.querySelectorAll('code.language-mermaid, code[class*="language-mermaid"]');
  if (mermaidBlocks.length === 0) return;

  mermaidBlocks.forEach(function (codeEl, idx) {
    var pre = codeEl.parentElement;
    // Decode HTML entities since marked escapes content inside <code>
    var source = codeEl.textContent;
    var div = document.createElement('div');
    div.className = 'mermaid';
    div.textContent = source;
    if (pre && pre.tagName === 'PRE') {
      pre.parentNode.replaceChild(div, pre);
    } else if (codeEl.parentNode) {
      codeEl.parentNode.replaceChild(div, codeEl);
    }
  });

  try {
    mermaid.run({ nodes: container.querySelectorAll('.mermaid') });
  } catch (e) {
    console.warn('Mermaid render failed:', e);
  }
}

// Render markdown content to the detail view
function renderProjectDetail(rawMarkdown) {
  var lang = document.body.classList.contains('lang-zh') ? 'zh' : 'en';
  var content = extractLanguageContent(rawMarkdown, lang);

  // Configure marked
  if (typeof marked !== 'undefined') {
    marked.setOptions({
      breaks: false,
      gfm: true
    });
    var html = marked.parse(content);
    projectDetailContent.innerHTML = html;

    // Fix image paths: relative ./images/ should point to projects/images/
    var images = projectDetailContent.querySelectorAll('img');
    images.forEach(function (img) {
      var src = img.getAttribute('src');
      if (src && src.indexOf('./') === 0) {
        img.setAttribute('src', 'projects/' + src.substring(2));
      }
    });

    // Render KaTeX formulas
    renderKaTeX(projectDetailContent);

    // Render Mermaid diagrams
    renderMermaid(projectDetailContent);
  } else {
    // Fallback: show raw markdown in a pre block
    projectDetailContent.innerHTML = '<pre style="white-space:pre-wrap;">' + content + '</pre>';
  }
}

// Show project detail view
function showProjectDetail(slug) {
  projectDetailState.currentSlug = slug;
  projectDetailState.active = true;

  // Hide project list elements
  projectListElements.tabs.style.display = 'none';
  projectListElements.panels.forEach(function (p) { p.style.display = 'none'; });
  if (projectListElements.title) projectListElements.title.style.display = 'none';

  // Show detail container with loading state
  projectDetail.style.display = 'block';
  var lang = document.body.classList.contains('lang-zh') ? 'zh' : 'en';
  projectDetailContent.innerHTML = '<p class="project-loading">' +
    (lang === 'zh' ? '加载中...' : 'Loading...') + '</p>';

  // Fetch the markdown file
  fetch('projects/' + slug + '.md')
    .then(function (response) {
      if (!response.ok) throw new Error('Not found');
      return response.text();
    })
    .then(function (markdown) {
      projectDetailState.rawMarkdown = markdown;
      renderProjectDetail(markdown);
    })
    .catch(function () {
      var lang = document.body.classList.contains('lang-zh') ? 'zh' : 'en';
      projectDetailContent.innerHTML =
        '<div class="project-placeholder">' +
        '<p>' + (lang === 'zh'
          ? '项目详情即将更新，敬请期待...'
          : 'Project details coming soon, stay tuned...') + '</p>' +
        '</div>';
    });

  window.scrollTo(0, 0);
}

// Hide project detail and restore list view
function hideProjectDetail() {
  projectDetailState.active = false;
  projectDetailState.rawMarkdown = null;
  projectDetailState.currentSlug = null;

  projectDetail.style.display = 'none';
  projectDetailContent.innerHTML = '';

  // Restore project list elements
  projectListElements.tabs.style.display = '';
  projectListElements.panels.forEach(function (p) { p.style.display = ''; });
  if (projectListElements.title) projectListElements.title.style.display = '';
}

// Back button click handler
projectBackBtn.addEventListener('click', function () {
  hideProjectDetail();
  window.scrollTo(0, 0);
});

// Project item click handler
document.querySelectorAll('.project-item[data-project]').forEach(function (item) {
  item.addEventListener('click', function () {
    var slug = item.dataset.project;
    if (slug) {
      showProjectDetail(slug);
    }
  });
});
