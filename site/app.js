const $ = (s) => document.querySelector(s);
const dialog = $('#lesson');
let curriculum = [];
let completed = false;
try { completed = localStorage.getItem('jaxpathways-first-gradient') === 'true'; } catch {}
const example = 'from jax import grad\n\ndef f(x):\n    return x ** 2\n\nprint(grad(f)(3.0))  # 6.0';
function progress() { $('#progress').textContent = `${completed ? 1 : 0} / 1 sample completed · saved in this browser`; }
function showLesson(route, lesson) {
  const sample = lesson.status === 'sample';
  $('#lesson-body').innerHTML = `<p class="edition">${sample ? 'Foundation / Sample lesson' : 'Curriculum / Planned lesson'}</p><h2>${lesson.title}</h2>` + (sample ? `<p>A derivative tells you how fast an output changes as its input changes. Let’s ask JAX to find one.</p><h3>Before you begin</h3><p>You need basic Python and a local Python environment. For this CPU experiment, install JAX in a virtual environment:</p><pre><code>python3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install jax</code></pre><h3>A function you already understand</h3><p>For f(x) = x², the derivative is 2x. At x = 3, that gives 6. JAX’s grad takes a function and returns a new function that computes its derivative.</p><pre><code>${example}</code></pre><h3>Make it yours</h3><p>Replace x ** 2 with x ** 3. Predict the derivative at 3 before running it. You should get 27.</p><h3>Check your understanding</h3><p>Does grad(f) return a number or a function? Why do we pass 3.0 instead of 3? What happens to the derivative at x = 0?</p><details><summary>Reveal the explanation</summary><p>grad(f) returns a function. We pass a floating-point input because ordinary grad differentiates floating-point or complex inputs. For x² the derivative at zero is zero.</p></details><p><a href="https://docs.jax.dev/en/latest/automatic-differentiation.html">Continue with the official autodiff guide ↗</a></p><button class="primary" id="complete">${completed ? 'Mark as unfinished' : 'I ran it and can explain it'}</button>` : `<p>This lesson is part of the proposed <strong>${route.title}</strong> pathway. The full explanation and runnable lab are still being written.</p><h3>Bring this with you</h3><p>${route.prerequisites}.</p><h3>What this route builds toward</h3><p>${route.outcome}.</p><h3>Explore while it takes shape</h3><p><a href="${route.source}">Read the primary documentation ↗</a></p><p>Start with the working gradient sample if you’re new to JAX.</p><button id="sample" class="primary">Open the first gradient</button>`);
  if (sample) $('#complete').onclick = () => { completed = !completed; try {localStorage.setItem('jaxpathways-first-gradient', completed);} catch {} progress(); $('#complete').textContent = completed ? 'Mark as unfinished' : 'I ran it and can explain it'; };
  else $('#sample').onclick = () => showLesson(curriculum[0], curriculum[0].lessons[0]);
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
}
function openRoute(id) {
  const d = document.getElementById(`route-${id}`);
  $('#search').value = '';
  filter('');
  d.open = true;
  d.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
}
function filter(query) {
  let visible = 0;
  for (const route of curriculum) {
    const d = document.getElementById(`route-${route.id}`);
    const routeMatch = `${route.title} ${route.description} ${route.libraries}`.toLowerCase().includes(query);
    let matches = 0;
    for (const lesson of route.lessons) {
      const match = routeMatch || lesson.title.toLowerCase().includes(query);
      document.getElementById(lesson.id).hidden = !match;
      matches += Number(match);
    }
    d.hidden = matches === 0;
    if (query && matches) d.open = true;
    visible += Number(matches > 0);
  }
  $('#empty').hidden = visible !== 0;
}
async function init() {
  try {
    const response = await fetch('curriculum.json');
    if (!response.ok) throw new Error('Could not load curriculum');
    curriculum = await response.json();
    const symbols = ['⌘','∿','∴','↺','⇶','λ','▤','↗'];
    $('#routes').innerHTML = curriculum.slice(1).map((r,i) => `<button class="route" data-route="${r.id}"><span class="route-symbol">${symbols[i]}</span><h3>${r.title}</h3><p>${r.description}</p><span class="route-meta">${r.libraries} ↗</span></button>`).join('');
    $('#phases').innerHTML = curriculum.map((r,i) => `<details id="route-${r.id}"><summary><span class="num">0${i}</span>${r.title}<span class="badge">${r.lessons.length} topics</span></summary><div class="route-details"><p><strong>Prerequisites:</strong> ${r.prerequisites}.</p><p><strong>You’ll build:</strong> ${r.outcome}.</p><a href="${r.source}">Primary documentation ↗</a><ul class="lessons">${r.lessons.map(l=>`<li id="${l.id}"><button class="lesson-link" data-lesson="${l.id}">${l.title}<span class="badge ${l.status}">${l.status === 'sample' ? 'Open sample ↗' : 'Planned · outline ↗'}</span></button></li>`).join('')}</ul></div></details>`).join('');
    const total = curriculum.reduce((n,r)=>n+r.lessons.length,0);
    $('#counts').textContent = `${curriculum.length} pathways · ${total} proposed topics · 1 working sample`;
    document.querySelectorAll('[data-route]').forEach(b=>b.onclick=()=>openRoute(b.dataset.route));
    document.querySelectorAll('[data-lesson]').forEach(b=>b.onclick=()=> {const r=curriculum.find(r=>r.lessons.some(l=>l.id===b.dataset.lesson));showLesson(r,r.lessons.find(l=>l.id===b.dataset.lesson));});
    $('#start').onclick=()=>showLesson(curriculum[0],curriculum[0].lessons[0]);
    $('#search').oninput=e=>filter(e.target.value.trim().toLowerCase());
    progress();
    setupPlatform(curriculum);
  } catch { $('#phases').textContent='The curriculum could not load. Refresh the page or run the site with npm run dev.'; }
}
$('.close').onclick=()=>dialog.close();
$('#copy').onclick=async()=> {try {await navigator.clipboard.writeText(example);$('#copy-status').textContent='Copied. Paste it into your Python session.';} catch {$('#copy-status').textContent='Select the code and copy it manually.';}};
init();
