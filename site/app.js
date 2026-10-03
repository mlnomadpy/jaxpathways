const $ = (s) => document.querySelector(s);
const dialog = $('#lesson');
let course = null;
let curriculum = [];
let activePath = 'all';
let completed = false;
try { completed = localStorage.getItem('jaxpathways-first-gradient') === 'true'; } catch {}
const example = 'from jax import grad\n\ndef f(x):\n    return x ** 2\n\nprint(grad(f)(3.0))  # 6.0';
const escapeHtml = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function phaseById(id) { return course.phases.find(p=>p.id===id); }
function lessonById(id) { for(const p of course.phases){const lesson=p.lessons.find(l=>l.id===id);if(lesson)return {phase:p,lesson};} }
function progress() { if(!$('#progress'))return; $('#progress').textContent = `${completed ? 1 : 0} / 1 sample completed · saved in this browser`; }
function showLesson(phase, lesson) {
  const sample = lesson.status === 'sample';
  const index = phase.lessons.findIndex(l=>l.id===lesson.id);
  const prerequisite = index ? phase.lessons[index-1].title : phase.prerequisitePhaseIds.map(id=>phaseById(id).title).join('; ') || 'Basic Python; begin with the setup phase';
  $('#lesson-body').innerHTML = `<p class="edition">Phase ${phase.number} / ${escapeHtml(phase.title)}</p><h2>${escapeHtml(lesson.title)}</h2><p class="lesson-status">${sample ? 'Authored sample lesson' : 'Planned lesson · learning brief'}</p><p>${escapeHtml(lesson.objective)}</p><p class="soft"><strong>Before this:</strong> ${escapeHtml(prerequisite)}<br><strong>Hardware:</strong> ${escapeHtml(phase.hardware)}</p>` + (sample ? `<h3>The idea</h3><p>A derivative tells you how fast an output changes as its input changes. For f(x) = x², the derivative is 2x. At x = 3, that gives 6. JAX’s grad takes a function and returns a new function that computes its derivative.</p><h3>Build it</h3><p>In a Python environment with JAX installed, run:</p><pre><code>${example}</code></pre><p>Expected output: <code>6.0</code>.</p><h3>Make it yours</h3><p>Replace x ** 2 with x ** 3. Predict the derivative at 3 before running it. You should get 27.</p><h3>Check your understanding</h3><p>Does grad(f) return a number or a function? Why do we pass 3.0 instead of 3? What happens to the derivative at x = 0?</p><details><summary>Reveal the explanation</summary><p>grad(f) returns a function. We pass a floating-point input because ordinary grad differentiates floating-point or complex inputs. For x² the derivative at zero is zero.</p></details><h3>Keep your evidence</h3><p>Save your original function, the changed function, and both outputs. Explain why they differ.</p><button class="primary" id="complete">${completed ? 'Mark as unfinished' : 'I ran it and can explain it'}</button>` : `<p class="soft">The lesson narrative and runnable exercise are still being authored. This brief defines its place in the course.</p><h3>Build and modify</h3><p>${escapeHtml(lesson.exercise)}</p><h3>Keep your evidence</h3><p>${escapeHtml(lesson.evidence)}</p><h3>Checkpoint</h3><p>${escapeHtml(lesson.check)}</p><h3>How it fits</h3><p>This lesson contributes to the phase project: <strong>${escapeHtml(phase.project)}</strong>.</p>`);
  $('#lesson-body').insertAdjacentHTML('beforeend',`<p><a href="${escapeHtml(phase.source)}">Primary documentation ↗</a></p><div class="lesson-nav">${index ? `<button id="previous">Previous: ${escapeHtml(phase.lessons[index-1].title)}</button>` : '<span>First lesson in this phase</span>'}${index<phase.lessons.length-1?`<button id="next">Next: ${escapeHtml(phase.lessons[index+1].title)}</button>`:'<button id="phase-project">View the phase project</button>'}</div>`);
  if(sample)$('#complete').onclick=()=>{completed=!completed;try{localStorage.setItem('jaxpathways-first-gradient',completed);}catch{}progress();$('#complete').textContent=completed?'Mark as unfinished':'I ran it and can explain it';};
  if($('#previous'))$('#previous').onclick=()=>showLesson(phase,phase.lessons[index-1]);
  if($('#next'))$('#next').onclick=()=>showLesson(phase,phase.lessons[index+1]);
  if($('#phase-project'))$('#phase-project').onclick=()=>{dialog.close();openPhase(phase.id);};
  if(!dialog.open)dialog.showModal();
  dialog.scrollTop=0;
  $('.close').focus();
}
function openPhase(id) {
  if(!$('#curriculum')){location.href=`index.html?phase=${encodeURIComponent(id)}#curriculum`;return;}
  const d=document.getElementById(`phase-${id}`);
  if(d.hidden){activePath='all';$('#course-route').value='all';}
  $('#search').value='';filter('');d.open=true;
  d.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
}
function openRoute(id) {
  if(!$('#curriculum')){location.href=`index.html?path=${encodeURIComponent(id)}#curriculum`;return;}
  activePath=id;$('#course-route').value=id;$('#search').value='';filter('');
  $('#curriculum').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});
}
function filter(query) {
  const route=curriculum.find(r=>r.id===activePath);
  let visible=0;
  for(const phase of course.phases){
    const d=document.getElementById(`phase-${phase.id}`);
    const routeMatch=!route||route.phaseIds.includes(phase.id);
    const phaseMatch=`${phase.title} ${phase.project} ${phase.part}`.toLowerCase().includes(query);
    let matches=0;
    for(const lesson of phase.lessons){const match=routeMatch&&(phaseMatch||`${lesson.title} ${lesson.objective}`.toLowerCase().includes(query));document.getElementById(lesson.id).hidden=!match;matches+=Number(match);}
    d.hidden=matches===0;if(query&&matches)d.open=true;visible+=Number(matches>0);
  }
  $('#empty').hidden=visible!==0;
  const summary=$('#path-summary');summary.hidden=!route;
  if(route) summary.innerHTML=`<p class="edition">Your route through the shared course</p><h3>${escapeHtml(route.title)}</h3><p>${route.phaseIds.map(id=>`<button class="phase-chip" data-phase="${id}">${phaseById(id).number} ${escapeHtml(phaseById(id).title)}</button>`).join('')}</p><p><strong>Final project:</strong> ${escapeHtml(route.capstone.title)}</p><p><strong>Demonstrate:</strong> ${escapeHtml(route.capstone.assessment)}</p><p class="soft">Project and assessment brief · planned. Earlier phases are prerequisites, not extra notebook collections.</p>`;
  summary.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>openPhase(b.dataset.phase));
  $('#course-view-status').textContent=route?`${visible} ${visible===1?'phase':'phases'} on this pathway`:`${visible} ${visible===1?'phase':'phases'} in the course`;
}
async function init(){
  try{
    const response=await fetch('curriculum.json');if(!response.ok)throw Error('Curriculum unavailable');
    course=await response.json();
    curriculum=course.pathways.map(r=>({...r,lessons:r.phaseIds.flatMap(id=>phaseById(id).lessons)}));
    if($('#phases')){
    $('#phases').innerHTML=course.phases.map(p=>`<details id="phase-${p.id}" class="phase-card"><summary><span class="num">${p.number}</span><span>${escapeHtml(p.title)}<small>${escapeHtml(p.part)}</small></span><span class="badge">${p.lessons.length} lessons</span></summary><div class="route-details"><p><strong>Start after:</strong> ${p.prerequisitePhaseIds.length?p.prerequisitePhaseIds.map(id=>`<button class="inline-phase" data-phase="${id}">${phaseById(id).number} ${escapeHtml(phaseById(id).title)}</button>`).join(' '):'Basic Python. No previous JAX experience needed.'}</p><p><strong>Hardware:</strong> ${escapeHtml(p.hardware)}</p><ul class="lessons">${p.lessons.map((l,i)=>`<li id="${l.id}"><button class="lesson-link" data-lesson="${l.id}"><span><span class="lesson-number">${p.number}.${String(i+1).padStart(2,'0')}</span>${escapeHtml(l.title)}</span><span class="badge ${l.status}">${l.status==='sample'?'Read sample ↗':'Planned ↗'}</span></button></li>`).join('')}</ul><div class="phase-project"><p class="edition">Bring the phase together</p><h3>${escapeHtml(p.project)}</h3><p><strong>Checkpoint:</strong> ${escapeHtml(p.checkpoint)}</p><p class="soft">Project brief · planned</p></div></div></details>`).join('');
    const total=course.phases.reduce((n,p)=>n+p.lessons.length,0);
    $('#counts').textContent=`${course.phases.length} phases · ${total} lesson briefs`;
    for(const route of curriculum){const option=document.createElement('option');option.value=route.id;option.textContent=route.title;$('#course-route').append(option);}
    document.querySelectorAll('[data-route]').forEach(b=>b.onclick=()=>openRoute(b.dataset.route));
    document.querySelectorAll('[data-phase]').forEach(b=>b.onclick=()=>openPhase(b.dataset.phase));
    document.querySelectorAll('[data-lesson]').forEach(b=>b.onclick=()=>{const {phase,lesson}=lessonById(b.dataset.lesson);showLesson(phase,lesson);});
    if($('#start'))$('#start').onclick=()=>openPhase('welcome');
    if($('#try-gradient'))$('#try-gradient').onclick=()=>{const {phase,lesson}=lessonById('first-gradient');showLesson(phase,lesson);};
    $('#search').oninput=e=>filter(e.target.value.trim().toLowerCase());
    $('#course-route').onchange=e=>{activePath=e.target.value;filter($('#search').value.trim().toLowerCase());};
    const params=new URLSearchParams(location.search);
    if(curriculum.some(r=>r.id===params.get('path'))){activePath=params.get('path');$('#course-route').value=activePath;}
    progress();filter('');
    if(phaseById(params.get('phase')))openPhase(params.get('phase'));
    }
    setupChoices();
    document.querySelectorAll('[data-route]').forEach(b=>b.onclick=()=>openRoute(b.dataset.route));
    if($('#learner'))setupPlatform(curriculum);
  }catch(error){if($('#phases'))$('#phases').textContent='The curriculum could not load. Refresh the page or run npm run dev.';console.error(error);}
}
$('.close').onclick=()=>dialog.close();
if($('#copy'))$('#copy').onclick=async()=>{try{await navigator.clipboard.writeText(example);$('#copy-status').textContent='Copied. Paste it into your Python session.';}catch{$('#copy-status').textContent='Select the code and copy it manually.';}};
init();
