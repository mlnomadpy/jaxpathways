function setupPlatform(routes) {
  const key = 'jaxpathways-notebook-v1';
  let state = {version:1,route:'foundations',evidence:[]};
  const valid = (s) => s && s.version===1 && routes.some(r=>r.id===s.route) && Array.isArray(s.evidence) && s.evidence.length<=500 && s.evidence.every(e=>typeof e.id==='string' && e.id.length<100 && routes.some(r=>r.id===e.route) && typeof e.title==='string' && e.title.length<=120 && typeof e.note==='string' && e.note.length<=4000 && typeof e.url==='string' && e.url.length<=2000 && (!e.url || /^https?:\/\//i.test(e.url)) && typeof e.date==='string' && e.date.length<60);
  try {const saved=JSON.parse(localStorage.getItem(key));if(valid(saved))state=saved;}catch{}
  const save=()=>{try{localStorage.setItem(key,JSON.stringify(state));return true;}catch{$('#backup-status').textContent='Device storage is unavailable. Export your notebook before closing this page.';return false;}};
  const download=(name,text,type)=>{const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  for(const selector of ['#active-route','#evidence-route','#club-route']){
    const select=$(selector); for(const route of routes){const option=document.createElement('option');option.value=route.id;option.textContent=route.title;select.append(option);}
  }
  $('#active-route').value=state.route;$('#club-route').value='tpu';
  function renderEvidence(){
    $('#evidence-count').textContent=state.evidence.length?`${state.evidence.length} saved artifact${state.evidence.length===1?'':'s'} · self-reported, awaiting review`:'No evidence saved yet. Your first experiment is a good place to start.';
    $('#evidence-list').replaceChildren();
    for(const entry of state.evidence.slice().reverse()){
      const article=document.createElement('article');article.className='evidence-entry';
      const meta=document.createElement('p');meta.className='edition';meta.textContent=routes.find(r=>r.id===entry.route).title+' / '+entry.date;
      const title=document.createElement('h3');title.textContent=entry.title;
      const note=document.createElement('p');note.textContent=entry.note;
      article.append(meta,title,note);
      if(entry.url){const link=document.createElement('a');link.href=entry.url;link.textContent='Open evidence ↗';link.rel='noopener noreferrer';link.target='_blank';article.append(link);}
      const remove=document.createElement('button');remove.textContent='Remove';remove.onclick=()=>{state.evidence=state.evidence.filter(e=>e.id!==entry.id);save();renderEvidence();$('#evidence-status').textContent='Evidence removed from this notebook.';};article.append(remove);$('#evidence-list').append(article);
    }
  }
  function routeSummary(){const r=routes.find(r=>r.id===state.route);$('#route-next').textContent=r.prerequisites+'. Your goal: '+r.outcome+'.';}
  $('#active-route').onchange=e=>{state.route=e.target.value;save();routeSummary();};
  $('#continue').onclick=()=>openRoute(state.route);
  $('#evidence-form').onsubmit=e=>{e.preventDefault();const url=$('#evidence-url').value.trim();if(url&&!/^https?:\/\//i.test(url)){$('#evidence-status').textContent='Use an http or https evidence link.';return;}state.evidence.push({id:crypto.randomUUID(),route:$('#evidence-route').value,title:$('#evidence-title').value.trim(),url,note:$('#evidence-note').value.trim(),date:new Date().toISOString().slice(0,10)});save();renderEvidence();e.target.reset();$('#evidence-status').textContent='Evidence saved as self-reported work. It does not award a competency or certificate.';};
  $('#export').onclick=()=>download('jaxpathways-notebook.json',JSON.stringify({...state,sampleCompleted:completed},null,2),'application/json');
  $('#import').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>3000000)throw Error();const incoming=JSON.parse(await file.text());if(!valid(incoming)||typeof incoming.sampleCompleted!=='boolean')throw Error();const ids=new Set(state.evidence.map(e=>e.id));const merged=[...state.evidence,...incoming.evidence.filter(e=>!ids.has(e.id))];if(merged.length>500)throw Error();state.evidence=merged;state.route=incoming.route;completed=completed||incoming.sampleCompleted;try{localStorage.setItem('jaxpathways-first-gradient',completed);}catch{}save();$('#active-route').value=state.route;renderEvidence();routeSummary();progress();$('#backup-status').textContent='Backup imported. Existing evidence was preserved.';}catch{$('#backup-status').textContent='This is not a supported JAX Pathways backup. Your saved notebook was not changed.';}e.target.value='';};
  let planText='';
  function renderClub(){
    const route=routes.find(r=>r.id===$('#club-route').value),weeks=Number($('#club-weeks').value);
    const units=route.phaseIds.map(id=>({title:phaseById(id).number+' / '+phaseById(id).title,evidence:phaseById(id).project}));
    units.push({title:'Final project: '+route.capstone.title,evidence:route.capstone.assessment});
    $('#club-plan').replaceChildren();
    planText=`# ${route.title}: ${weeks}-week learning club\n\nDraft facilitator plan. Lessons and labs require authoring and validation. A short schedule assumes prior knowledge or substantial independent study; do not interpret it as a time-to-mastery promise.\nProject: ${route.outcome}\n\n`;
    for(let i=0;i<weeks;i++){
      const start=Math.floor(i*units.length/weeks),end=Math.floor((i+1)*units.length/weeks),assigned=units.slice(start,end);
      const article=document.createElement('article'),h=document.createElement('h3');h.textContent=`Week ${i+1}`;article.append(h);
      const titles=assigned.length?assigned.map(u=>u.title):['Practice and review the previous phase'];
      const p=document.createElement('p');p.textContent=titles.join('; ');article.append(p);
      const note=document.createElement('p');note.className='soft';note.textContent=assigned.length?'Evidence: '+assigned.map(u=>u.evidence).join('; '):'Explain, modify, and diagnose the previous experiment with a partner.';
      article.append(note);$('#club-plan').append(article);
      planText+=`## Week ${i+1}\n${titles.map(t=>'- '+t).join('\n')}\n\n${note.textContent}\n\n`;
    }
  }
  $('#club-route').onchange=renderClub;$('#club-weeks').onchange=renderClub;$('#club-download').onclick=()=>download('jaxpathways-club-plan.md',planText,'text/markdown');
  renderClub();renderEvidence();routeSummary();
}
