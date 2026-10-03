function setupChoices(){
  const routeName=id=>course.pathways.find(r=>r.id===id).title;
  const roleName=id=>course.roles.find(r=>r.id===id).title;
  const phaseStart=route=>phaseById(route.phaseIds[0]);
  const sources=route=>`https://github.com/mlnomadpy/jaxpathways/tree/main/phases/${phaseStart(route).number}-${phaseStart(route).id}`;
  if($('#domain-cards')){
    $('#domain-cards').innerHTML=course.domains.map(d=>`<article class="domain-card"><h3>${escapeHtml(d.title)}</h3><ul>${d.skills.map(s=>`<li>${escapeHtml(s)}</li>`).join('')}</ul><p class="domain-careers">${d.roleIds.map(id=>`<a href="pathways.html?role=${id}#roles">${escapeHtml(roleName(id))}</a>`).join(' · ')}</p><div class="domain-links"><button data-domain="${d.id}">Explore domain</button><button data-start-path="${d.defaultPathwayId}">Start path</button></div></article>`).join('');
    $('#domain-cards').querySelectorAll('[data-domain]').forEach(b=>b.onclick=()=>{
      const d=course.domains.find(d=>d.id===b.dataset.domain),detail=$('#domain-detail');
      detail.hidden=false;
      detail.innerHTML=`<p class="edition">${escapeHtml(d.title)}</p><h3>${escapeHtml(d.headline)}</h3><p>${escapeHtml(d.description)}</p><p><strong>You will build:</strong> ${escapeHtml(d.artifact)}</p><p><strong>Career routes:</strong> ${d.roleIds.map(id=>`<a href="pathways.html?role=${id}#roles">${escapeHtml(roleName(id))}</a>`).join(' · ')}</p><p><strong>Choose a pathway:</strong></p><div class="actions">${d.pathwayIds.map(id=>`<button data-choice="${id}">${escapeHtml(routeName(id))}</button>`).join('')}</div>`;
      detail.querySelectorAll('[data-choice]').forEach(button=>button.onclick=()=>openRoute(button.dataset.choice));
      detail.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});
    });
    $('#domain-cards').querySelectorAll('[data-start-path]').forEach(b=>b.onclick=()=>startPath(b.dataset.startPath));
    $('#begin-course').onclick=()=>openPhase('welcome');
  }
  if($('#role-cards')){
    $('#role-cards').innerHTML=course.roles.map(r=>`<article class="role-card"><h3>${escapeHtml(r.title)}</h3><p class="role-headline">${escapeHtml(r.headline)}</p><ul>${r.responsibilities.map(s=>`<li>${escapeHtml(s)}</li>`).join('')}</ul><button data-role="${r.id}">Explore this role</button></article>`).join('');
    function showRole(id){
      const role=course.roles.find(r=>r.id===id);if(!role)return;
      const detail=$('#role-detail');detail.hidden=false;
      detail.innerHTML=`<p class="edition">Career route</p><h3>${escapeHtml(role.title)}</h3><p>${escapeHtml(role.description)}</p><div class="role-columns"><div><h4>Skills you’ll connect</h4><ul>${role.skills.map(s=>`<li>${escapeHtml(s)}</li>`).join('')}</ul></div><div><h4>Your portfolio project</h4><p>${escapeHtml(role.portfolio)}</p><h4>Demonstrate the capability</h4><p>${escapeHtml(role.readiness)}</p></div></div><p><strong>Follow a pathway:</strong></p><div class="actions">${role.pathwayIds.map(id=>`<button data-choice="${id}" class="${id===role.defaultPathwayId?'primary':''}">${escapeHtml(routeName(id))}</button>`).join('')}</div><p class="soft">These are learning goals and planned project briefs. Readiness requires reviewed evidence.</p>`;
      detail.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>openRoute(b.dataset.choice));
      detail.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});
    }
    $('#role-cards').querySelectorAll('[data-role]').forEach(b=>b.onclick=()=>showRole(b.dataset.role));
    const selected=new URLSearchParams(location.search).get('role');if(selected)showRole(selected);
  }
  if($('#routes')){
    $('#routes').innerHTML=course.pathways.filter(r=>r.id!=='foundations').map(r=>`<article class="route"><p class="edition">${escapeHtml(course.domains.find(d=>d.pathwayIds.includes(r.id))?.title||'Focused pathway')}</p><h3>${escapeHtml(r.title)}</h3><p>${escapeHtml(r.description)}</p><p class="route-meta">${r.phaseIds.map(id=>phaseById(id).number).join(' → ')}</p><p><strong>Build:</strong> ${escapeHtml(r.capstone.title)}</p><div class="domain-links"><button data-route="${r.id}">View curriculum</button><button data-start-path="${r.id}">Start path</button><a href="${sources(r)}">GitHub source ↗</a></div></article>`).join('');
    $('#routes').querySelectorAll('[data-route]').forEach(b=>b.onclick=()=>openRoute(b.dataset.route));
    $('#routes').querySelectorAll('[data-start-path]').forEach(b=>b.onclick=()=>startPath(b.dataset.startPath));
  }
}
function startPath(id){
  const route=course.pathways.find(r=>r.id===id);
  if(!$('#curriculum')){location.href=`index.html?path=${encodeURIComponent(id)}&phase=${encodeURIComponent(route.phaseIds[0])}#curriculum`;return;}
  openRoute(id);openPhase(route.phaseIds[0]);
}
