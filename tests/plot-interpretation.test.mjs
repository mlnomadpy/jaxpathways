import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Window } from 'happy-dom';
import { gradientFigure, attentionMaskFigure } from '../src/lib/diagram-markup.js';
import { bindGradientFigure, bindAttentionMaskFigure } from '../src/scripts/lesson-diagrams.js';

const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
const lessons = course.phases.flatMap(p => p.lessons);
const data = id => JSON.parse(fs.readFileSync(`${lessons.find(l=>l.id===id).path}/outputs/visual.json`)).data;
const near = (actual, expected, tolerance=1e-5) => assert(Math.abs(actual-expected)<=tolerance, `${actual} differs from ${expected}`);

test('optimizer walkthrough landmarks describe the recorded paths, not a monotone Adam trajectory', () => {
  const d=data('optimization-12');
  assert.equal(d.yscale,'log');
  assert.equal(d.x[0],0);assert.equal(d.x.at(-1),50);
  const [sgd,momentum,adam]=d.series.map(s=>s.y);
  for(const ys of [sgd,momentum,adam]){near(ys[0],5.5);assert(ys.every(y=>y>0));}
  assert(sgd.every((y,i)=>!i||y<sgd[i-1]));
  near(momentum[2],0.568,0.0005);near(momentum[4],3.714,0.0005);
  near(adam[11],1.45e-4,5e-7);near(adam[19],0.410,0.0005);
  near(adam.at(-1),1.28e-4,5e-7);near(sgd.at(-1),2.96e-3,5e-6);near(momentum.at(-1),2.49e-2,5e-5);
  assert(adam[19]>sgd[19]);assert(adam.at(-1)<sgd.at(-1));assert(sgd.at(-1)<momentum.at(-1));
});

test('gradient-check and curvature captions agree with scales, initial states, and residuals', () => {
  const numerical=data('optimization-02');
  assert.equal(numerical.xscale,'log');assert.equal(numerical.yscale,'log');
  const errors=numerical.series[0].y;
  assert(errors.every(y=>y>0));near(errors[0],3);assert(Math.min(...errors)<3e-6);
  const descent=data('optimization-03');
  assert.equal(descent.yscale,'symlog');assert(descent.series[1].y.slice(1).every(y=>y===0));
  for(let i=1;i<descent.x.length;i++)near(descent.series[2].y[i]/descent.series[2].y[i-1],1.44);
  const curvature=data('optimization-08');
  assert.equal(curvature.x[0],0);near(curvature.series[0].y[0],1);near(curvature.series[0].y[1],0.9);
  near(curvature.series[1].y[0],1);assert(curvature.series[1].y.slice(1).every(y=>y===0));
  const fit=data('optimization-06').series;
  const residuals=fit[1].y.map((y,i)=>y-fit[0].y[i]);
  residuals.forEach((v,i)=>near(v,[-1/3,-1/3,1/3][i]));
  near(residuals.reduce((sum,r)=>sum+r*r,0)/3,1/9);
});

test('attention explanations preserve row normalization, tied keys, and token-specific corrections', () => {
  const focused=data('transformers-01').panels[1].values;
  for(const row of focused)near(row.reduce((a,b)=>a+b),1);
  near(focused[0][0],focused[0][2]);near(focused[1][1],focused[1][2]);
  assert(focused[0][1]<focused[0][0]);assert(focused[1][0]<focused[1][1]);
  const masked=data('transformers-02').values;
  masked.forEach((row,q)=>row.forEach((v,k)=>near(v,k<=q?1/(q+1):0)));
  const residual=data('transformers-03').values;
  for(const row of residual)row.forEach((v,i)=>near(v,residual[0][i]));
  near(residual[0][3],0.0897,0.00005);near(residual[0][7],-0.385,0.0005);
  const tokens=data('transformers-04').values;
  assert.deepEqual(tokens.map(row=>row.indexOf(Math.max(...row))),[2,0,1,2]);
  assert(tokens.every(row=>Math.max(...row)>0.99));
});

test('recovery and deployment comparisons mean equal adjacent bars; p95 comes from the plotted samples', () => {
  for(const [id,a,b] of [['recovery-02',1,2],['distributed-02',0,1],['deployment-03',0,1]]){
    const d=data(id);assert.equal(d.kind,'bar');d.series[a].y.forEach((v,i)=>near(v,d.series[b].y[i]));
  }
  const d=data('deployment-06');const sorted=[...d.series[0].y].sort((a,b)=>a-b);
  assert.equal(sorted.length,30);const at=(sorted.length-1)*0.95;const lo=Math.floor(at);
  const percentile=sorted[lo]+(sorted[lo+1]-sorted[lo])*(at-lo);
  d.series[1].y.forEach(v=>near(v,percentile,1e-12));
  assert.equal(d.ylabel,'end-to-end milliseconds');
  const timing=data('performance-01').panels;
  assert(timing[0].series[0].y[0]>Math.max(...timing[1].series[0].y));
});

test('interactive interpretations change with the selected point and attention boundary, with typeset math', async () => {
  const window=new Window();const old=globalThis.document;globalThis.document=window.document;
  try{
    window.document.body.innerHTML=gradientFigure()+attentionMaskFigure();
    bindGradientFigure();bindAttentionMaskFigure();
    const slider=document.querySelector('#gradient-x');
    for(const [x,word] of [[3,'rises'],[-2,'falls'],[0,'horizontal']]){
      slider.value=String(x);slider.dispatchEvent(new window.Event('input'));
      const status=document.querySelector('#gradient-values');
      assert(status.textContent.includes(word));assert(status.querySelector('math'));
      near(Number(document.querySelector('#gradient-point').getAttribute('cy')),240-x*x*12);
    }
    const mode=document.querySelector('#attention-mode');const last=document.querySelector('#attention-value');
    last.value='32';last.dispatchEvent(new window.Event('input'));
    const rows=()=>[...document.querySelectorAll('#attention-grid tbody tr')].map(tr=>[...tr.querySelectorAll('td')].map(td=>Number(td.textContent)));
    assert.deepEqual(rows()[2],[0.33,0.33,0.33,0]);
    assert(document.querySelector('#attention-result').textContent.includes('stays fixed'));
    mode.value='packed';mode.dispatchEvent(new window.Event('change'));
    assert.deepEqual(rows()[2],[0,0,1,0]);assert.deepEqual(rows()[3],[0,0,0.5,0.5]);
    assert(document.querySelector('#attention-result annotation').textContent.includes('32'));
    mode.value='full';mode.dispatchEvent(new window.Event('change'));
    assert(rows().every(row=>row.every(v=>v===0.25)));
    assert([...document.querySelectorAll('#attention-result annotation')].some(n=>n.textContent.includes('/4=11.500')));
  }finally{globalThis.document=old;await window.happyDOM.abort();}
});
