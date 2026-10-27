import {createRequire} from 'node:module';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
const require=createRequire('/private/tmp/jaxpathways-teaching-lineage-review/package.json');
const {chromium}=require('@playwright/test');
const out='/private/tmp/jaxpathways-full-print-016/output/pdf';
await mkdir(out,{recursive:true});
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const browser=await chromium.launch({headless:true});
try {
 const page=await browser.newPage({viewport:{width:1280,height:900}});
 const url='http://127.0.0.1:4200/downloads/jax-foundations.html';
 const sourceResponse=await page.request.get(url);
 if(!sourceResponse.ok())throw Error('Source book unavailable');
 const source=await sourceResponse.body();
 const response=await page.goto(url,{waitUntil:'load',timeout:90000});
 if(!response?.ok())throw Error('Source book navigation failed');
 if(sha(source)!=='06f851086a28ac27aff6611543efa048d6deda7ee03db7f8e24cbdd18735c96a')throw Error('Frozen source hash changed');
 await page.evaluate(()=>{for(const image of document.images)image.loading='eager';});
 await page.waitForFunction(()=>document.fonts.status==='loaded'&&Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0),null,{timeout:90000});
 await page.emulateMedia({media:'print',colorScheme:'light'});
 await page.waitForFunction(()=>Array.from(document.querySelectorAll('details:not(.solution-reveal)')).every(d=>d.open),null,{timeout:30000});
 const state=await page.evaluate(()=>({
  articles:document.querySelectorAll('article').length,
  phaseSections:document.querySelectorAll('main > section').length,
  images:document.images.length,
  math:document.querySelectorAll('.katex').length,
  allImagesLoaded:Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0),
  inlineSolutionsHidden:Array.from(document.querySelectorAll('.solution-reveal')).every(e=>getComputedStyle(e).display==='none'),
  appendicesVisible:Array.from(document.querySelectorAll('.print-solutions')).every(e=>getComputedStyle(e).display==='block')
 }));
 if(state.articles!==104||!state.inlineSolutionsHidden||!state.appendicesVisible)throw Error('Full-book print mode incomplete: '+JSON.stringify(state));
 const artifacts=[];
 for(const format of ['A4','Letter']){
  console.log('Rendering full '+format+' book');
  const filename='jax-pathways-0.16.0-full-'+format.toLowerCase()+'.pdf';
  const bytes=await page.pdf({path:join(out,filename),format,scale:1,printBackground:true,displayHeaderFooter:false,preferCSSPageSize:false,tagged:true,outline:true,timeout:240000});
  artifacts.push({format,filename,bytes:bytes.length,sha256:sha(bytes)});
  console.log('Wrote '+filename+' ('+bytes.length+' bytes)');
 }
 const receipt={recordedAt:new Date().toISOString(),edition:'0.16.0',sourceUrl:url,sourceHtmlSha256:sha(source),browser:await browser.version(),printState:state,settings:{renderer:'Playwright headless Chromium page.pdf',scale:1,margins:'Frozen book CSS @page 18mm',printBackground:true,headerFooter:false,tagged:true,outline:true},artifacts,scope:'Entire frozen HTML book, unpruned. Native reader and assistive-technology acceptance remain separate.',visualReview:'Pending geometry and raster inspection.'};
 await writeFile(join(out,'print-receipt.json'),JSON.stringify(receipt,null,2)+'\n');
}finally{await browser.close();}
