import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

test('browser modules have an acyclic dependency graph and domain libraries do not import controllers',()=>{
  const files=fs.readdirSync('src',{recursive:true}).filter(file=>file.endsWith('.js')).map(file=>path.resolve('src',file));
  const graph=new Map();
  for(const file of files){
    const source=fs.readFileSync(file,'utf8');
    const edges=[...source.matchAll(/from\s+['"](\.[^'"]+)['"]/g)].map(match=>path.resolve(path.dirname(file),match[1])).filter(dependency=>dependency.endsWith('.js'));
    if(file.includes('/src/lib/'))for(const edge of edges)assert(!edge.includes('/src/scripts/'),`domain library imports a browser controller: ${file} → ${edge}`);
    graph.set(file,edges);
  }
  const complete=new Set(),active=[];
  function visit(file){
    assert(!active.includes(file),`circular module dependency: ${[...active,file].map(file=>path.relative(process.cwd(),file)).join(' → ')}`);
    if(complete.has(file))return;
    active.push(file);for(const dependency of graph.get(file)||[])visit(dependency);active.pop();complete.add(file);
  }
  files.forEach(visit);
});
