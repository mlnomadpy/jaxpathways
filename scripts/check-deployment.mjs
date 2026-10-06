import { spawnSync } from 'node:child_process';
const base='/jaxpathways';
function run(script,args=[],env=process.env){
 const result=spawnSync(process.execPath,[script,...args],{stdio:'inherit',env});
 if(result.status!==0)throw Error(`${script} failed (${result.status ?? result.error})`);
}
try {
 run('node_modules/astro/bin/astro.mjs',['build'],{...process.env,BASE_PATH:base});
 run('scripts/check-site.mjs',[],{...process.env,BASE_PATH:base});
 run('scripts/check-seo.mjs',[],{...process.env,BASE_PATH:base});
 run('tests/browser-migration.test.mjs',[],{...process.env,BASE_PATH:base});
} finally {
 // Leave the normal local preview build in place, even when a subpath check fails.
 run('node_modules/astro/bin/astro.mjs',['build'],{...process.env,BASE_PATH:'/'});
}
