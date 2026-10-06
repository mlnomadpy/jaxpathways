const test=require('node:test');
const assert=require('node:assert/strict');
const {highlight}=require('../src/lib/code.js').CourseCode;
const decoded=html=>html.replace(/<span class="syntax-[a-z]+">|<\/span>/g,'').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
test('Python coloring preserves multiline source and escapes embedded HTML',()=>{
 const source='from jax import grad\n\ndef f(x):\n    text = "<img src=x> # literal"\n    return 2.5e-3 * x  # comment\n';
 const html=highlight(source,'python');
 assert.equal(decoded(html),source);
 assert.ok(html.includes('syntax-keyword'));
 assert.ok(html.includes('syntax-string'));
 assert.ok(html.includes('syntax-comment'));
 assert.ok(!html.includes('<img'));
});
test('PowerShell commands preserve paths and separate flags, strings and comments',()=>{
 const source='.\\.venv\\Scripts\\python.exe -m pip install "jax[cpu]"\n# Restart kernel\n$env:JAX_PLATFORMS = "cpu"';
 const html=highlight(source,'shell');
 assert.equal(decoded(html),source);
 assert.ok(html.includes('syntax-option'));
 assert.ok(html.includes('syntax-variable'));
});
test('Text outputs remain uncolored and literal',()=>{
 const source='x = [-2, 0, 3]\nExpected <not observed>';
 assert.equal(decoded(highlight(source,'text')),source);
 assert.ok(!highlight(source,'text').includes('<span'));
});
