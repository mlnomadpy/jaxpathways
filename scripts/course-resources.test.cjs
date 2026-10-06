const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {spawnSync}=require('node:child_process');
const course=JSON.parse(fs.readFileSync('public/curriculum.json'));
const read=path=>JSON.parse(fs.readFileSync(path));
test('catalog and lesson resources preserve authored versus planned status',()=>{
  const catalog=read('public/api/v1/catalog.json');
  const all=catalog.phases.flatMap(p=>p.lessons);
  assert.equal(all.length,course.phases.flatMap(p=>p.lessons).length);
  for(const lesson of all){
    assert.equal('content' in lesson,false,'catalog should not embed full lesson bodies');
    const resource=read(`public/api/v1/lessons/${lesson.id}.json`);
    assert.equal(resource.status,lesson.status);
    assert.equal(Boolean(resource.content),lesson.status==='authored');
    if(lesson.status==='authored'){
      assert.equal(fs.readFileSync(lesson.artifacts.scriptSource,'utf8'),fs.readFileSync('public/'+lesson.artifacts.script,'utf8'));
      assert.deepEqual(read(lesson.artifacts.notebookSource),read('public/'+lesson.artifacts.notebook));
      assert.equal(read(lesson.artifacts.quizSource).questions[0].correct,resource.content.answer);
    }
  }
});
test('public pathway and role resources use the same course manifests',()=>{
  assert.deepEqual(read('public/api/v1/pathways.json').pathways,course.pathways);
  assert.deepEqual(read('public/api/v1/roles.json').roles,course.roles);
  assert.equal(read('public/openapi.json').openapi,'3.1.0');
});
test('CLI rejects unknown IDs and planned execution without running a program',()=>{
  const path=require('node:path');
  const folder=fs.mkdtempSync(path.join(require('node:os').tmpdir(),'course-cli-fixture-'));
  try {
    fs.mkdirSync(path.join(folder,'scripts'));
    fs.mkdirSync(path.join(folder,'public'));
    fs.copyFileSync('scripts/course.py',path.join(folder,'scripts/course.py'));
    fs.writeFileSync(path.join(folder,'public/curriculum.json'),JSON.stringify({phases:[{lessons:[{id:'future-lesson',status:'planned'}]}]}));
    for(const id of ['../../README.md','future-lesson']){
      const result=spawnSync('python3',['scripts/course.py','run',id],{encoding:'utf8',cwd:folder});
      assert.equal(result.status,2);
      assert.equal(result.stdout,'');
      assert.match(result.stderr,id==='future-lesson'?/planned brief/:/Unknown lesson/);
    }
  } finally {fs.rmSync(folder,{recursive:true,force:true});}
});
test('CLI role plan retains prerequisites and labels planned lessons',()=>{
  const result=spawnSync('python3',['scripts/course.py','plan','inference-engineer'],{encoding:'utf8'});
  assert.equal(result.status,0);
  assert.match(result.stdout,/welcome-01/);
  assert.match(result.stdout,/deployment/);
  assert.match(result.stdout,/Planned lessons are not authored labs/);
  assert.match(result.stdout,/executable evidence/);
});
