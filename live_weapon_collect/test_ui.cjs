// Optional developer test: Node + existing jsdom, no network or browser required.
// SPLATOON_JSDOM_PATH can point to an already installed jsdom package.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { JSDOM } = require(process.env.SPLATOON_JSDOM_PATH || 'jsdom');
const html = fs.readFileSync(__dirname + '/ui.html', 'utf8');
const keys = ['left0','left1','left2','left3','right0','right1','right2','right3'];
const geometry = Object.fromEntries(keys.map(k => [k,[.3,.067,.052,.092]]));
function makeMatch(id) {
  return {capture:{match_id:id,session_id:'s1',source:'0',created_at:'2026-10-02T00:00:00Z',status:'complete',warnings:[],
    frames:Array.from({length:5},(_,i)=>({ordinal:i,frame_index:1200+i*60,timestamp:20+i,remaining_seconds:300-i}))},
    annotation:{schema_version:1,match_id:id,ally_side:'right',reference_ordinal:0,geometry:structuredClone(geometry),
      slots:Object.fromEntries(keys.map(k=>[k,{weapon:null,status:'unknown',reviewed:false}])),revision:0,
      confirmed:false,rejected:false,opening_reviewed:false,notes:'',created_at:'2026-10-02T00:00:00Z',updated_at:'2026-10-02T00:00:00Z'}};
}
const matches = new Map([['m1',makeMatch('m1')]]);
const weapons = [{name:'Sploosh-o-matic',display_name:'ボールドマーカー',category:'シューター'},
  {name:'Splash-o-matic',display_name:'シャープマーカー',category:'シューター'},
  {name:'new-weapon',display_name:'未収集の武器',category:'ローラー'}];
let writes = 0, delayNextSave = false, releaseSave;
let delayNextCandidates=false,releaseCandidates;
const image = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==';
async function fetchMock(path, options={}) {
  const url = new URL(path,'http://127.0.0.1:8780');
  const data = options.body ? JSON.parse(options.body) : {};
  const ok = value => ({ok:true,json:async()=>structuredClone(value)});
  const fail = error => ({ok:false,json:async()=>({error})});
  if(url.pathname === '/api/state') return ok({collector:{status:'running',phase:'seeking_intro'},can_capture:true,
    candidate_model:{engine:'hybrid',configuration:'blend_soft',weapon_count:2,available:true},
    matches:Array.from(matches.values(),m=>({match_id:m.capture.match_id,created_at:m.capture.created_at,frame_count:m.capture.frames.length,
      capture_status:m.capture.status,...m.annotation})),weapons,
    session_dir:'mock',catalog_path:'wepons.json',resume:{match_id:'m1'}});
  if(url.pathname === '/api/match') return ok(matches.get(url.searchParams.get('match_id')));
  if(url.pathname === '/api/coverage') {
    const target=Number(url.searchParams.get('target'));
    const confirmed=Array.from(matches.values()).filter(m=>m.annotation.confirmed&&!m.annotation.rejected);
    const rows=weapons.map(w=>{const count=confirmed.filter(m=>Object.values(m.annotation.slots).some(s=>s.reviewed&&s.status==='labeled'&&s.weapon===w.name)).length;return {...w,match_count:count,video_match_count:0,live_match_count:count,incomplete_match_count:0,remaining_matches:Math.max(0,target-count),status:count===0?'missing':count<target?'insufficient':'at_target'};});
    return ok({target,weapons:rows,warnings:[],sources:[{exists:true,directory:'mock',annotation_files:matches.size}],counting_note:'distinct human matches',summary:{weapons_total:rows.length,weapons_missing:rows.filter(w=>!w.match_count).length,weapons_below_target:rows.filter(w=>w.remaining_matches).length,weapons_at_target:rows.filter(w=>!w.remaining_matches).length,unique_confirmed_matches:confirmed.length}});
  }
  if(url.pathname === '/api/resume') return ok({ok:true});
  if(url.pathname === '/api/frame') return ok({frame:image,hud:image,metadata:matches.get(data.match_id).capture.frames[data.ordinal],
    slots:keys.map(slot=>({slot,image,state:'alive'}))});
  if(url.pathname === '/api/candidates') {
    if(delayNextCandidates){delayNextCandidates=false;await new Promise(resolve=>releaseCandidates=resolve);}
    return ok({slots:keys.map(slot=>({slot,candidates:[
    {weapon:'Sploosh-o-matic',display_name:'ボールドマーカー',score:.8},
    {weapon:'Splash-o-matic',display_name:'シャープマーカー',score:.7}]}))});
  }
  if(url.pathname === '/api/save') {
    if(delayNextSave){delayNextSave=false;await new Promise(resolve=>releaseSave=resolve);}
    const m = matches.get(data.match_id);
    if(data.revision!==m.annotation.revision)return fail('revision conflict');
    for(const [key,s] of Object.entries(data.slots))if(s.weapon&&!['Sploosh-o-matic','Splash-o-matic'].includes(s.weapon))return fail(key+': 無効な武器名（保存していません）');
    if(data.confirmed&&(!data.opening_reviewed||!Object.values(data.slots).every(s=>s.reviewed)))return fail('人力確認が必要です');
    m.annotation={...data,revision:data.revision+1};writes++;return ok(m.annotation);
  }
  throw Error('Unexpected route '+url.pathname);
}
const dom = new JSDOM(html,{url:'http://127.0.0.1:8780/',runScripts:'dangerously',pretendToBeVisual:true,
  beforeParse(w){w.fetch=fetchMock;w.structuredClone=structuredClone;w.setInterval=()=>0;}});
const w = dom.window;
async function eventually(fn){for(let i=0;i<100;i++){if(fn())return;await new Promise(r=>setTimeout(r,10));}throw Error('UI timeout');}
function type(id,value){const e=w.document.getElementById(id);e.value=value;e.dispatchEvent(new w.Event('input',{bubbles:true}));}
(async()=>{
  await eventually(()=>!w.document.getElementById('editor').hidden);
  assert.equal(w.document.querySelectorAll('.slot').length,8);
  assert.equal(w.document.querySelectorAll('#weaponNames option').length,3);
  assert.equal(w.document.querySelector('#weaponNames option').label,'ボールドマーカー');
  assert.match(w.document.getElementById('candidateModel').textContent,/hybrid blend_soft.*2武器/);
  await w.refreshCoverage();
  assert.equal(w.document.querySelectorAll('#coverageRows tr').length,3);
  assert.equal(w.document.querySelector('[data-weapon="Sploosh-o-matic"] td').textContent,'ボールドマーカー');
  await w.document.getElementById('candidates').onclick();
  assert.equal(w.document.querySelector('#candidates0 .candidate').textContent,'ボールドマーカー 0.800');
  assert.equal(matches.get('m1').annotation.slots.left0.weapon,null);
  type('weapon0','ボールドマーカー');
  w.document.getElementById('weapon0').dispatchEvent(new w.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
  assert.equal(w.document.activeElement.id,'weapon1');
  await w.save();
  assert.equal(matches.get('m1').annotation.slots.left0.weapon,'Sploosh-o-matic');
  await w.openMatch('m1');
  assert.equal(w.document.getElementById('weapon0').value,'ボールドマーカー');
  assert.equal(matches.get('m1').annotation.slots.left1.reviewed,false);
  type('weapon1','シューター');
  const before=writes;
  await assert.rejects(w.save(),/無効な武器名/);
  assert.equal(writes,before);
  assert.equal(w.document.getElementById('weapon1').value,'シューター');
  type('weapon1','シャープマーカー');
  await w.save();
  // Another capture does not change the current match or typed draft.
  matches.set('m2',makeMatch('m2'));
  type('notes','試合終了後に入力中');
  await w.refresh();
  assert.equal(w.eval('current.capture.match_id'),'m1');
  assert.equal(w.document.getElementById('notes').value,'試合終了後に入力中');
  await w.save();
  // New edits during an in-flight save must be persisted before navigation.
  type('notes','first');delayNextSave=true;
  const pending=w.save();
  await eventually(()=>!!releaseSave);
  type('notes','latest');releaseSave();await pending;
  assert.equal(matches.get('m1').annotation.notes,'latest');
  assert.equal(w.eval('dirty'),false);
  await w.openMatch('m2');
  assert.equal(w.eval('current.capture.match_id'),'m2');
  assert.equal(matches.get('m1').annotation.notes,'latest');
  for(let i=0;i<8;i++){const s=w.document.getElementById('status'+i);s.value='unknown';s.dispatchEvent(new w.Event('change'));}
  const review=w.document.getElementById('reviewOpening');review.checked=true;review.dispatchEvent(new w.Event('input'));
  await w.document.getElementById('confirm').onclick();
  assert.equal(matches.get('m2').annotation.confirmed,true);
  assert.equal(w.document.getElementById('error').textContent,'');
  for(const id of ['m3','m4']){const m=makeMatch(id);m.annotation.confirmed=true;for(const s of Object.values(m.annotation.slots))Object.assign(s,{weapon:'Sploosh-o-matic',status:'labeled',reviewed:true});matches.set(id,m);}
  await w.refreshCoverage();
  assert.equal(w.document.querySelector('[data-weapon="Sploosh-o-matic"] .count').textContent,'2試合');
  const target=w.document.getElementById('coverageTarget');target.value='2';target.dispatchEvent(new w.Event('change'));
  await eventually(()=>w.eval('coverage.target')===2);
  assert.equal(w.document.querySelector('[data-weapon="Sploosh-o-matic"]'),null);
  const filter=w.document.getElementById('coverageFilter');filter.value='all';filter.dispatchEvent(new w.Event('change'));
  assert.equal(w.document.querySelector('[data-weapon="Sploosh-o-matic"] .count').textContent,'2試合');
  type('coverageSearch','sploosh');
  assert.equal(w.document.querySelectorAll('#coverageRows tr').length,1);
  assert.match(w.document.getElementById('coverageCsv').href,/target=2&missing_only=1/);
  type('coverageSearch','');filter.value='zero';filter.dispatchEvent(new w.Event('change'));
  assert.equal(w.document.querySelectorAll('#coverageRows tr').length,2);
  const category=w.document.getElementById('coverageCategory');category.value='ローラー';category.dispatchEvent(new w.Event('change'));
  assert.equal(w.document.querySelectorAll('#coverageRows tr').length,1);
  assert.equal(w.eval('current.capture.match_id'),'m2');
  delayNextCandidates=true;
  const candidatesPending=w.document.getElementById('candidates').onclick();
  await eventually(()=>!!releaseCandidates);
  await w.stepFrame(1);releaseCandidates();await candidatesPending;
  assert.equal(w.document.querySelector('#candidates0 .candidate'),null);
  assert.match(w.document.getElementById('candidateState').textContent,/表示画像が変わりました/);
  console.log('UI logic OK: labeling, autosave, no capture hijack; coverage counts, 2/3 targets, missing/search/category filters, CSV');
  w.close();
})().catch(e=>{console.error(e);w.close();process.exitCode=1;});
