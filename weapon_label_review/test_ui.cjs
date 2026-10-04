// Optional developer-only DOM test; no camera, sockets or application dependencies.
const assert=require('node:assert/strict'),fs=require('node:fs');
const {JSDOM}=require(process.env.SPLATOON_JSDOM_PATH||'jsdom');
const weapons=[{name:'Sploosh-o-matic',display_name:'ボールドマーカー',category:'シューター'},
 {name:'Splash-o-matic',display_name:'シャープマーカー',category:'シューター'},
 {name:'__unassigned__',display_name:'未指定',category:'その他'}];
const rows=new Map([
 ['a',{item_id:'a',source_id:'s',match_id:'m1',slot:'left0',kind:'live',weapon:'Sploosh-o-matic',status:'labeled',revision:1,checked:false}],
 ['b',{item_id:'b',source_id:'s',match_id:'m2',slot:'left1',kind:'video',weapon:'Sploosh-o-matic',status:'labeled',revision:1,checked:false}],
 ['c',{item_id:'c',source_id:'s',match_id:'m3',slot:'right0',kind:'live',weapon:'Splash-o-matic',status:'labeled',revision:1,checked:false}]]);
let corrections=0,checks=0,delayCheck=false,releaseCheck,undo=null;
const ok=v=>({ok:true,json:async()=>structuredClone(v)}),fail=error=>({ok:false,json:async()=>({error})});
function state(){return {weapons:weapons.map(w=>{const group=[...rows.values()].filter(r=>r.weapon===w.name);return {...w,slots:group.length,matches:group.length,checked:group.filter(r=>r.checked).length,remaining:group.filter(r=>!r.checked).length,by_kind:Object.fromEntries(['live','video'].map(k=>[k,{slots:group.filter(r=>r.kind===k).length,matches:group.filter(r=>r.kind===k).length,remaining:group.filter(r=>r.kind===k&&!r.checked).length}]))};}),items:rows.size,checked:[...rows.values()].filter(r=>r.checked).length,sources:[],warnings:[],data_dir:'mock review',resume:{weapon:'Sploosh-o-matic',item_id:'a'}};}
async function fetchMock(path,options={}){
 const url=new URL(path,'http://127.0.0.1:8783'),body=options.body?JSON.parse(options.body):{};
 if(url.pathname==='/api/state')return ok(state());
 if(url.pathname==='/api/items')return ok([...rows.values()].filter(r=>r.weapon===url.searchParams.get('weapon')));
 if(url.pathname==='/api/item'){const r=rows.get(url.searchParams.get('item_id'));return ok({...r,display_name:weapons.find(w=>w.name===r.weapon).display_name,label:{weapon:r.weapon,status:r.status},reference_ordinal:0,frames:[0,1,2,3,4].map(i=>({ordinal:i,timestamp:20+i,frame_index:1200+i*60}))});}
 if(url.pathname==='/api/resume')return ok({ok:true});
 if(url.pathname==='/api/check'){
  if(delayCheck){delayCheck=false;await new Promise(resolve=>releaseCheck=resolve);}
  const r=rows.get(body.item_id);if(r.revision!==body.revision)return fail('別画面で更新されています');
  r.checked=true;checks++;return ok({ok:true});
 }
 if(url.pathname==='/api/correct'){
  const r=rows.get(body.item_id);if(r.revision!==body.revision)return fail('別画面で更新されています');
  if(body.status==='labeled'&&!weapons.some(w=>w.name===body.weapon))return fail('無効な武器名');
  undo=structuredClone(r);r.weapon=body.status==='labeled'?body.weapon:'__unassigned__';r.status=body.status;r.revision++;r.checked=true;corrections++;return ok({operation:'0123456789abcdef0123456789abcdef',revision:r.revision});
 }
 if(url.pathname==='/api/undo'){rows.set(undo.item_id,{...undo,revision:undo.revision+2});return ok({ok:true});}
 throw Error('unexpected route '+url.pathname);
}
const dom=new JSDOM(fs.readFileSync(__dirname+'/ui.html','utf8'),{url:'http://127.0.0.1:8783/',runScripts:'dangerously',pretendToBeVisual:true,beforeParse(w){w.fetch=fetchMock;w.structuredClone=structuredClone;w.confirm=()=>true;}}),w=dom.window;
const $=id=>w.document.getElementById(id);
async function eventually(f){for(let i=0;i<100;i++){if(f())return;await new Promise(r=>setTimeout(r,10));}throw Error('UI timeout');}
function type(id,value){$(id).value=value;$(id).dispatchEvent(new w.Event('input',{bubbles:true}));}
(async()=>{
 await eventually(()=>!$('editor').hidden);
 assert.equal($('weaponInput').value,'ボールドマーカー');
 assert.equal(w.document.querySelectorAll('.card').length,2);
 assert.match($('crop').src,/ordinal=0/);
 w.stepFrame(1);assert.match($('crop').src,/ordinal=1/);
 w.stepFrame(-1);
 type('weaponInput','シューター');await w.saveCorrection();
 assert.equal(corrections,0);assert.match($('error').textContent,/無効な武器名/);
 type('weaponInput','シャープマーカー');await w.saveCorrection();
 assert.equal(rows.get('a').weapon,'Splash-o-matic');assert.equal(corrections,1);
 assert.equal(w.eval('current.item_id'),'b');assert.equal($('undo').disabled,false);
 await w.undo();assert.equal(rows.get('a').weapon,'Sploosh-o-matic');
 assert.equal(w.document.querySelectorAll('.card').length,2);
 await w.selectItem('b');
 delayCheck=true;const pending=w.mark();await eventually(()=>!!releaseCheck);
 assert.equal($('nextWeapon').disabled,true);releaseCheck();await pending;
 assert.equal(checks,1);assert.equal(w.eval('current.item_id'),'a');
 assert.equal(w.document.querySelectorAll('.card').length,1);
 await w.mark();assert.equal($('editor').hidden,true);
 w.document.body.dispatchEvent(new w.KeyboardEvent('keydown',{key:'n',bubbles:true}));
 await eventually(()=>w.eval('weapon')==='Splash-o-matic'&&w.eval('current?.item_id')==='c');
 assert.equal($('weaponInput').value,'シャープマーカー');
 const filter=$('sourceFilter');filter.value='video';filter.dispatchEvent(new w.Event('change'));
 assert.equal(w.document.querySelectorAll('.card').length,0);
 const show=$('showChecked');show.checked=true;show.dispatchEvent(new w.Event('change'));
 await w.openWeapon('Sploosh-o-matic');assert.equal(w.document.querySelectorAll('.card').length,1);
  assert.equal(w.eval('current.item_id'),'b');
  rows.get('b').revision++;
  await assert.rejects(w.mark(),/別画面で更新/);
  assert.equal(checks,2);
  assert.equal($('nextWeapon').disabled,false);
 console.log('Review UI OK: Japanese grouping, frame stepping, invalid labels, correction/undo, busy guard, next weapon and source filters');
 w.close();
})().catch(e=>{console.error(e);w.close();process.exitCode=1;});
