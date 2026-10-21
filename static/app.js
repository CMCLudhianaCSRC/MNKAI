/* web-demo copy of demo_app/static/app.js (asset paths only; scripts/build_pages_site.py) */
"use strict";
/* MNKAI-III front end (2026-09 overhaul). Every number shown comes from the API (the running bundle's own
   evaluation files); nothing about model performance is hard-coded here.

   Safe rendering: all markup goes through the html`` tag, which escapes every interpolated value unless it is an
   html`` fragment itself (or an array of them). mount() is the single place fragments enter the DOM. */
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const FRAG=Symbol("frag");
const escText=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function part(v){ if(v==null||v===false) return ""; if(Array.isArray(v)) return v.map(part).join("");
  if(typeof v==="object"&&FRAG in v) return v[FRAG]; return escText(v); }
function html(strings,...vals){ let s=strings[0]; vals.forEach((v,i)=>{ s+=part(v)+strings[i+1]; }); return {[FRAG]:s}; }
function mount(el,frag){ if(!el) return; const r=document.createRange(); r.selectNodeContents(el);   // parse in el's context
  el.replaceChildren(r.createContextualFragment(part(frag))); }

let LAST=null, lastImageDataURL=null, charts={}, EVID=null;
const C_GRID="rgba(60,60,67,.10)", C_TXT="#6e6e73", C_ACC="#0071e3";   // Liquid Glass palette (styles.css)
if(window.Chart){ Chart.defaults.font.family=getComputedStyle(document.body).fontFamily; Chart.defaults.color=C_TXT;
  Chart.defaults.animation.duration=900; Chart.defaults.animation.easing="easeOutQuart"; }
const pct=(v,d=0)=>v==null?"–":(v*100).toFixed(d)+"%";
const niceName=c=>String(c||"").replace(/_/g," ").replace(/\b\w/,m=>m.toUpperCase());
const isImgURL=u=>typeof u==="string"&&/^(data:image\/|\/static\/|static\/)/.test(u);   // only our own image sources
const src=u=>isImgURL(u)?u:"";

/* ---- health, model selector, hero, evidence ---- */
let MODELS=[], CURRENT_MODEL=null;
fetch("/api/health").then(r=>r.json()).then(j=>{
  if(!j.ok) return; const m=j.model;
  mount($("#model-pill"),html`<span class="dot"></span>${m.architecture.split(".")[0]} · ${m.num_classes} classes · ${m.device.toUpperCase()}`);
  $("#model-pill").classList.add("ready");
  MODELS=j.models||[]; CURRENT_MODEL=m.model_key||(MODELS.find(x=>x.default)||{}).key||(MODELS[0]||{}).key;
  heroStats(m); buildModelSelect();
  if(m.version) $("#foot-version").textContent=m.version;
  if(m.num_classes) $("#tech-classes").textContent=`One model, ${m.num_classes} conditions`;
}).catch(()=>mount($("#model-pill"),html`<span class="dot"></span>offline`));
fetch("/api/evidence").then(r=>r.json()).then(j=>{ EVID=j.ok?j.evidence:null; evidenceSection(EVID); }).catch(()=>evidenceSection(null));

function heroStats(m){ const cells=$$("#hero-stats .stat"); if(cells.length<4)return;
  const set=(i,val,dec,suf)=>{ if(val==null)return; const n=cells[i].querySelector(".stat-num");
    n.dataset.count=val; n.dataset.dec=dec; n.dataset.suffix=suf; n.textContent=Number(val).toFixed(dec)+suf;
    if(typeof countUp==="function") countUp(n); };
  set(0,m.n_train!=null?m.n_train/1000:null,0,"K"); set(1,m.num_classes,0,"");
  set(2,m.test_top3!=null?m.test_top3*100:null,0,"%"); set(3,m.atlas_top3!=null?m.atlas_top3*100:null,0,"%"); }

function buildModelSelect(){ const sel=$("#model-select");
  if(!sel||MODELS.length<=1){ $("#model-bar").style.display="none"; return; }
  mount(sel,MODELS.map(m=>html`<option value="${m.key}" ${m.available?"":html`disabled`} ${m.key===CURRENT_MODEL?html`selected`:""}>${m.label}</option>`));
  sel.onchange=()=>{ CURRENT_MODEL=sel.value; if(lastImageDataURL) analyze(lastImageDataURL); }; }

function evBars(rows,key,fmt=v=>pct(v)){ return rows.filter(r=>r[1]&&r[1][key]!=null).map(([lab,o])=>
  html`<div class="ev-bar"><span class="lab">${lab}</span><div class="bar"><i data-w="${(o[key]*100).toFixed(1)}"></i></div><span class="val">${fmt(o[key])}</span></div>`); }
function evidenceSection(ev){ const g=$("#evidence-grid");
  if(!ev){ mount(g,html`<div class="glass hud ev-card"><h4>Evidence file not built</h4><div class="ev-sub">Run scripts/build_app_evidence.py for this bundle.</div></div>`); return; }
  const t=ev.test||{}, cal=ev.calibration||{}, fzb=(ev.outside_fitzpatrick17k||{}).by_fitzpatrick||{}, dd=ev.cancer_check_ddi||{};
  mount(g,html`
   <div class="glass hud ev-card reveal"><h4>Held-out test set</h4><div class="ev-sub">${t.n??"–"} images never used in training</div>
     <div class="ev-big"><div><b>${t.macro_f1!=null?t.macro_f1.toFixed(2):"–"}</b><span>macro-F1</span></div><div><b>${pct(t.top1)}</b><span>top-1</span></div><div><b>${pct(t.top3)}</b><span>top-3</span></div></div>
     <div class="ev-sub" style="margin-top:14px">Calibration error ${cal.ECE_T!=null?cal.ECE_T.toFixed(3):"–"} · the 90% possible-diagnoses list held the right answer ${pct(cal.raps_coverage,1)} of the time (average ${cal.raps_mean_set!=null?cal.raps_mean_set.toFixed(1):"–"} conditions)</div></div>
   <div class="glass hud ev-card reveal"><h4>Outside test by skin tone</h4><div class="ev-sub">Fitzpatrick17k images from sites never trained on · top-1 accuracy</div>
     ${evBars([["FST I-II",fzb["I-II"]],["FST III-IV",fzb["III-IV"]],["FST V-VI",fzb["V-VI"]]],"top1")}
     <div class="ev-sub">Darker skin is harder: this is the gap MNKAI-III is built to close.</div></div>
   <div class="glass hud ev-card reveal"><h4>Cancer check by skin tone</h4><div class="ev-sub">DDI (biopsy-proven, Stanford) · AUROC, cross-validated</div>
     ${evBars([["FST I-II",dd["FST I-II"]],["FST III-IV",dd["FST III-IV"]],["FST V-VI",dd["FST V-VI"]]],"auroc",v=>v.toFixed(2))}
     <div class="ev-sub">AUROC 0.5 = chance, 1.0 = perfect. Two lines: amber catches about 9 in 10 cancers; red is reached by about 1 in 10 harmless lesions.</div></div>
   ${ev.outside_skindisnet?html`<div class="glass hud ev-card ev-alert reveal"><h4>South-Asian clinic photos</h4><div class="ev-sub">SkinDisNet (Bangladesh, ${ev.outside_skindisnet.n} smartphone photos, never trained on) · eczema, contact dermatitis, scabies, tinea, seb. derm</div>
     <div class="ev-big"><div><b>${pct(ev.outside_skindisnet.top1)}</b><span>top-1</span></div><div><b>${pct(ev.outside_skindisnet.top3)}</b><span>top-3</span></div><div><b>${pct(ev.outside_skindisnet.cancer_alarm_rate)}</b><span>cancer alarms (none are cancer)</span></div></div>
     <div class="ev-sub">The model is not yet reliable on this kind of photo, the intended use. Treat its answer as a list to consider, not a diagnosis.</div></div>`:""}
   ${ev.disease_groups?html`<div class="glass hud ev-card reveal"><h4>Disease group</h4><div class="ev-sub">Right kind of disease (infection, inflammation, tumour, ...), even when the exact condition is wrong</div>
     <div class="ev-big">${[["test","held-out"],["atlas","atlas"],["skindisnet","S-Asian clinic"]].filter(([k])=>ev.disease_groups[k]).map(([k,l])=>html`<div><b>${pct(ev.disease_groups[k].accuracy)}</b><span>${l}</span></div>`)}</div>
     ${ev.disease_groups.skindisnet?html`<div class="ev-sub">On the clinic photos, ${pct(ev.disease_groups.skindisnet.infection_called_inflammatory)} of infections were taken for inflammation: the steroid risk.</div>`:""}</div>`:""}
   ${ev.cancer_tiers?html`<div class="glass hud ev-card reveal"><h4>Cancer check: two levels</h4><div class="ev-sub">Harmless outside photos reaching each level (atlas / clinic)</div>
     <div class="ev-big"><div><b>${pct(ev.cancer_tiers.amber_rate_benign_outside.atlas)} / ${pct(ev.cancer_tiers.amber_rate_benign_outside.skindisnet)}</b><span>amber: not ruled out</span></div><div><b>${pct(ev.cancer_tiers.red_rate_benign_outside.atlas)} / ${pct(ev.cancer_tiers.red_rate_benign_outside.skindisnet)}</b><span>red: possible cancer</span></div></div>
     <div class="ev-sub">Red still caught ${pct(ev.cancer_tiers.red_sens_unseen_atlas_cancers)} of atlas cancers the check was never trained on.</div></div>`:""}`);
  window.FX_observe&&FX_observe();
  setTimeout(()=>$$("#evidence-grid .bar i").forEach(i=>i.style.width=i.dataset.w+"%"),400); }

/* ---- upload ---- */
const dz=$("#dropzone");
dz.addEventListener("click",()=>{ FX&&FX.toast("Uploading your own photo is only available on the live version. Select one of the samples instead."); $("#sample-strip").scrollIntoView({behavior:"smooth",block:"center"}); });
$("#file").addEventListener("change",e=>{ if(e.target.files[0]) handleFile(e.target.files[0]); });
["dragover","dragenter"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add("drag");}));
["dragleave","drop"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove("drag");}));
dz.addEventListener("drop",e=>{const f=e.dataTransfer.files[0]; if(f)handleFile(f);});
document.addEventListener("paste",e=>{ const f=[...(e.clipboardData?.files||[])].find(x=>x.type.startsWith("image/")); if(f)handleFile(f); });
$("#cta-analyze").onclick=()=>{ $("#studio").scrollIntoView({behavior:"smooth"}); setTimeout(()=>dz.classList.add("drag"),600); setTimeout(()=>dz.classList.remove("drag"),1400); };
$("#cta-sample").onclick=()=>{ const f=$("#sample-strip figure"); if(f){ $("#studio").scrollIntoView({behavior:"smooth"}); setTimeout(()=>f.click(),500);} };
function handleFile(file){ FX&&FX.toast("Uploading your own photo is only available on the live version. Select one of the samples instead."); return; if(!file.type.startsWith("image/")){ FX&&FX.toast("Please choose an image file"); return; }
  SAMPLE=null; const rd=new FileReader(); rd.onload=()=>{ lastImageDataURL=rd.result; analyze(rd.result); }; rd.readAsDataURL(file); }

/* ---- samples (each carries its recorded diagnosis, shown beside the model's answer) ---- */
let SAMPLES=[], SAMPLE=null;
fetch("static/samples/samples.json").then(r=>r.json()).then(list=>{ SAMPLES=list;
  mount($("#sample-strip"),list.map(s=>html`<figure data-f="${s.file}" title="${s.note||""}"><img src="${src("static/samples/"+s.file)}" alt="${s.class}"/><figcaption>${s.label||niceName(s.class)}</figcaption></figure>`));
  $$("#sample-strip figure").forEach(f=>f.onclick=()=>loadSample(f.dataset.f));
}).catch(()=>{ $(".samples").style.display="none"; });
async function loadSample(file){ SAMPLE=SAMPLES.find(s=>s.file===file)||null; window.WD_context&&WD_context(SAMPLE); const blob=await fetch("static/samples/"+encodeURIComponent(file)).then(r=>r.blob());
  const rd=new FileReader(); rd.onload=()=>{ lastImageDataURL=rd.result; analyze(rd.result); }; rd.readAsDataURL(blob); }

/* ---- patient context: skin-tone swatches + body map ---- */
$$("#fst-picker button").forEach(b=>b.onclick=()=>{ $$("#fst-picker button").forEach(x=>x.classList.remove("on")); b.classList.add("on"); $("#m-fst").value=b.dataset.v; });
let backOfTrunk=false;
function setSite(site){ $$("#body-svg .bm").forEach(g=>g.classList.toggle("on",g.dataset.site===site||(site==="posterior torso"&&g.dataset.site==="anterior torso")));
  $("#m-site").value=site; $("#site-label").textContent=site||"not set"; }
$$("#body-svg .bm").forEach(g=>g.onclick=()=>{ let s=g.dataset.site; if(s==="anterior torso"&&backOfTrunk) s="posterior torso";
  setSite($("#m-site").value===s?"":s); });
$("#btn-back").onclick=()=>{ backOfTrunk=!backOfTrunk; $("#btn-back").classList.toggle("on",backOfTrunk);
  const v=$("#m-site").value; if(v==="anterior torso"||v==="posterior torso") setSite(backOfTrunk?"posterior torso":"anterior torso"); };
function meta(){ const m={}; if($("#m-age").value)m.age=+$("#m-age").value; if($("#m-sex").value)m.sex=$("#m-sex").value;
  if($("#m-site").value)m.site=$("#m-site").value; if($("#m-fst").value)m.fitzpatrick=+$("#m-fst").value; return m; }

/* ---- analyse ---- */
const STEPS=["Quality","Features","Diagnosis","Cancer check","Heatmaps","Brief"];
let stepTimer=null;
async function analyze(dataURL){
  $("#results").classList.add("hidden"); $("#img-orig").src=src(dataURL);
  const st=$("#status"); st.classList.remove("hidden");
  $("#status-title").textContent="Analysing"; $("#status-msg").textContent="PanDerm features → 5-head ensemble → cancer check → explanations";
  mount($("#status-steps"),STEPS.map(s=>html`<span class="st">${s}</span>`));
  let i=0; clearInterval(stepTimer);
  stepTimer=setInterval(()=>{ const els=$$("#status-steps .st"); if(i>0)els[i-1].classList.replace("on","done"); if(i<els.length){els[i].classList.add("on");i++;} },450);
  st.scrollIntoView({behavior:"smooth",block:"center"});
  try{
    const r=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:dataURL,model:CURRENT_MODEL,meta:meta(),options:{mc_dropout:true,mc_samples:15}})});
    const j=await r.json(); if(!j.ok) throw new Error(j.error||"analysis failed");
    clearInterval(stepTimer); $$("#status-steps .st").forEach(e=>{e.classList.remove("on");e.classList.add("done");});
    LAST=j; setTimeout(()=>{ st.classList.add("hidden"); $("#results").classList.remove("hidden"); render(j);
      window.FX_observe&&FX_observe(); $("#results").scrollIntoView({behavior:"smooth"});
      FX&&FX.toast(`${j.prediction.info.name||niceName(j.prediction.class)} · ${pct(j.prediction.prob)}`);
      logSession(j,dataURL); },300);
  }catch(e){ clearInterval(stepTimer); $("#status-title").textContent=/web demo/i.test(e.message)?"Web demo":"Error"; $("#status-msg").textContent=e.message; }
}

/* ---- render ---- */
function render(j){
  $("#img-proc").src=src(j.processed_image);
  $("#infer-chip").textContent=`${j.prediction.inference_ms} ms · ${j.model.device.toUpperCase()}`;
  mount($("#img-meta"),[`${j.image_meta.original_size.join("×")} px`,`input ${j.image_meta.input_size.join("×")}`,j.model.architecture.split(".")[0],`${j.model.params_millions}M params`]
    .map(t=>html`<span>${t}</span>`));
  quality(j.quality); verdict(j); triage(j.decision_support); decisionCard(j.decision_support);
  brief(j); differential(j); askPanel(j); interviewPanel(j); reliability(j);
  softmaxChart(j.softmax); analytics(j.uncertainty); uncChart(j.uncertainty);
  heatTabs(j.heatmaps); film(j.pipeline); featureMaps(j.feature_maps); netflow(j.feature_maps); latent(j.latent);
}

function quality(q){ if(!q){ mount($("#quality"),""); return; }
  mount($("#quality"),html`<div class="q-head"><span class="q-score">${q.score}</span><span class="mono" style="font-size:10.5px;color:var(--dim)">PHOTO QUALITY</span>
    <span class="q-verdict q-${q.verdict}">${q.verdict}</span></div>
    <div class="q-list">${q.checks.map(c=>html`<div class="q-item ${c.status}" title="${c.check}: ${c.value}"><i></i><span><b>${c.check}</b> · ${c.message}</span></div>`)}</div>`); }

function truthLine(j){ if(!SAMPLE) return "";
  const rank=[...(j.softmax||[])].sort((a,b)=>b.prob-a.prob).findIndex(x=>x.class===SAMPLE.class)+1;
  const inSet=((j.decision_support||{}).conformal_set||[]).includes(SAMPLE.class);
  const ord=n=>n+(["th","st","nd","rd"][(n%100>10&&n%100<14)?0:(n%10<4?n%10:0)]);
  const verdictTxt=rank===1?"the model's first choice":rank>0?`ranked ${ord(rank)} by the model${inSet?", inside its 90% list":""}`:"not in the model's list";
  return html`<div class="vtruth ${rank===1?"ok":"miss"}">Demo sample · recorded diagnosis <b>${SAMPLE.label||niceName(SAMPLE.class)}</b> · ${verdictTxt}</div>`; }
function verdict(j){ const p=j.prediction, d=j.decision_support||{}, k=(j.clinical||{})[p.class]||{};
  mount($("#verdict"),html`<div class="ring" id="ring"><b>0%</b></div>
    <div class="vmain"><h2>${p.info.name||niceName(p.class)}</h2>
      <span class="grp ${p.info.risk||""}">${p.info.group||""} · ${p.info.risk||"–"} risk</span>
      <p>${p.info.desc||""}</p>
      <div class="vmeta">${k.contagious?html`<span>contagious: ${k.contagious}</span>`:""}${d.conformal_set?html`<span>${d.conformal_set.length} in 90% list</span>`:""}</div>
      ${truthLine(j)}${groupBlock(d)}</div>`);
  animateRing($("#ring"),Math.round(p.prob*100));
  $("#gate-chip").textContent=d.confident_single_answer===true?"CONFIDENT ANSWER":d.confident_single_answer===false?"POSSIBLE DIAGNOSES":""; }

/* disease group (7 taxonomy groups; the app sums the condition probabilities in each group). It matters even when
   the exact condition is uncertain: an infection taken for an inflammatory rash may be given a steroid. */
const GROUPS={neoplastic_malignant:"Skin cancer & precancer",neoplastic_benign:"Benign growths",infectious:"Infections & infestations",
  inflammatory:"Inflammatory conditions",autoimmune_bullous:"Autoimmune & blistering",pigmentary_adnexal:"Pigment & hair",normal:"Normal skin"};
function groupBlock(d){ const gp=Object.entries(d.superclass_probs||{}); if(!gp.length) return "";
  return html`<div class="vgroup"><div class="vg-head"><span class="vg-label">Disease group</span><b>${GROUPS[gp[0][0]]||niceName(gp[0][0])}</b><span class="vg-p">${pct(gp[0][1])}</span></div>
    ${d.confident_single_answer===false?html`<div class="vg-note">Not sure of the exact condition; the group is still useful (infection vs inflammation changes treatment).</div>`:""}
    <div class="vg-bars">${gp.slice(0,4).map(([g,v])=>html`<div class="vg-row"><span>${GROUPS[g]||niceName(g)}</span><div class="vg-meter"><i style="width:${(v*100).toFixed(1)}%"></i></div><em>${pct(v)}</em></div>`)}</div></div>`; }

function triage(d){ if(!d){ mount($("#triage"),""); return; }
  const lvl={decision_support_ok:0,refer_to_dermatologist:1,urgent_referral:2}[d.level]??1;
  const note=["Routine: manage with the clinical brief; review if it does not improve.",
    "Review by a dermatologist or senior clinician: the model is not sure enough for a single answer, or the condition needs specialist care.",
    "Urgent in-person review: the cancer check's red line or a serious condition was flagged."][lvl];
  mount($("#triage"),html`<div class="tg-bar"><div class="tg-seg g ${lvl===0?"on":""}">Routine</div><div class="tg-seg a ${lvl===1?"on":""}">Review</div><div class="tg-seg r ${lvl===2?"on":""}">Urgent</div></div>
    <div class="tg-note">${note}</div>`); }

const LEVEL_META={urgent_referral:{cls:"urgent",tag:"Urgent referral"},refer_to_dermatologist:{cls:"refer",tag:"Refer"},decision_support_ok:{cls:"ok",tag:"Decision support"}};
function decisionCard(d){ if(!d){ mount($("#decision"),""); return; }
  const lm=LEVEL_META[d.level]||LEVEL_META.refer_to_dermatologist, rows=[];
  if(d.cancer_check_score!=null){ const s=d.cancer_check_score, t=d.cancer_check_threshold, r=d.cancer_check_threshold_red;
    const tierTxt={red:"RED · possible cancer: urgent in-person review",amber:"AMBER · cancer not ruled out: dermatologist review advised",
      none:"Below the rule-out line: the photo check is reassuring (it does not replace examination)"}[d.cancer_tier||"none"];
    rows.push(html`<div class="ds-row">Cancer check score <b>${pct(s,1)}</b> · <b>${tierTxt}</b>
      <div class="meter"><i style="width:${Math.min(100,s*100).toFixed(1)}%"></i><span class="thr" style="left:${Math.min(100,t*100).toFixed(1)}%"></span>${r!=null?html`<span class="thr" style="left:${Math.min(100,r*100).toFixed(1)}%"></span>`:""}</div>
      <span style="font-size:10.5px;color:var(--dim)">Two lines: amber at ${pct(t,1)} (catches about 9 in 10 cancers, so it also marks many harmless lesions) and red at ${r!=null?pct(r,1):"–"} (about 1 in 10 harmless lesions reach it). The score is not a probability.</span></div>`); }
  if(d.malignant_probability!=null) rows.push(html`<div class="ds-row">Probability of the cancer classes (sum): <b>${pct(d.malignant_probability)}</b></div>`);
  if(d.confident_single_answer!=null&&d.display_gate) rows.push(d.confident_single_answer
    ? html`<div class="ds-row">✓ Confident answer: in testing, answers this sure were right about <b>${pct(d.display_gate.oof_precision)}</b> of the time.</div>`
    : html`<div class="ds-row">? Not sure enough for a single answer: use the list of possible diagnoses.</div>`);
  if(d.conformal_set&&d.conformal_set.length) rows.push(html`<div class="ds-row ds-set">90% possible-diagnoses list: <b>${d.conformal_set.map(niceName).join(" · ")}</b></div>`);
  if(d.superclass_probs) rows.push(html`<div class="ds-row">Disease group: ${Object.entries(d.superclass_probs).slice(0,3).map(([k,v])=>`${niceName(k)} ${pct(v)}`).join(" · ")}</div>`);
  if(d.fairness_caution) rows.push(html`<div class="ds-row ds-fair">⚖ ${d.fairness_caution}</div>`);
  mount($("#decision"),html`<div class="ds-card ds-${lm.cls}"><div class="ds-head"><b>${d.headline}</b><span class="ds-tag">${lm.tag}</span></div>${rows}<div class="ds-disc">${d.disclaimer||""}</div></div>`); }

/* clinical brief: tabs for the top 3 */
function brief(j){ const cl=j.clinical||{}, top=(j.topk||[]).slice(0,3).filter(t=>cl[t.class]);
  if(!top.length){ mount($("#brief-tabs"),""); mount($("#brief"),html`<p class="hint">No brief available for these conditions.</p>`); return; }
  mount($("#brief-tabs"),top.map((t,i)=>html`<div class="tab ${i?"":"active"}" data-c="${t.class}">${t.info.name||niceName(t.class)} <span class="mono" style="opacity:.7">${pct(t.prob)}</span></div>`));
  $$("#brief-tabs .tab").forEach(t=>t.onclick=()=>{ $$("#brief-tabs .tab").forEach(x=>x.classList.remove("active")); t.classList.add("active"); showBrief(cl[t.dataset.c]); });
  showBrief(cl[top[0].class]); }
function showBrief(k){ const li=a=>(a||[]).map(x=>html`<li>${x}</li>`);
  mount($("#brief"),html`<div class="brief-tags"><span class="${/^(yes|often)/i.test(k.contagious||"")?"contag":""}">Contagious: ${k.contagious||"–"}</span></div>
    <div class="brief-grid">
      <div class="bcell skin"><h4>ON BROWN &amp; DARK SKIN</h4><p>${k.dark_skin}</p></div>
      <div class="bcell"><h4>INDIA</h4><p>${k.india}</p></div>
      <div class="bcell"><h4>CONFIRM AT THE BEDSIDE</h4><ul>${li(k.confirm)}</ul></div>
      <div class="bcell redflag"><h4>REFER IF</h4><ul>${li(k.refer_if)}</ul></div>
      <div class="bcell" style="grid-column:1/-1"><h4>FIRST STEPS (GENERAL)</h4><ul>${li(k.first_steps)}</ul></div>
    </div>`); }

function differential(j){ const set=new Set((j.decision_support||{}).conformal_set||[]);
  mount($("#topk"),html`${j.topk.map(t=>html`<div class="row ${set.has(t.class)?"in-set":""}"><span class="cname">${t.info.name||niceName(t.class)}</span>
    <div class="bar"><i data-w="${(t.prob*100).toFixed(1)}"></i></div><span class="pct mono">${(t.prob*100).toFixed(1)}%</span></div>`)}
    <p class="hint">● = inside the 90% possible-diagnoses list</p>`);
  setTimeout(()=>$$("#topk .bar i").forEach(i=>i.style.width=i.dataset.w+"%"),120);
  const top=j.prediction.class, k=(j.clinical||{})[top], others=new Set(j.topk.slice(1).map(t=>t.class));
  const cues=[...((k&&k.tell_apart)||[])].sort((a,b)=>others.has(b.vs)-others.has(a.vs));
  mount($("#tell-apart"),cues.length?html`<h4>HOW TO TELL ${(j.prediction.info.name||niceName(top)).toUpperCase()} APART</h4>
    ${cues.map(c=>html`<div class="ta-row">vs <b>${c.vs_name}</b>${others.has(c.vs)?html` <span class="mono" style="color:var(--cyan);font-size:10px">· IN TOP 5</span>`:""}: ${c.cue}</div>`)}`:""); }

function askPanel(j){ const cl=j.clinical||{}, seen=new Set(), qs=[];
  (j.topk||[]).slice(0,3).forEach(t=>((cl[t.class]||{}).ask||[]).forEach(q=>{ if(!seen.has(q)){ seen.add(q); qs.push({q,for:t.info.name||niceName(t.class)}); } }));
  mount($("#askbox"),qs.length?qs.map(x=>html`<label class="aq"><input type="checkbox"/><span>${x.q}<small>helps with: ${x.for}</small></span></label>`)
    :html`<p class="hint">No questions for these conditions.</p>`);
  $$("#askbox .aq input").forEach(c=>c.onchange=()=>c.closest(".aq").classList.toggle("done",c.checked)); }

function reliability(j){ const c=j.prediction.class, r=(j.reliability||{})[c], name=j.prediction.info.name||niceName(c);
  if(!r){ mount($("#reliability"),html`<p class="hint">Track record not built for this model (scripts/build_app_evidence.py).</p>`); return; }
  mount($("#reliability"),html`<div class="rel-head">For <b>${name}</b>, from ${r.n_train.toLocaleString()} training images checked out-of-fold:</div>
    <div class="rel-grid">
      <div class="rel"><div class="n">${pct(r.precision)}</div><div class="l">precision</div><div class="s">when it says ${name}, it is right this often</div></div>
      <div class="rel"><div class="n">${pct(r.sensitivity)}</div><div class="l">sensitivity</div><div class="s">of real ${name} cases, it finds this many</div></div>
      <div class="rel"><div class="n">${pct(r.confident_precision)}</div><div class="l">when confident</div><div class="s">precision of its confident single answers</div></div>
    </div>
    <div class="rel-note">Cross-validated on the training pool, not the sealed test. Rare classes and near-duplicate-heavy sources can make these look better than real-world use.</div>`); }

/* ---- LLM interview (optional, needs a Gemini key) ---- */
let _lastDiff=[];
function interviewPanel(j){ const iv=j.interview||{};
  _lastDiff=(j.topk||[]).map(t=>({name:t.class,prob:t.prob,label:(t.info&&t.info.name)||t.class}));
  if(!iv.available||!iv.triggered){ mount($("#interview"),""); return; }
  if(iv.on_request&&!(iv.questions||[]).length){        // never automatic: the photo would leave this laptop
    mount($("#interview"),html`<div class="iv-card iv-ask"><div class="iv-head"><b>AI follow-up questions</b><span class="iv-tag">optional · online</span></div>
      <p>The model is unsure. Gemini can look at the photo and suggest history questions. This sends the photo to Google over the internet and takes 10–20 seconds; do not use it for study patients.</p>
      <button id="iv-ask" class="btn-ghost small iv-btn">Ask Gemini for questions</button><div id="iv-out" class="iv-out"></div></div>`);
    $("#iv-ask").onclick=async()=>{ const b=$("#iv-ask"); b.disabled=true; mount($("#iv-out"),html`<div class="iv-load">Asking Gemini…</div>`);
      try{ const r=await fetch("/api/interview_questions",{method:"POST",headers:{"Content-Type":"application/json"},
          body:JSON.stringify({image:lastImageDataURL,meta:meta(),differential:_lastDiff.map(d=>({name:d.name,prob:d.prob}))})}).then(x=>x.json());
        if(!r.ok){ mount($("#iv-out"),html`<div class="iv-warn">${r.error||"Gemini did not answer."}</div>`); b.disabled=false; return; }
        interviewPanel({...j,interview:{...iv,on_request:false,questions:r.questions,note:r.note}});
      }catch(e){ mount($("#iv-out"),html`<div class="iv-warn">${e.message}</div>`); b.disabled=false; } };
    return; }
  mount($("#interview"),html`<div class="iv-card iv-ask"><div class="iv-head"><b>AI follow-up questions about this photo</b><span class="iv-tag">margin ${iv.margin}</span></div>
    <p>${iv.note||""}</p>
    <div class="iv-qs">${(iv.questions||[]).map(q=>html`<label class="iv-q"><span class="iv-qt">${q.q}</span><em>${q.why||""}</em>
      <input class="iv-a" data-q="${q.q}" placeholder="Patient's answer…"/></label>`)}</div>
    <button id="iv-go" class="btn-primary small iv-btn">Refine with history →</button><div id="iv-out" class="iv-out"></div></div>`);
  $("#iv-go").onclick=fuseInterview; }
async function fuseInterview(){ const out=$("#iv-out"), btn=$("#iv-go");
  const qa=$$(".iv-a").map(i=>({q:i.dataset.q,a:i.value.trim()})).filter(x=>x.a);
  if(!qa.length){ mount(out,html`<div class="iv-warn">Answer at least one question first.</div>`); return; }
  btn.disabled=true; mount(out,html`<div class="iv-load">Weighing the history against the image…</div>`);
  try{ const r=await fetch("/api/interview_fuse",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({differential:_lastDiff.map(d=>({name:d.name,prob:d.prob})),qa})}).then(x=>x.json());
    if(!r.ok){ mount(out,html`<div class="iv-warn">${r.error||"Fusion failed."}</div>`); btn.disabled=false; return; }
    const nm=n=>{const f=_lastDiff.find(d=>d.name===n); return f?f.label:n;}, before=Object.fromEntries(r.before.map(b=>[b.class,b.prob]));
    mount(out,html`<div class="iv-res"><div class="iv-res-h">Refined differential (image + history)</div>
      ${r.posterior.slice(0,5).map(p=>{ const d=p.prob-(before[p.class]||0);
        return html`<div class="iv-row"><span>${nm(p.class)}</span><div class="bar"><i style="width:${(p.prob*100).toFixed(1)}%"></i></div>
          <span class="mono">${pct(p.prob)}</span><span class="iv-d ${d>=0?"up":"down"}">${d>=0?"▲":"▼"}${Math.abs(d*100).toFixed(0)}</span></div>`; })}
      ${r.red_flags&&r.red_flags.length?html`<div class="iv-flags">⚑ ${r.red_flags.join(" · ")}</div>`:""}
      ${r.explanation?html`<div class="iv-exp">${r.explanation}</div>`:""}<div class="iv-disc">${r.disclaimer||""}</div></div>`);
  }catch(e){ mount(out,html`<div class="iv-warn">${e.message}</div>`); }
  btn.disabled=false; }

function animateRing(el,target){ const b=el.querySelector("b"), t0=performance.now(), dur=1100;
  (function s(t){ const k=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-k,3), v=Math.round(target*e);
    el.style.setProperty("--p",v); b.textContent=v+"%"; if(k<1)requestAnimationFrame(s); })(t0); }

/* ---- charts ---- */
function softmaxChart(sm){ if(!window.Chart)return; const ctx=$("#softmaxChart"); charts.sm&&charts.sm.destroy();
  const s=[...sm].sort((a,b)=>b.prob-a.prob).slice(0,12);
  const g=ctx.getContext("2d").createLinearGradient(0,0,0,210); g.addColorStop(0,"#2b8cff"); g.addColorStop(1,C_ACC);
  charts.sm=new Chart(ctx,{type:"bar",data:{labels:s.map(x=>niceName(x.class)),datasets:[{data:s.map(x=>+(x.prob*100).toFixed(2)),
    backgroundColor:s.map((x,i)=>i===0?g:"rgba(0,113,227,.16)"),borderRadius:8,borderSkipped:false,maxBarThickness:40}]},
    options:{plugins:{legend:{display:false},tooltip:{backgroundColor:"rgba(255,255,255,.96)",titleColor:"#1d1d1f",bodyColor:"#1d1d1f",borderColor:"rgba(60,60,67,.15)",borderWidth:1,callbacks:{label:c=>` ${c.parsed.y}%`}}},
      scales:{y:{ticks:{callback:v=>v+"%"},grid:{color:C_GRID}},x:{ticks:{color:"#515154",font:{size:10}},grid:{display:false}}}}}); }
function uncChart(u){ if(!window.Chart)return; const ctx=$("#uncChart"); charts.unc&&charts.unc.destroy();
  charts.unc=new Chart(ctx,{type:"radar",data:{labels:["Confidence","Certainty","Margin","Purity","Stability"],
    datasets:[{data:[u.confidence,1-u.normalized_entropy,u.margin_top1_top2,1-u.gini_impurity,1-(u.mc_dropout_mean_std||0)*5].map(v=>+Math.max(0,Math.min(1,v)).toFixed(3)),
      backgroundColor:"rgba(0,113,227,.12)",borderColor:C_ACC,borderWidth:2,pointBackgroundColor:C_ACC,pointRadius:3}]},
    options:{plugins:{legend:{display:false}},scales:{r:{min:0,max:1,angleLines:{color:C_GRID},grid:{color:C_GRID},pointLabels:{color:C_TXT,font:{size:11}},ticks:{display:false}}}}}); }
const WD_TIPS={"Confidence": "The probability given to the first choice, after calibration. Close to 100% means almost all the weight is on one condition.", "Entropy": "How spread out the probabilities are over the 29 conditions (in nats). 0 means all on one condition; higher means more unsure.", "Norm. entropy": "The same spread on a 0 to 1 scale: 0 means all on one condition, 1 means spread evenly over all 29.", "Margin": "The gap between the first and second choices. Near 1 means a clear winner; near 0 means two conditions are almost tied.", "Eff. classes": "Roughly how many conditions the model is really choosing between: 1 means one clear answer, 3 means torn between about three.", "Head spread": "How much the answer varies across the model's repeated runs of its five classifiers. Low means stable; high means they disagree, and the system may then decline to give a single answer.", "Band": "A one-word summary of the spread (normalised entropy): high below 0.35, moderate from 0.35 to 0.6, low above 0.6."};
const wdTip=l=>WD_TIPS[l]?html`<span class="wd-i" tabindex="0" role="button" aria-label="What this means: ${WD_TIPS[l]}" data-tip="${WD_TIPS[l]}">i</span>`:"";
function analytics(u){ const cell=(l,v,sm="")=>html`<div class="metric"><div class="l">${l}${wdTip(l)}</div><div class="v">${v}<small>${sm}</small></div></div>`;
  mount($("#analytics"),[cell("Confidence",(u.confidence*100).toFixed(1)+"%"),cell("Entropy",u.entropy.toFixed(2)," nats"),
    cell("Norm. entropy",u.normalized_entropy.toFixed(2)),cell("Margin",u.margin_top1_top2.toFixed(2)),cell("Eff. classes",u.effective_classes),
    u.epistemic_uncertainty!=null?cell("Head spread",u.epistemic_uncertainty.toFixed(2)):"",
    html`<div class="metric"><div class="l">Band${wdTip("Band")}</div><div class="v band-${u.confidence_band}">${u.confidence_band}</div></div>`]); }

/* ---- heatmaps ---- */
const HEAT=[["grad_cam","Grad-CAM","Class-discriminative evidence from the final block."],
  ["grad_cam_pp","Grad-CAM++","Sharper localisation via higher-order gradients."],["attention_rollout","Attention","How attention flows from the class token to image patches."],
  ["saliency","Saliency","Raw input-gradient sensitivity."],["score_cam","Score-CAM","Gradient-free mask-perturbation importance."]];
const SC_LABEL=[...$("#btn-scorecam").childNodes].map(n=>n.cloneNode(true));          // "+ Run Score-CAM (slow)"
const resetScoreCam=()=>$("#btn-scorecam").replaceChildren(...SC_LABEL.map(n=>n.cloneNode(true)));
function heatTabs(h){ resetScoreCam(); mount($("#heat-tabs"),HEAT.map(([k,n],idx)=>html`<div class="tab ${idx===0?"active":""} ${k==="score_cam"||h[k]?"":"disabled"}" data-k="${k}">${n}</div>`));
  $$("#heat-tabs .tab").forEach(t=>t.onclick=()=>{ $$("#heat-tabs .tab").forEach(x=>x.classList.remove("active")); t.classList.add("active");
    showHeat(t.dataset.k,(HEAT.find(x=>x[0]===t.dataset.k)||[])[2]); });
  showHeat("grad_cam",HEAT[0][2]); }
function showHeat(k,cap){ const img=$("#heat-img");
  if(k==="score_cam"&&!LAST.heatmaps.score_cam){ $("#heat-caption").textContent="Click ＋ Run Score-CAM."; img.src=src(LAST.processed_image); return; }
  img.style.opacity=0; setTimeout(()=>{ img.src=src(LAST.heatmaps[k]||LAST.processed_image); img.style.opacity=1; },150); $("#heat-caption").textContent=cap; }
$("#btn-scorecam").onclick=async()=>{ const b=$("#btn-scorecam"); b.textContent="Running Score-CAM…";
  try{ const j=await fetch("/api/scorecam",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image:lastImageDataURL,model:CURRENT_MODEL,meta:meta()})}).then(r=>r.json());
    if(j.ok){ LAST.heatmaps.score_cam=j.score_cam; $$("#heat-tabs .tab").forEach(x=>{ if(x.dataset.k==="score_cam"){ x.classList.remove("disabled"); x.click(); } }); }
    if(j.ok) b.textContent="✓ Score-CAM ready"; else resetScoreCam(); }
  catch(e){ resetScoreCam(); } };

/* ---- film ---- */
let filmI=0, filmTimer=null;
function film(stages){ if(!stages||!stages.length)return; const strip=$("#film-strip"); strip.replaceChildren();
  stages.forEach((s,i)=>{ const im=new Image(); im.src=src(s.img); im.onclick=()=>setFilm(i); strip.appendChild(im); });
  $("#film-slider").max=stages.length-1; $("#film-slider").oninput=e=>setFilm(+e.target.value); $("#film-play").onclick=togglePlay; setFilm(0); }
function setFilm(i){ const s=LAST.pipeline[i]; filmI=i; const img=$("#film-img");
  img.style.opacity=0; setTimeout(()=>{ img.src=src(s.img); img.style.opacity=1; },120);
  $("#film-title").textContent=s.title; $("#film-caption").textContent=s.caption; $("#film-slider").value=i;
  $("#film-count").textContent=`${i+1}/${LAST.pipeline.length}`; $$("#film-strip img").forEach((x,k)=>x.classList.toggle("active",k===i)); }
function togglePlay(){ if(filmTimer){ clearInterval(filmTimer); filmTimer=null; $("#film-play").textContent="▶ Play"; return; }
  $("#film-play").textContent="⏸ Pause";
  filmTimer=setInterval(()=>{ const n=(filmI+1)%LAST.pipeline.length; setFilm(n); if(n===LAST.pipeline.length-1){ clearInterval(filmTimer); filmTimer=null; $("#film-play").textContent="▶ Play"; } },1100); }
addEventListener("keydown",e=>{ if(!LAST||$("#results").classList.contains("hidden")||e.target.matches("input,select"))return;
  if(e.key==="ArrowRight")setFilm((filmI+1)%LAST.pipeline.length); if(e.key==="ArrowLeft")setFilm((filmI-1+LAST.pipeline.length)%LAST.pipeline.length); });

/* ---- feature maps, flow, latent ---- */
function featureMaps(f){ mount($("#fmaps"),(f||[]).map(m=>html`<div class="fmap"><img src="${src(m.img)}" alt=""/><span class="tag">block ${m.layer} · ${Math.round(m.depth_frac*100)}%</span></div>`)); }
function netflow(f){ const layers=[{n:"Input"},...(f||[]).map(m=>({n:"Blk "+m.layer,a:m.depth_frac})),{n:"Heads"}];
  const W=560,H=180,n=layers.length,gap=W/(n-1);
  const lines=layers.slice(0,-1).map((_,i)=>html`<line x1="${i*gap}" y1="${H/2}" x2="${(i+1)*gap}" y2="${H/2}" stroke="url(#nf)" stroke-width="2" opacity=".45" stroke-dasharray="4 4"><animate attributeName="stroke-dashoffset" values="8;0" dur="1s" repeatCount="indefinite"/></line>`);
  const nodes=layers.map((L,i)=>{ const a=L.a||(i===0?.2:.95), r=10+a*14;
    return html`<circle cx="${i*gap}" cy="${H/2}" r="${r}" fill="rgba(0,113,227,${(.10+a*.4).toFixed(2)})" stroke="#0071e3" stroke-width="1.5"><animate attributeName="r" values="${r};${r+2};${r}" dur="${2+i*.2}s" repeatCount="indefinite"/></circle>
      <text x="${i*gap}" y="${H/2+r+15}" fill="#6e6e73" font-size="11" font-family="-apple-system,Inter,Segoe UI,sans-serif" text-anchor="middle">${L.n}</text>`; });
  mount($("#netflow"),html`<svg viewBox="-20 0 ${W+40} ${H}"><defs><linearGradient id="nf" x1="0" x2="1"><stop offset="0" stop-color="#0071e3"/><stop offset="1" stop-color="#5e5ce6"/></linearGradient></defs>${lines}${nodes}</svg>`); }
function latent(L){ const near=L.nearest_class_centroid;
  if(near){ const mx=Math.max(...near.map(x=>x.similarity));
    mount($("#latent-near"),html`<h4 class="mono" style="margin:0 0 12px;font-size:10.5px;letter-spacing:.12em;color:var(--cyan)">COSINE SIMILARITY TO CLASS CENTROIDS</h4>
      ${near.slice(0,6).map(s=>html`<div class="row"><span class="cname">${niceName(s.class)}</span><div class="bar"><i data-w="${Math.max(0,s.similarity/mx*100).toFixed(1)}"></i></div><span class="pct mono">${s.similarity.toFixed(2)}</span></div>`)}`);
    setTimeout(()=>$$("#latent-near .bar i").forEach(i=>i.style.width=i.dataset.w+"%"),150);
  } else mount($("#latent-near"),html`<p class="hint">Latent centroids not built. Embedding dim ${L.embedding_dim}.</p>`);
  embeddingScatter(L); }
async function embeddingScatter(L){ if(!window.Chart)return; const r=await fetch("/api/embedding_space").then(x=>x.json()).catch(()=>({ok:false}));
  const ctx=$("#latentChart"); charts.lat&&charts.lat.destroy();
  if(!r.ok){ $("#latent-note").textContent="2-D latent map appears once prototypes are built."; return; }
  const by={}; r.coords.forEach((c,i)=>{ (by[r.labels[i]]=by[r.labels[i]]||[]).push({x:c[0],y:c[1]}); });
  const pal=["#0071e3","#5e5ce6","#34c759","#ff9f0a","#ff3b30","#5ac8fa","#af52de","#30b0c7","#ff2d55","#a2845e"];   // Apple system colours
  const ds=Object.keys(by).map((k,i)=>({label:niceName(k),data:by[k],backgroundColor:pal[i%pal.length],pointRadius:2}));
  if(L.uploaded_xy) ds.push({label:"★ this photo",data:[{x:L.uploaded_xy[0],y:L.uploaded_xy[1]}],backgroundColor:"#1d1d1f",pointRadius:9,pointStyle:"star"});
  charts.lat=new Chart(ctx,{type:"scatter",data:{datasets:ds},options:{plugins:{legend:{labels:{color:C_TXT,boxWidth:7,font:{size:9.5}}}},
    scales:{x:{ticks:{display:false},grid:{color:C_GRID}},y:{ticks:{display:false},grid:{color:C_GRID}}}}});
  $("#latent-note").textContent=`${r.method.toUpperCase()} projection of training embeddings · ★ = this photo.`; }

/* ---- session log (this browser only) ---- */
const LOG_KEY="mnkai_session_log";
function readLog(){ try{ const l=JSON.parse(localStorage.getItem(LOG_KEY)||"[]"); return Array.isArray(l)?l:[]; }catch(e){ return []; } }
function writeLog(l){ try{ localStorage.setItem(LOG_KEY,JSON.stringify(l.slice(0,20))); }catch(e){} }
function thumb(dataURL){ return new Promise(res=>{ const im=new Image(); im.onload=()=>{ const c=document.createElement("canvas"), s=96/Math.max(im.width,im.height);
  c.width=Math.round(im.width*s); c.height=Math.round(im.height*s); c.getContext("2d").drawImage(im,0,0,c.width,c.height); res(c.toDataURL("image/jpeg",.7)); };
  im.onerror=()=>res(""); im.src=src(dataURL); }); }
async function logSession(j,dataURL){ const l=readLog();
  l.unshift({t:new Date().toLocaleTimeString(),name:j.prediction.info.name||niceName(j.prediction.class),p:j.prediction.prob,
    level:(j.decision_support||{}).level||"",img:await thumb(dataURL)}); writeLog(l); renderLog(); }
function renderLog(){ const l=readLog();
  mount($("#history-list"),l.length?l.map(x=>html`<div class="h-item" title="${x.level}"><img src="${src(x.img)}" alt=""/><div><b>${x.name}</b><span>${pct(x.p)} · ${x.t} · ${(LEVEL_META[x.level]||{}).tag||""}</span></div></div>`)
    :html`<p class="h-empty">No scans yet in this browser.</p>`); }
$("#btn-history").onclick=()=>{ renderLog(); $("#history").classList.toggle("open"); };
$("#history-close").onclick=()=>$("#history").classList.remove("open");
$("#history-clear").onclick=()=>{ writeLog([]); renderLog(); };
addEventListener("keydown",e=>{ if(!e.target.matches("input,select,textarea")&&e.key.toLowerCase()==="h") $("#btn-history").click(); });

/* ---- report ---- */
$("#btn-report").onclick=()=>{ if(!LAST)return; const j=LAST, d=j.decision_support||{}, p=j.prediction, k=(j.clinical||{})[p.class]||{}, m=meta();
  const li=a=>(a||[]).map(x=>html`<li>${x}</li>`);
  mount($("#print-report"),html`<h1>MNKAI-III scan report</h1><div>${new Date().toLocaleString()} · ${j.model.version||""}</div>
    <div>Patient context: age ${m.age??"–"} · sex ${m.sex||"–"} · site ${m.site||"–"} · Fitzpatrick ${m.fitzpatrick??"–"}</div>
    ${d.level==="urgent_referral"?html`<div class="pr-flag">URGENT: ${d.headline}</div>`:html`<div><b>Triage:</b> ${d.headline||""}</div>`}
    <h2>Images</h2><div class="pr-imgs"><img src="${src(lastImageDataURL)}" alt=""/><img src="${src(j.heatmaps.grad_cam||j.processed_image)}" alt=""/></div>
    <h2>Result</h2><table><tr><th>Condition</th><th>Probability</th><th>In 90% list</th></tr>
      ${j.topk.map(t=>html`<tr><td>${t.info.name||niceName(t.class)}</td><td>${pct(t.prob,1)}</td><td>${(d.conformal_set||[]).includes(t.class)?"yes":""}</td></tr>`)}</table>
    ${d.cancer_check_score!=null?html`<p>Cancer check score ${pct(d.cancer_check_score,1)}: ${(d.cancer_tier||"none").toUpperCase()} (amber at ${pct(d.cancer_check_threshold,1)}, red above ${d.cancer_check_threshold_red!=null?pct(d.cancer_check_threshold_red,1):"–"}).</p>`:""}
    <p>Photo quality: ${j.quality?`${j.quality.score}/100 (${j.quality.verdict})`:"–"}</p>
    <div class="pr-foot">Research and education demo. Decision support for clinicians, not a diagnosis. Numbers come from the model's own evaluation files.</div>`);
  window.print(); };

/* ---- controls ---- */
$("#btn-present").onclick=()=>{ document.body.classList.toggle("present");
  if(document.body.classList.contains("present")&&!document.fullscreenElement) document.documentElement.requestFullscreen?.();
  FX&&FX.toast(document.body.classList.contains("present")?"Presentation mode · Esc to exit":"Exited presentation"); };
$("#btn-new").onclick=()=>{ $("#results").classList.add("hidden"); $("#studio").scrollIntoView({behavior:"smooth"}); $("#file").value=""; };
