"use strict";
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
let LAST=null, lastImageDataURL=null, charts={};
const C_GRID="rgba(120,160,230,.12)", C_TXT="#9fb2d6";
if(window.Chart){ Chart.defaults.font.family="Inter"; Chart.defaults.color=C_TXT;
  Chart.defaults.animation.duration=900; Chart.defaults.animation.easing="easeOutQuart"; }

/* ---- model health + selector ---- */
let MODELS=[], CURRENT_MODEL=null;
fetch("/api/health").then(r=>r.json()).then(j=>{
  if(j.ok){const m=j.model;
    $("#model-pill").innerHTML=`<span class="dot"></span>${m.architecture.split(".")[0]} · ${m.params_millions}M · ${m.device.toUpperCase()}`;
    $("#model-pill").classList.add("ready");
    MODELS=j.models||[]; CURRENT_MODEL=(j.model.model_key)||(MODELS.find(x=>x.default)||{}).key||(MODELS[0]||{}).key;
    buildModelSelect(); }
}).catch(()=>{ $("#model-pill").innerHTML=`<span class="dot"></span>offline`; });

function buildModelSelect(){ const sel=$("#model-select"); if(!sel||!MODELS.length){$("#model-bar")&&($("#model-bar").style.display="none");return;}
  sel.innerHTML=MODELS.map(m=>`<option value="${m.key}" ${m.available?"":"disabled"} ${m.key===CURRENT_MODEL?"selected":""}>${m.label}${m.available?"":" (unavailable)"}</option>`).join("");
  setModelVersion();
  sel.onchange=()=>{ CURRENT_MODEL=sel.value; setModelVersion();
    FX&&FX.toast(`Switched to ${(MODELS.find(m=>m.key===CURRENT_MODEL)||{}).label||CURRENT_MODEL}`);
    if(lastImageDataURL) analyze(lastImageDataURL); };   // re-run on the same image
}
function setModelVersion(){ const m=MODELS.find(x=>x.key===CURRENT_MODEL); $("#model-version").textContent=m?m.version:""; }

/* ---- upload ---- */
const dz=$("#dropzone");
dz.addEventListener("click",()=>$("#file").click());
$("#file").addEventListener("change",e=>{ if(e.target.files[0]) handleFile(e.target.files[0]); });
["dragover","dragenter"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add("drag");}));
["dragleave","drop"].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove("drag");}));
dz.addEventListener("drop",e=>{const f=e.dataTransfer.files[0];if(f)handleFile(f);});
$("#cta-analyze").onclick=()=>{ $("#studio").scrollIntoView({behavior:"smooth"}); setTimeout(()=>dz.classList.add("drag"),600); setTimeout(()=>dz.classList.remove("drag"),1400); };
$("#cta-sample").onclick=()=>{ const f=$("#sample-strip figure"); if(f){ $("#studio").scrollIntoView({behavior:"smooth"}); setTimeout(()=>f.click(),500);} };

function handleFile(file){ const rd=new FileReader();
  rd.onload=()=>{ lastImageDataURL=rd.result; $("#img-orig").src=rd.result; analyze(rd.result); }; rd.readAsDataURL(file); }

/* ---- samples ---- */
const SN={mel:"Melanoma",nv:"Nevus",bcc:"BCC",scc:"SCC",ak:"Actinic K.",bkl:"Benign K.",df:"Dermatofib.",vasc:"Vascular",normal:"Normal"};
fetch("/static/samples/samples.json").then(r=>r.json()).then(list=>{
  $("#sample-strip").innerHTML=list.map(s=>`<figure data-f="${s.file}"><img src="/static/samples/${s.file}"/><figcaption>${SN[s.class]||s.class}</figcaption></figure>`).join("");
  $$("#sample-strip figure").forEach(f=>f.onclick=()=>loadSample(f.dataset.f));
}).catch(()=>{$(".samples").style.display="none";});
async function loadSample(file){ const blob=await fetch("/static/samples/"+file).then(r=>r.blob());
  const rd=new FileReader(); rd.onload=()=>{lastImageDataURL=rd.result;$("#img-orig").src=rd.result;analyze(rd.result);}; rd.readAsDataURL(blob); }

function meta(){ const m={};
  if($("#m-age").value)m.age=+$("#m-age").value; if($("#m-sex").value)m.sex=$("#m-sex").value;
  if($("#m-site").value)m.site=$("#m-site").value; if($("#m-fst").value)m.fitzpatrick=+$("#m-fst").value; return m; }

/* ---- analyze ---- */
const STEPS=["Preprocess","Inference","Grad-CAM","Attention","Uncertainty","Embedding"];
let stepTimer=null;
async function analyze(dataURL){
  $("#results").classList.add("hidden");
  const st=$("#status"); st.classList.remove("hidden");
  $("#status-title").textContent="Analyzing"; $("#status-msg").textContent="running the full interpretability stack…";
  const steps=$("#status-steps"); steps.innerHTML=STEPS.map(s=>`<span class="st" data-s="${s}">${s}</span>`).join("");
  let i=0; clearInterval(stepTimer);
  stepTimer=setInterval(()=>{ const els=$$("#status-steps .st"); if(i>0)els[i-1].classList.replace("on","done");
    if(i<els.length){els[i].classList.add("on");i++;} },420);
  st.scrollIntoView({behavior:"smooth",block:"center"});
  try{
    const r=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:dataURL,model:CURRENT_MODEL,meta:meta(),options:{mc_dropout:true,mc_samples:15}})});
    const j=await r.json(); if(!j.ok) throw new Error(j.error||"failed");
    clearInterval(stepTimer); $$("#status-steps .st").forEach(e=>e.classList.add("done"));
    LAST=j; setTimeout(()=>{ st.classList.add("hidden"); $("#results").classList.remove("hidden"); render(j);
      window.FX_observe&&window.FX_observe(); $("#results").scrollIntoView({behavior:"smooth"});
      window.FX&&FX.toast(`Diagnosis: ${j.prediction.info.name||j.prediction.class} · ${(j.prediction.prob*100).toFixed(0)}%`); },300);
  }catch(e){ clearInterval(stepTimer); $("#status-title").textContent="Error"; $("#status-msg").textContent=e.message; }
}

/* ---- render ---- */
function render(j){
  $("#img-proc").src=j.processed_image;
  $("#infer-chip").textContent=`${j.prediction.inference_ms} ms · ${j.model.device.toUpperCase()}`;
  $("#img-meta").innerHTML=[`orig ${j.image_meta.original_size.join("×")}`,`input ${j.image_meta.input_size.join("×")}`,
    `mode ${j.image_meta.mode}`,`${j.model.architecture.split(".")[0]}`,`${j.model.params_millions}M`].map(t=>`<span>${t}</span>`).join("");

  const p=j.prediction, pct=Math.round(p.prob*100);
  $("#verdict").innerHTML=`<div class="ring" id="ring"><b>0%</b></div>
    <div class="vmain"><h2>${p.info.name||p.class}</h2>
      <span class="grp ${p.info.group}">${p.info.group||""} · ${p.info.risk||""} risk</span>
      <p>${p.info.desc||""}</p></div>`;
  animateRing($("#ring"),pct);

  decisionBanner(j.decision_support);

  $("#topk").innerHTML=j.topk.map(t=>`<div class="row"><span class="cname">${t.info.name||t.class}</span>
    <div class="bar"><i data-w="${(t.prob*100).toFixed(1)}"></i></div><span class="pct mono">${(t.prob*100).toFixed(1)}%</span></div>`).join("");
  setTimeout(()=>$$("#topk .bar i").forEach(i=>i.style.width=i.dataset.w+"%"),120);

  softmaxChart(j.softmax); analytics(j.uncertainty); uncChart(j.uncertainty);
  heatTabs(j.heatmaps); film(j.pipeline); featureMaps(j.feature_maps); netflow(j.feature_maps); latent(j.latent);
}

/* ---- clinical decision support (abstention / melanoma-safety / fairness) ---- */
const LEVEL_META={
  urgent_referral:{cls:"urgent",ic:"⚠",tag:"Urgent referral"},
  refer_to_dermatologist:{cls:"refer",ic:"🩺",tag:"Refer to dermatologist"},
  decision_support_ok:{cls:"ok",ic:"✓",tag:"Decision support"}
};
function decisionBanner(d){ const box=$("#decision"); if(!box)return; if(!d){box.innerHTML="";return;}
  const lm=LEVEL_META[d.level]||LEVEL_META.refer_to_dermatologist;
  let html=`<div class="ds-card ds-${lm.cls}">
    <div class="ds-head"><span class="ds-ic">${lm.ic}</span><b>${d.headline}</b><span class="ds-tag">${lm.tag}</span></div>`;
  if(d.melanoma_safety) html+=`<div class="ds-mel">🛡 ${d.melanoma_safety}</div>`;
  if(d.reasons&&d.reasons.length) html+=`<div class="ds-why">Why: ${d.reasons.join("; ")}.</div>`;
  if(d.malignant_probability!=null) html+=`<div class="ds-mal">Cumulative malignant probability: <b>${(d.malignant_probability*100).toFixed(0)}%</b></div>`;
  if(d.fairness_caution) html+=`<div class="ds-fair">⚖ ${d.fairness_caution}</div>`;
  html+=`<div class="ds-disc">${d.disclaimer}</div></div>`;
  box.innerHTML=html; }

function animateRing(el,target){ const b=el.querySelector("b"); const t0=performance.now(),dur=1100;
  (function s(t){ const k=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-k,3); const v=Math.round(target*e);
    el.style.setProperty("--p",v); b.textContent=v+"%"; if(k<1)requestAnimationFrame(s); })(t0); }

/* ---- charts ---- */
function softmaxChart(sm){ if(!window.Chart)return; const ctx=$("#softmaxChart"); charts.sm&&charts.sm.destroy();
  const top=sm.map(x=>x.prob).indexOf(Math.max(...sm.map(x=>x.prob)));
  const g=ctx.getContext("2d").createLinearGradient(0,0,0,210); g.addColorStop(0,"#38e0ff"); g.addColorStop(1,"#8b6dff");
  charts.sm=new Chart(ctx,{type:"bar",data:{labels:sm.map(s=>s.class),
    datasets:[{data:sm.map(s=>+(s.prob*100).toFixed(2)),
      backgroundColor:sm.map((s,i)=>i===top?g:"rgba(120,160,230,.22)"),borderRadius:8,borderSkipped:false,maxBarThickness:46}]},
    options:{plugins:{legend:{display:false},tooltip:{backgroundColor:"rgba(8,12,22,.95)",borderColor:"rgba(56,224,255,.3)",borderWidth:1,padding:12,cornerRadius:10,callbacks:{label:c=>` ${c.parsed.y}%`}}},
      scales:{y:{ticks:{callback:v=>v+"%"},grid:{color:C_GRID}},x:{ticks:{color:"#cfe0ff"},grid:{display:false}}}}}); }

function uncChart(u){ if(!window.Chart)return; const ctx=$("#uncChart"); charts.unc&&charts.unc.destroy();
  charts.unc=new Chart(ctx,{type:"radar",data:{labels:["Confidence","Certainty","Margin","Purity","Stability"],
    datasets:[{data:[u.confidence,1-u.normalized_entropy,u.margin_top1_top2,1-u.gini_impurity,1-(u.mc_dropout_mean_std||0)*5].map(v=>+Math.max(0,Math.min(1,v)).toFixed(3)),
      backgroundColor:"rgba(56,224,255,.16)",borderColor:"#38e0ff",borderWidth:2,pointBackgroundColor:"#8b6dff",pointRadius:3}]},
    options:{plugins:{legend:{display:false}},scales:{r:{min:0,max:1,angleLines:{color:C_GRID},grid:{color:C_GRID},
      pointLabels:{color:C_TXT,font:{size:11}},ticks:{display:false}}}}}); }

function analytics(u){ const cell=(l,v,sm="")=>`<div class="metric"><div class="l">${l}</div><div class="v">${v}<small>${sm}</small></div></div>`;
  $("#analytics").innerHTML=[cell("Confidence",(u.confidence*100).toFixed(1)+"%"),cell("Entropy",u.entropy.toFixed(2)," nats"),
    cell("Norm entropy",u.normalized_entropy.toFixed(2)),cell("Margin",u.margin_top1_top2.toFixed(2)),
    cell("Eff. classes",u.effective_classes),u.epistemic_uncertainty!=null?cell("Epistemic",u.epistemic_uncertainty.toFixed(2)):"",
    `<div class="metric"><div class="l">Confidence band</div><div class="v band-${u.confidence_band}">${u.confidence_band}</div></div>`].join(""); }

/* ---- heat ---- */
function heatTabs(h){ const defs=[["grad_cam","Grad-CAM","Class-discriminative evidence from the final block."],
  ["grad_cam_pp","Grad-CAM++","Sharper localisation via higher-order gradients."],
  ["attention_rollout","Attention","ViT-native: how attention flows cls→patches."],
  ["saliency","Saliency","Raw input-gradient sensitivity."],
  ["score_cam","Score-CAM","Gradient-free mask-perturbation importance."]];
  const tabs=$("#heat-tabs"); tabs.innerHTML="";
  defs.forEach(([k,n,cap],idx)=>{ const avail=k==="score_cam"||!!h[k];
    const t=document.createElement("div"); t.className="tab"+(idx===0?" active":"")+(avail?"":" disabled");
    t.textContent=n; t.dataset.k=k; t.dataset.cap=cap;
    t.onclick=()=>{$$("#heat-tabs .tab").forEach(x=>x.classList.remove("active"));t.classList.add("active");showHeat(k,cap);};
    tabs.appendChild(t); });
  showHeat("grad_cam",defs[0][2]); }
function showHeat(k,cap){ const img=$("#heat-img");
  if(k==="score_cam"&&!LAST.heatmaps.score_cam){ $("#heat-caption").textContent="Click ＋ Run Score-CAM."; img.src=LAST.processed_image; return; }
  img.style.opacity=0; setTimeout(()=>{img.src=LAST.heatmaps[k]||LAST.processed_image;img.style.opacity=1;},150);
  $("#heat-caption").textContent=cap; }
$("#btn-scorecam").onclick=async()=>{ const b=$("#btn-scorecam"); b.innerHTML="Running Score-CAM…";
  try{ const r=await fetch("/api/scorecam",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image:lastImageDataURL,model:CURRENT_MODEL,meta:meta()})});
    const j=await r.json(); if(j.ok){LAST.heatmaps.score_cam=j.score_cam; $$("#heat-tabs .tab").forEach(x=>{if(x.dataset.k==="score_cam"){x.classList.remove("disabled");x.click();}}); FX&&FX.toast("Score-CAM ready");}
  }finally{ b.innerHTML="✓ Score-CAM ready"; } };

/* ---- film ---- */
let filmI=0,filmTimer=null;
function film(stages){ if(!stages||!stages.length)return; const strip=$("#film-strip"); strip.innerHTML="";
  stages.forEach((s,i)=>{const im=new Image();im.src=s.img;im.onclick=()=>setFilm(i);strip.appendChild(im);});
  $("#film-slider").max=stages.length-1; $("#film-slider").oninput=e=>setFilm(+e.target.value);
  $("#film-play").onclick=togglePlay; setFilm(0); }
function setFilm(i){ const s=LAST.pipeline[i]; filmI=i; const img=$("#film-img");
  img.style.opacity=0; setTimeout(()=>{img.src=s.img;img.style.opacity=1;},120);
  $("#film-title").textContent=s.title; $("#film-caption").textContent=s.caption;
  $("#film-slider").value=i; $("#film-count").textContent=`${i+1}/${LAST.pipeline.length}`;
  $$("#film-strip img").forEach((x,j)=>x.classList.toggle("active",j===i)); }
function togglePlay(){ if(filmTimer){clearInterval(filmTimer);filmTimer=null;$("#film-play").textContent="▶ Play";return;}
  $("#film-play").textContent="⏸ Pause";
  filmTimer=setInterval(()=>{let n=(filmI+1)%LAST.pipeline.length;setFilm(n);
    if(n===LAST.pipeline.length-1){clearInterval(filmTimer);filmTimer=null;$("#film-play").textContent="▶ Play";}},1100); }
/* arrow keys scrub */
addEventListener("keydown",e=>{ if(!LAST||$("#results").classList.contains("hidden"))return;
  if(e.target.matches("input,select"))return;
  if(e.key==="ArrowRight")setFilm((filmI+1)%LAST.pipeline.length);
  if(e.key==="ArrowLeft")setFilm((filmI-1+LAST.pipeline.length)%LAST.pipeline.length); });

/* ---- feature maps + netflow ---- */
function featureMaps(f){ $("#fmaps").innerHTML=(f||[]).map(m=>`<div class="fmap"><img src="${m.img}"/><span class="tag">block ${m.layer} · ${Math.round(m.depth_frac*100)}%</span></div>`).join(""); }
function netflow(f){ const layers=[{n:"Input"},...(f||[]).map(m=>({n:"Blk "+m.layer,a:m.depth_frac})),{n:"Head"}];
  const W=560,H=180,n=layers.length,gap=W/(n-1); let svg=`<svg viewBox="0 0 ${W} ${H}">`;
  svg+=`<defs><linearGradient id="nf" x1="0" x2="1"><stop offset="0" stop-color="#38e0ff"/><stop offset="1" stop-color="#8b6dff"/></linearGradient></defs>`;
  for(let i=0;i<n-1;i++)svg+=`<line x1="${i*gap}" y1="${H/2}" x2="${(i+1)*gap}" y2="${H/2}" stroke="url(#nf)" stroke-width="2" opacity=".4"/>`;
  layers.forEach((L,i)=>{const a=L.a||(i===0?.2:.95),r=11+a*15;
    svg+=`<circle cx="${i*gap}" cy="${H/2}" r="${r}" fill="rgba(56,224,255,${.2+a*.55})" stroke="#38e0ff" stroke-width="1.5"><animate attributeName="r" values="${r};${r+2};${r}" dur="${2+i*.2}s" repeatCount="indefinite"/></circle>`;
    svg+=`<text x="${i*gap}" y="${H/2+r+15}" fill="#9fb2d6" font-size="10" text-anchor="middle">${L.n}</text>`;});
  $("#netflow").innerHTML=svg+"</svg>"; }

/* ---- latent ---- */
function latent(L){ const near=L.nearest_class_centroid;
  if(near){ const mx=Math.max(...near.map(x=>x.similarity));
    $("#latent-near").innerHTML="<h4 style='margin:0 0 12px;font-size:13px;color:#cfe0ff'>Cosine similarity to class centroids</h4>"+
      near.slice(0,6).map(s=>`<div class="row"><span class="cname">${s.class}</span><div class="bar"><i data-w="${Math.max(0,s.similarity/mx*100)}"></i></div><span class="pct mono">${s.similarity.toFixed(2)}</span></div>`).join("");
    setTimeout(()=>$$("#latent-near .bar i").forEach(i=>i.style.width=i.dataset.w+"%"),150);
  } else $("#latent-near").innerHTML=`<p class="hint">Latent centroids not built yet — run <span class="mono">demo_app/_compute_prototypes.py</span>. Embedding dim ${L.embedding_dim}, ‖z‖=${L.embedding_l2}.</p>`;
  embeddingScatter(L); }
async function embeddingScatter(L){ if(!window.Chart)return; const r=await fetch("/api/embedding_space").then(x=>x.json()).catch(()=>({ok:false}));
  const ctx=$("#latentChart"); charts.lat&&charts.lat.destroy();
  if(!r.ok){ $("#latent-note").textContent="2-D latent map appears once prototypes are built."; return; }
  const by={}; r.coords.forEach((c,i)=>{(by[r.labels[i]]=by[r.labels[i]]||[]).push({x:c[0],y:c[1]});});
  const pal=["#38e0ff","#8b6dff","#43f3c7","#ffc24b","#ff647c","#5ad1ff","#b07cff","#7CFFB2","#ff9ec4"];
  const ds=Object.keys(by).map((k,i)=>({label:k,data:by[k],backgroundColor:pal[i%pal.length],pointRadius:2.5}));
  if(L.uploaded_xy)ds.push({label:"⭐ uploaded",data:[{x:L.uploaded_xy[0],y:L.uploaded_xy[1]}],backgroundColor:"#fff",pointRadius:9,pointStyle:"star"});
  charts.lat=new Chart(ctx,{type:"scatter",data:{datasets:ds},options:{plugins:{legend:{labels:{color:C_TXT,boxWidth:8,font:{size:10}}}},
    scales:{x:{ticks:{display:false},grid:{color:C_GRID}},y:{ticks:{display:false},grid:{color:C_GRID}}}}});
  $("#latent-note").textContent=`${r.method.toUpperCase()} projection of training embeddings · ⭐ = your image.`; }

/* ---- controls ---- */
$("#btn-present").onclick=()=>{ document.body.classList.toggle("present");
  if(document.body.classList.contains("present")&&!document.fullscreenElement)document.documentElement.requestFullscreen?.();
  FX&&FX.toast(document.body.classList.contains("present")?"Presentation mode · press Esc to exit":"Exited presentation"); };
$("#btn-new").onclick=()=>{ $("#results").classList.add("hidden"); $("#studio").scrollIntoView({behavior:"smooth"}); $("#file").value=""; };
$("#btn-report").onclick=()=>window.print();
