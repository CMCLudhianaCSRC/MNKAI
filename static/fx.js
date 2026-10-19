/* MNKAI-III · motion & effects engine ===================================== */
"use strict";
(function(){
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---- cinematic loader ---- */
const msgs=["Starting","Loading PanDerm-L","Preparing the five classifiers","Calibrating","Ready"];
let li=0; const sub=$("#loader-sub"), fill=$("#loader-bar-fill");
const loadStart=performance.now();
const tick=setInterval(()=>{ li=Math.min(li+1,msgs.length-1); if(sub)sub.textContent=msgs[li];
  if(fill)fill.style.width=(20+li*20)+"%"; },360);
function hideLoader(){
  clearInterval(tick); if(fill)fill.style.width="100%"; if(sub)sub.textContent="Ready.";
  setTimeout(()=>$("#loader")?.classList.add("done"),350);
}
window.addEventListener("load",()=>{ const wait=Math.max(0,1500-(performance.now()-loadStart)); setTimeout(hideLoader,wait); });
setTimeout(hideLoader,4000); // safety

/* ---- neural particle background ---- */
(function neural(){
  if(reduce) return;
  const cv=$("#neural-bg"); if(!cv||getComputedStyle(cv).display==="none") return;   // hidden in the glass theme const ctx=cv.getContext("2d");
  let W,H,DPR=Math.min(devicePixelRatio||1,2), nodes=[], mouse={x:-999,y:-999};
  function resize(){ W=cv.width=innerWidth*DPR; H=cv.height=innerHeight*DPR; cv.style.width=innerWidth+"px"; cv.style.height=innerHeight+"px";
    const n=Math.round(Math.min(72,innerWidth/22));
    nodes=Array.from({length:n},()=>({x:Math.random()*W,y:Math.random()*H,
      vx:(Math.random()-.5)*.18*DPR,vy:(Math.random()-.5)*.18*DPR,r:(Math.random()*1.6+.6)*DPR})); }
  resize(); addEventListener("resize",resize);
  addEventListener("mousemove",e=>{mouse.x=e.clientX*DPR;mouse.y=e.clientY*DPR;});
  let running=true; document.addEventListener("visibilitychange",()=>{running=!document.hidden; if(running)loop();});
  const LINK=130*DPR;
  function loop(){
    if(!running) return;
    ctx.clearRect(0,0,W,H);
    for(const p of nodes){
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>W)p.vx*=-1; if(p.y<0||p.y>H)p.vy*=-1;
      const dx=p.x-mouse.x,dy=p.y-mouse.y,md=Math.hypot(dx,dy);
      if(md<150*DPR){p.x+=dx/md*.6;p.y+=dy/md*.6;}
    }
    for(let i=0;i<nodes.length;i++){
      for(let j=i+1;j<nodes.length;j++){
        const a=nodes[i],b=nodes[j],d=Math.hypot(a.x-b.x,a.y-b.y);
        if(d<LINK){ const o=(1-d/LINK)*.5;
          ctx.strokeStyle=`rgba(90,180,255,${o*.5})`; ctx.lineWidth=DPR*.6;
          ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke(); }
      }
    }
    for(const p of nodes){ ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,7); ctx.fillStyle="rgba(120,200,255,.8)";ctx.fill(); }
    requestAnimationFrame(loop);
  }
  loop();
})();

/* ---- scroll reveal (staggered) ---- */
const io=new IntersectionObserver((es)=>{
  es.forEach(e=>{ if(e.isIntersecting){
    const sibs=[...e.target.parentElement.children].filter(c=>c.classList.contains("reveal")||c.classList.contains("r-item"));
    const idx=sibs.indexOf(e.target);
    e.target.style.transitionDelay=Math.min(idx,8)*70+"ms";
    e.target.classList.add("in"); io.unobserve(e.target);
  }});
},{threshold:.12});
function observeReveals(){ $$(".reveal:not(.in),.r-item:not(.in)").forEach(el=>io.observe(el)); }
observeReveals(); window.FX_observe=observeReveals;

/* ---- count-up ---- */
const cup=new IntersectionObserver((es)=>{ es.forEach(e=>{ if(e.isIntersecting){ countUp(e.target); cup.unobserve(e.target);} }); },{threshold:.5});
$$(".stat-num[data-count]").forEach(el=>cup.observe(el));
function countUp(el){
  const to=+el.dataset.count, dec=+(el.dataset.dec||0), suf=el.dataset.suffix||"", dur=1400, t0=performance.now();
  (function step(t){ const p=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-p,3);
    el.textContent=(to*e).toFixed(dec)+(p===1?suf:""); if(p<1)requestAnimationFrame(step); else el.textContent=to.toFixed(dec)+suf;
  })(t0);
}

/* ---- tilt ---- */
if(!reduce) $$(".tilt").forEach(el=>{
  el.addEventListener("mousemove",e=>{ const r=el.getBoundingClientRect();
    const rx=((e.clientY-r.top)/r.height-.5)*-6, ry=((e.clientX-r.left)/r.width-.5)*6;
    el.style.transform=`perspective(700px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-6px)`; });
  el.addEventListener("mouseleave",()=>el.style.transform="");
});

/* ---- ripple on buttons ---- */
document.addEventListener("click",e=>{ const b=e.target.closest(".btn-primary,.btn-ghost,.tab,.icon-btn"); if(!b)return;
  const r=b.getBoundingClientRect(),s=document.createElement("span"); s.className="ripple";
  const d=Math.max(r.width,r.height); s.style.width=s.style.height=d+"px";
  s.style.left=e.clientX-r.left-d/2+"px"; s.style.top=e.clientY-r.top-d/2+"px";
  b.appendChild(s); setTimeout(()=>s.remove(),600);
});

/* ---- nav scrolled ---- */
addEventListener("scroll",()=>{ $("#nav")?.classList.toggle("scrolled",scrollY>30); },{passive:true});

/* ---- toast ---- */
window.FX={ toast(msg,ms=2600){ const t=$("#toast"); if(!t)return; t.textContent=msg; t.classList.add("show");
  clearTimeout(t._t); t._t=setTimeout(()=>t.classList.remove("show"),ms); } };

/* ---- onboarding tour ---- */
const TOUR=[
  {h:"Welcome to MNKAI-III",p:"Dermatology decision support built for Indian and darker skin. 20 seconds to get oriented."},
  {h:"1 · Scan",p:"Drop, paste or pick a sample photo. Add skin tone and body site if you know them."},
  {h:"2 · Triage and diagnosis",p:"A routine / review / urgent gauge, the ranked differential, the 90% possible-diagnoses list and the cancer check."},
  {h:"3 · Clinical brief",p:"On the study laptop a clinical brief follows. It is hidden in this web demo until a dermatologist has reviewed it."},
  {h:"4 · Trust but verify",p:"Photo-quality checks, the model's track record for this condition, heatmaps, and evidence by skin tone. H opens the session log."},
];
let ts=0;
function showTour(i){ ts=i; const el=$("#tour-step"); const s=TOUR[i];
  el.innerHTML=`<h4>${s.h}</h4><p>${s.p}</p>`;
  $("#tour-next").textContent=i===TOUR.length-1?"Done ✓":"Next →";
  $("#tour").classList.remove("hidden"); }
function endTour(){ $("#tour").classList.add("hidden"); try{localStorage.setItem("mnkai_tour","1")}catch(e){} }
$("#btn-tour")?.addEventListener("click",()=>showTour(0));
$("#tour-skip")?.addEventListener("click",endTour);
$("#tour-next")?.addEventListener("click",()=>{ ts<TOUR.length-1?showTour(ts+1):endTour(); });
try{ if(!localStorage.getItem("mnkai_tour") && !/[?&]embed(=|&|$)/.test(location.search)) setTimeout(()=>showTour(0),2400); }catch(e){}   // no tour inside the presentation

/* ---- keyboard shortcuts ---- */
addEventListener("keydown",e=>{
  if(e.target.matches("input,select,textarea"))return;
  const k=e.key.toLowerCase();
  if(k==="u"){ $("#file")?.click(); }
  else if(k==="h"){ /* session log handled in app.js */ }
  else if(k==="p"){ $("#btn-present")?.click(); }
  else if(k==="?"||k==="/"){ showTour(0); }
  else if(k==="f"){ document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen?.(); }
  else if(k==="escape"){ document.body.classList.remove("present"); $("#tour")?.classList.add("hidden"); }
});

/* fade kbd hint after a while */
setTimeout(()=>$("#kbd-hint")?.style.setProperty("opacity",".55"),6000);
})();
