/* MNKAI-III web demo (GitHub Pages). No model runs in this page. It answers the app's API calls with the
   responses the study laptop's server returned for three sample photographs (scripts/record_pages_demo.py),
   with the laptop's timing, so the real front end (app.js, fx.js and styles.css, unchanged apart from asset
   paths; scripts/build_pages_site.py) behaves exactly as it does on the laptop. Nothing is uploaded. */
"use strict";
(function(){
  const ROOT=new URL("./",location.href);
  const realFetch=window.fetch.bind(window);
  const json=o=>new Response(JSON.stringify(o),{status:200,headers:{"Content-Type":"application/json"}});
  const wait=ms=>new Promise(r=>setTimeout(r,ms));
  const memo={};
  const data=name=>memo[name]||(memo[name]=realFetch(new URL("data/"+name,ROOT)).then(r=>{
    if(!r.ok) throw new Error("web demo: "+name+" is missing"); return r.json(); })
    .catch(e=>{ delete memo[name]; throw e; }));          // a network blip must not stick: the next tap retries
  let latest={analyze:0,scorecam:0};                       // only the newest request of each kind is answered
  /* which sample is on screen: app.js sets its global SAMPLE when a sample is picked and clears it for uploads */
  const current=()=>{ try{ return (typeof SAMPLE!=="undefined"&&SAMPLE)?SAMPLE.file.replace(/\.[^.]+$/,""):null; }catch(e){ return null; } };
  const ANALYZE_MS=1900;   // measured in the browser on the study laptop: 1.8-2.1 s per photograph
  const OWN_PHOTO="This web demo replays recorded examples. To analyse a new photograph the system has to run on the study laptop. Pick one of the samples below.";
  const NO_GEMINI="Not available in the web demo. On the study laptop this optional step asks Gemini over the internet, so it is never used for study patients.";

  window.fetch=async function(input,init){
    const url=typeof input==="string"?input:(input&&input.url)||"";
    if(!url.startsWith("/api/")) return realFetch(input,init);
    const path=url.split("?")[0];
    if(path==="/api/health") return json(await data("health.json"));
    if(path==="/api/evidence") return json(await data("evidence.json"));
    if(path==="/api/question_lrs") return json(await data("question_lrs.json"));
    if(path==="/api/embedding_space") return json(await data("embedding_space.json"));
    if(path==="/api/models"){ const h=await data("health.json"); return json({ok:true,models:h.models}); }
    if(path==="/api/analyze"||path==="/api/scorecam"){
      const kind=path.slice(5), id=++latest[kind], stem=current();
      if(!stem){ await wait(700); return json({ok:false,error:OWN_PHOTO}); }
      const rec=(await data("recording.json")).samples.find(s=>s.file.replace(/\.[^.]+$/,"")===stem)||{};
      const ms=kind==="analyze"?ANALYZE_MS:Math.max(700,rec.scorecam_ms||800);
      const [body]=await Promise.all([data(kind+"_"+stem+".json"),wait(ms)]);
      if(id!==latest[kind]) return new Promise(()=>{});    // a newer tap superseded this one: never render a stale answer
      return json(body);
    }
    if(path==="/api/interview_questions"||path==="/api/interview_fuse"){ await wait(500); return json({ok:false,error:NO_GEMINI}); }
    return json({ok:false,error:"Not available in the web demo."});
  };

  /* patient context: app.js calls this when a sample is picked; the form (locked in the web demo) shows the
     sample's own skin type and body site, so it can never disagree with the recorded answer */
  window.WD_context=function(s){
    const c=(s&&s.context)||{}, v=c.fitzpatrick?String(c.fitzpatrick):"";
    const age=document.getElementById("m-age"), sex=document.getElementById("m-sex"), fst=document.getElementById("m-fst");
    if(age) age.value=c.age||""; if(sex) sex.value=c.sex||""; if(fst) fst.value=v;
    document.querySelectorAll("#fst-picker button").forEach(b=>b.classList.toggle("on",b.dataset.v===v));
    if(typeof setSite==="function") setSite(c.site||"");
  };

  /* "i" tooltips: hover on a computer; a tap opens one (and closes the others), a tap elsewhere closes it */
  document.addEventListener("click",e=>{
    const t=e.target.closest&&e.target.closest(".wd-i");
    document.querySelectorAll(".wd-i.open").forEach(x=>{ if(x!==t) x.classList.remove("open"); });
    if(t){ t.classList.toggle("open"); e.stopPropagation(); }
  });

  /* warm the cache for the sample answers once the page is idle, so a tap on a phone feels as quick as the laptop */
  addEventListener("load",()=>setTimeout(()=>data("recording.json").then(r=>r.samples.forEach((s,i)=>
    setTimeout(()=>data("analyze_"+s.file.replace(/\.[^.]+$/,"")+".json").catch(()=>{}),1500*i))).catch(()=>{}),2500));
})();
