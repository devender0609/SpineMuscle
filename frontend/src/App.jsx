import React,{useEffect,useMemo,useState} from 'react';
import {UploadCloud,CheckCircle2,Activity,ShieldCheck,AlertTriangle,FileDown,RefreshCcw,ChevronRight} from 'lucide-react';

const API=(import.meta.env.VITE_API_URL||'').replace(/\/$/,'');
const api=(p,o={})=>fetch(`${API}${p}`,o).then(async r=>{const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(j.detail||j.error||`HTTP ${r.status}`);return j});

function Step({n,title,active,done}){return <div className={`step ${active?'active':''} ${done?'done':''}`}><span>{done?<CheckCircle2 size={18}/>:n}</span><b>{title}</b></div>}
function Badge({children,tone='neutral'}){return <span className={`badge ${tone}`}>{children}</span>}

export default function App(){
  const [job,setJob]=useState(null),[status,setStatus]=useState(null),[err,setErr]=useState(''),[busy,setBusy]=useState(false);
  const [files,setFiles]=useState([]),[selected,setSelected]=useState({});
  const stage=status?.stage||'upload';
  const stepNo=stage==='upload'?1:stage==='level_review'?2:stage==='processing'?3:stage==='complete'?4:stage==='blocked'?4:1;

  useEffect(()=>{if(!job)return;let stop=false;const tick=async()=>{try{const s=await api(`/api/jobs/${job}`);if(!stop)setStatus(s)}catch(e){if(!stop)setErr(e.message)}};tick();const id=setInterval(tick,1600);return()=>{stop=true;clearInterval(id)}},[job]);

  async function start(){
    if(!files.length)return;
    setBusy(true);setErr('');
    try{
      const fd=new FormData(); for(const f of files)fd.append('files',f);
      const r=await api('/api/jobs',{method:'POST',body:fd});setJob(r.job_id);setStatus(r);
    }catch(e){setErr(e.message)}finally{setBusy(false)}
  }
  async function confirmLevels(){
    setBusy(true);setErr('');
    try{await api(`/api/jobs/${job}/confirm-levels`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({selections:selected})});}
    catch(e){setErr(e.message)}finally{setBusy(false)}
  }
  async function reset(){setJob(null);setStatus(null);setFiles([]);setSelected({});setErr('')}

  const proposals=status?.level_proposals||{};
  useEffect(()=>{if(stage==='level_review'){const x={};Object.entries(proposals).forEach(([lev,cands])=>{if(cands?.length)x[lev]=cands[0].dicom_file});setSelected(x)}},[stage,JSON.stringify(proposals)]);

  return <div className="shell">
    <header><div className="brand"><div className="logo">SM</div><div><h1>SpineMuscle AI</h1><p>Research MRI muscle-quality pipeline</p></div></div><Badge tone="research">RESEARCH USE</Badge></header>
    <main>
      <section className="hero"><div><p className="eyebrow">Lumbar MRI • Paravertebral muscle analysis</p><h2>One study. One review. One research report.</h2><p>Frozen level-localization proposals, MuscleMap segmentation, confirmed-plane muscle metrics, and frozen PVMQ quality control—without presenting unsupported patient-specific PJK risk.</p></div><div className="heroCard"><ShieldCheck/><b>Frozen-method governance</b><small>v3 level localizer + MuscleMap v1.4 + PVMQ AUTO v5.2 when its required CSF mask input is available.</small></div></section>

      <nav className="steps"><Step n="1" title="Upload" active={stepNo===1} done={stepNo>1}/><ChevronRight/><Step n="2" title="Confirm levels" active={stepNo===2} done={stepNo>2}/><ChevronRight/><Step n="3" title="Analyze" active={stepNo===3} done={stepNo>3}/><ChevronRight/><Step n="4" title="Report" active={stepNo===4}/></nav>

      {err&&<div className="alert error"><AlertTriangle/> {err}</div>}

      {!job&&<section className="panel upload">
        <div className="drop" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();setFiles([...e.dataTransfer.files])}}>
          <UploadCloud size={46}/><h3>Upload one lumbar MRI study</h3><p>Select the DICOM files from a single patient study.</p>
          <label className="button primary">Choose DICOM files<input hidden multiple type="file" onChange={e=>setFiles([...e.target.files])}/></label>
          <small>{files.length?`${files.length} files selected`:'No files selected'}</small>
        </div>
        <button className="button primary wide" disabled={!files.length||busy} onClick={start}>{busy?'Uploading…':'Start analysis'}</button>
      </section>}

      {stage==='level_review'&&<section className="panel">
        <div className="panelHead"><div><p className="eyebrow">Human verification safeguard</p><h3>Confirm the four PVMQ planes</h3><p>Top frozen-v3 proposal is preselected. Change only if the neighboring image better represents the intended mid-disc plane.</p></div><Badge tone="good">4 levels required</Badge></div>
        {Object.entries(proposals).map(([lev,cands])=><div className="level" key={lev}><div className="levelTitle"><b>{lev}</b><span>{cands?.[0]?.score!=null?`Top score ${Number(cands[0].score).toFixed(4)}`:''}</span></div><div className="cards">{cands.map((c,i)=><label className={`candidate ${selected[lev]===c.dicom_file?'chosen':''}`} key={c.dicom_file+i}><input type="radio" name={lev} checked={selected[lev]===c.dicom_file} onChange={()=>setSelected({...selected,[lev]:c.dicom_file})}/><img src={`${API}/api/jobs/${job}/preview/${encodeURIComponent(c.preview_id)}`} /><div><b>#{i+1}</b><small>{c.dicom_file}</small><small>score {Number(c.score).toFixed(4)}</small></div></label>)}</div></div>)}
        <button className="button primary wide" disabled={busy||Object.keys(selected).length!==4} onClick={confirmLevels}>{busy?'Confirming…':'Confirm levels & continue'}</button>
      </section>}

      {stage==='processing'&&<section className="panel center"><Activity className="spin" size={42}/><h3>Analyzing confirmed planes</h3><p>{status?.message||'Running the controlled research pipeline…'}</p><div className="progress"><i style={{width:`${status?.progress||15}%`}}/></div><small>{status?.progress||0}% complete</small></section>}

      {(stage==='complete'||stage==='blocked')&&<section className="panel report">
        <div className="panelHead"><div><p className="eyebrow">Research measurement report</p><h3>{stage==='complete'?'Analysis complete':'Measurement complete with PVMQ gate'}</h3></div><Badge tone={stage==='complete'?'good':'warn'}>{status?.qc_decision||'REVIEW'}</Badge></div>
        <div className="metrics">
          <article><span>Four-level muscle SI</span><strong>{fmt(status?.result?.muscle_numerator)}</strong><small>bilateral multifidus + erector spinae</small></article>
          <article><span>CSF mean SI</span><strong>{fmt(status?.result?.csf_mean_si)}</strong><small>{status?.result?.csf_tier||'Not available'}</small></article>
          <article className="accent"><span>PVMQ</span><strong>{fmt(status?.result?.pvmq,5)}</strong><small>research measurement</small></article>
          <article><span>QC</span><strong>{status?.qc_decision||'—'}</strong><small>{status?.result?.qc_reason||'Frozen hierarchy'}</small></article>
        </div>
        {status?.result?.levels&&<table><thead><tr><th>Level</th><th>Muscle SI</th><th>CSA mm²</th><th>Geometry</th></tr></thead><tbody>{status.result.levels.map(x=><tr key={x.level}><td>{x.level}</td><td>{fmt(x.mean_si)}</td><td>{fmt(x.csa_mm2,1)}</td><td>{x.geometry_qc}</td></tr>)}</tbody></table>}
        {stage==='blocked'&&<div className="alert warn"><AlertTriangle/><div><b>PVMQ not issued.</b><br/><span>{status?.result?.qc_reason||'Required frozen v5.2 CSF input was unavailable or failed quality control.'}</span></div></div>}
        <div className="note"><ShieldCheck/><div><b>Interpretation boundary</b><p>This report documents a technical research measurement. It does not provide a validated patient-specific PJK probability, clinical cutoff, diagnosis, or treatment recommendation.</p></div></div>
        <div className="actions">{status?.report_url&&<a className="button primary" href={`${API}${status.report_url}`}><FileDown size={18}/> Download report</a>}<button className="button" onClick={reset}><RefreshCcw size={18}/> New study</button></div>
      </section>}
    </main>
    <footer>SpineMuscle AI • Technical research workflow • Frozen-method audit trail retained</footer>
  </div>
}
const fmt=(v,d=3)=>v==null||Number.isNaN(Number(v))?'—':Number(v).toFixed(d);
