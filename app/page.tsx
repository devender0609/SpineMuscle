import validation from '../data/validation.json';

function pct(x:number){ return (x*100).toFixed(1)+'%'; }
function f3(x:number){ return Number(x).toFixed(3); }

export default function Home(){
  const d=validation.development;
  const l=validation.locked;
  return <main className="wrap">
    <header className="hero">
      <div>
        <div className="eyebrow">RESEARCH USE ONLY</div>
        <h1>SpineMuscle AI</h1>
        <p>Frozen PVMQ v5.2 technical-validation dashboard</p>
      </div>
      <span className="badge">v5.2 FROZEN</span>
    </header>

    <section className="notice">
      <b>Scope:</b> This release reports technical measurement validation. It does not provide a validated PJK probability, clinical cutoff, or treatment recommendation.
    </section>

    <section className="grid">
      <article className="card"><span>Development acceptance</span><strong>{pct(d.acceptance_rate)}</strong><small>{d.accepted}/{d.entered} entering PVMQ stage</small></article>
      <article className="card"><span>Locked-test acceptance</span><strong>{pct(l.acceptance_rate)}</strong><small>{l.accepted}/{l.entered} entering PVMQ stage</small></article>
      <article className="card"><span>Development median PVMQ</span><strong>{f3(d.pvmq_median)}</strong><small>IQR {f3(d.q1)}â€“{f3(d.q3)}</small></article>
      <article className="card"><span>Locked median PVMQ</span><strong>{f3(l.pvmq_median)}</strong><small>IQR {f3(l.q1)}â€“{f3(l.q3)}</small></article>
    </section>

    <section className="panel">
      <h2>Locked-test validation</h2>
      <div className="facts">
        <div><b>100</b><span>Prespecified locked patients</span></div>
        <div><b>98</b><span>Complete 4-level muscle numerator</span></div>
        <div><b>68</b><span>Accepted frozen-v5.2 PVMQ</span></div>
        <div><b>0</b><span>Accepted PVMQ &gt; 1</span></div>
      </div>
    </section>

    <section className="two">
      <article className="panel">
        <h2>Scanner performance</h2>
        <table><thead><tr><th>Scanner</th><th>Dev</th><th>Locked</th></tr></thead>
        <tbody>
          {['GE 1.5T','Philips 1.5T','Philips 3T'].map(name=>{
            const dv=validation.scanner.find((x:any)=>x.cohort==='Development'&&x.scanner===name);
            const lk=validation.scanner.find((x:any)=>x.cohort==='Locked Test'&&x.scanner===name);
            return <tr key={name}><td>{name}</td><td>{dv ? pct(dv.acceptance_rate) : "—"}</td><td>{lk ? pct(lk.acceptance_rate) : "—"}</td></tr>
          })}
        </tbody></table>
        <p className="foot">Scanner-specific differences remain; scanner independence is not claimed.</p>
      </article>
      <article className="panel">
        <h2>Accepted QC tiers</h2>
        <table><thead><tr><th>Tier</th><th>Dev</th><th>Locked</th></tr></thead>
        <tbody>
          {['A','B','C'].map(t=><tr key={t}><td>Tier {t}</td><td>{(validation.tier_counts as any).Development[t]}</td><td>{(validation.tier_counts as any)['Locked Test'][t]}</td></tr>)}
        </tbody></table>
        <p className="foot">Tier distribution did not show a clear development-to-test shift.</p>
      </article>
    </section>

    <section className="panel">
      <h2>MRI inference deployment status</h2>
      <p>The frozen measurement logic is validated here against the 500-patient research dataset. Arbitrary one-upload MRI inference is intentionally not enabled in this Vercel release because the current exact lumbar-level mapping used during validation is dataset-specific and MuscleMap/dcm2niix require a containerized inference service.</p>
      <p className="foot">Recommended production architecture: GitHub â†’ Vercel frontend + separate containerized inference API. Add generalizable lumbar-level localization and validate it before enabling clinical-study uploads.</p>
    </section>
  </main>;
}


