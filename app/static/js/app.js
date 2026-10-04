/* =============================================
   SemanticEye | Main Application JavaScript
   ============================================= */

'use strict';

const State = {
  inputMode: 'text', loadedText: '', loadedFilename: 'Pasted_Text.txt',
  activeReport: null, activeSentenceIdx: null, inspectorView: 'interactive',
  activeFilter: 'all', samples: [], corpus: [], isAnalyzing: false,
  currentUser: null
};
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 48;

document.addEventListener('DOMContentLoaded', async () => {
  loadTheme(); fetchSamples(); updateLiveStats(); checkCookieConsent();
  await loadSession();
});

/* ---- API helper: same-origin cookies, JSON errors, auto sign-in prompt on 401 ---- */
async function api(url, options) {
  const res = await fetch(url, Object.assign({ credentials: 'same-origin' }, options || {}));
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON body */ }
  if (res.status === 401 && !url.startsWith('/api/auth/')) {
    State.currentUser = null; updateAuthNavUI(); openAuthModal();
    throw new Error('Please sign in to continue.');
  }
  if (!res.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === 'string' ? detail : (Array.isArray(detail) ? 'Please check your input.' : 'Request failed (' + res.status + ')'));
  }
  return data;
}
function postJSON(url, body) {
  return api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}
async function loadSession() {
  try { const d = await api('/api/auth/me'); State.currentUser = d.user; } catch (e) { State.currentUser = null; }
  updateAuthNavUI();
  if (State.currentUser) fetchCorpus();
}
function requireLogin(msg) {
  if (State.currentUser) return true;
  showToast(msg || 'Please sign in first.', 'info'); openAuthModal(); return false;
}
function signInPromptRow(cols) {
  return '<tr><td colspan="' + cols + '" style="text-align:center;padding:24px;color:var(--text-muted);">Please <a href="#" onclick="openAuthModal();return false;">sign in</a> to see your saved scans.</td></tr>';
}

function loadTheme() {
  document.documentElement.setAttribute('data-theme', localStorage.getItem('se_theme') || 'dark');
}
function toggleTheme() {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('se_theme', next);
}
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function switchMainTab(tab) {
  ['detector','history','corpus','about'].forEach(t => {
    const p = document.getElementById('pane'+capitalize(t));
    const b = document.getElementById('tabBtn'+capitalize(t));
    if (p) { p.classList.toggle('active', t===tab); p.classList.toggle('hidden', t!==tab); }
    if (b) b.classList.toggle('active', t===tab);
  });
  if (tab==='corpus')  fetchCorpus();
  if (tab==='history') fetchScanHistory();
}
function openPrivacyModal()  { document.getElementById('privacyModal')?.classList.remove('hidden'); }
function closePrivacyModal() { document.getElementById('privacyModal')?.classList.add('hidden'); }
function checkCookieConsent() { if(localStorage.getItem('se_cookies_accepted')!=='true') document.getElementById('cookieBanner')?.classList.remove('hidden'); }
function acceptCookies() {
  localStorage.setItem('se_cookies_accepted','true');
  document.getElementById('cookieBanner')?.classList.add('hidden');
  showToast('Essential cookies accepted.','info');
}
function resetCookieConsent() {
  localStorage.removeItem('se_cookies_accepted');
  document.getElementById('cookieBanner')?.classList.remove('hidden');
  closePrivacyModal(); showToast('Cookie notice re-opened.','info');
}
function setInputMode(mode) {
  State.inputMode = mode;
  document.getElementById('sectionTextInput')?.classList.toggle('hidden', mode!=='text');
  document.getElementById('sectionFileInput')?.classList.toggle('hidden', mode!=='file');
  document.getElementById('modeBtnText')?.classList.toggle('active', mode==='text');
  document.getElementById('modeBtnFile')?.classList.toggle('active', mode==='file');
  updateLiveStats(mode==='text' ? document.getElementById('documentTextInput')?.value||'' : State.loadedText);
}
function handleTextareaInput() {
  const txt = document.getElementById('documentTextInput')?.value||'';
  State.loadedText=txt; State.loadedFilename='Pasted_Text.txt'; updateLiveStats(txt);
}
function updateLiveStats(text) {
  if (text===undefined||text===null) text = State.inputMode==='text' ? (document.getElementById('documentTextInput')?.value||'') : State.loadedText;
  const words=text.trim()?text.trim().split(/\s+/).length:0;
  const sents=text.trim()?text.split(/[.!?]+/).filter(Boolean).length:0;
  const readTime=Math.max(0,Math.round(words/200*10)/10);
  const g=id=>document.getElementById(id);
  if(g('statWordCount'))     g('statWordCount').innerHTML='<b>'+words.toLocaleString()+'</b> words';
  if(g('statSentenceCount')) g('statSentenceCount').innerHTML='<b>'+sents+'</b> sentences';
  if(g('statCharCount'))     g('statCharCount').innerHTML='<b>'+text.length.toLocaleString()+'</b> characters';
  if(g('statReadTime'))      g('statReadTime').innerHTML='~<b>'+readTime+'</b> min read';
}
function clearInput() {
  const el=document.getElementById('documentTextInput'); if(el) el.value='';
  State.loadedText=''; updateLiveStats('');
}
function handleDragOver(e) { e.preventDefault(); document.getElementById('fileDropzone')?.classList.add('drag-over'); }
function handleDragLeave()  { document.getElementById('fileDropzone')?.classList.remove('drag-over'); }
function handleFileDrop(e)  { e.preventDefault(); document.getElementById('fileDropzone')?.classList.remove('drag-over'); const f=e.dataTransfer.files[0]; if(f) processUploadedFile(f); }
function handleFileSelect(e){ const f=e.target.files[0]; if(f) processUploadedFile(f); }
async function processUploadedFile(file) {
  if (!['txt','pdf','docx'].includes(file.name.split('.').pop().toLowerCase())) { showToast('Only .txt, .pdf, .docx supported.','error'); return; }
  if(!requireLogin('Please sign in to upload documents.')) return;
  const fd=new FormData(); fd.append('file',file);
  try {
    showToast('Extracting document text...','info');
    const data=await api('/api/upload-file',{method:'POST',body:fd});
    State.loadedText=data.text; State.loadedFilename=file.name;
    const ne=document.getElementById('loadedFileName'); const se=document.getElementById('loadedFileSub');
    if(ne) ne.textContent=file.name;
    if(se) se.textContent=data.stats.word_count.toLocaleString()+' words - '+data.stats.sentence_count+' sentences extracted';
    document.getElementById('fileLoadedInfo')?.classList.remove('hidden');
    document.getElementById('fileDropzone')?.classList.add('hidden');
    updateLiveStats(data.text); showToast('"'+file.name+'" loaded - '+data.stats.word_count+' words','success');
  } catch(err) { showToast('Upload failed: '+err.message,'error'); }
}
function removeLoadedFile() {
  State.loadedText=''; State.loadedFilename='Pasted_Text.txt';
  document.getElementById('fileLoadedInfo')?.classList.add('hidden');
  document.getElementById('fileDropzone')?.classList.remove('hidden');
  const fi=document.getElementById('fileInputElement'); if(fi) fi.value='';
  updateLiveStats('');
}
async function fetchSamples() { try { const r=await fetch('/api/samples'); const d=await r.json(); State.samples=d.samples; } catch {} }
function loadSamplePreset(id) {
  const s=State.samples.find(x=>x.id===id); if(!s){showToast('Sample not found.','error');return;}
  setInputMode('text'); const el=document.getElementById('documentTextInput'); if(el) el.value=s.text;
  State.loadedText=s.text; State.loadedFilename=s.id+'.txt'; updateLiveStats(s.text);
  showToast('Loaded: "'+s.title+'"','info');
}
function updateThresholdDisplay(type,val) {
  const el=document.getElementById(type==='plagiarism'?'plagiarismThresholdVal':'aiThresholdVal');
  if(el) el.textContent=val+'%';
}

async function runAnalysis() {
  if(State.isAnalyzing) return;
  const text=State.inputMode==='text'?(document.getElementById('documentTextInput')?.value.trim()||''):State.loadedText.trim();
  if(!text||text.split(/\s+/).length<8){showToast('Please enter at least 15 words.','error');return;}
  if(!requireLogin('Please sign in to run a check - your results are saved privately to your account.')) return;
  const pt=parseInt(document.getElementById('plagiarismThresholdSlider').value)/100;
  const at=parseInt(document.getElementById('aiThresholdSlider').value)/100;
  State.isAnalyzing=true;
  const bl=document.getElementById('btnAuditLabel'); const sp=document.getElementById('auditSpinner'); const ba=document.getElementById('btnRunAudit');
  if(bl) bl.textContent='Analyzing...'; if(sp) sp.classList.remove('hidden'); if(ba) ba.disabled=true;
  try {
    const report=await postJSON('/api/analyze',{text,plagiarism_threshold:pt,ai_threshold:at,filename:State.loadedFilename});
    State.activeReport=report; renderResults(report);
    showToast('Analysis complete! Scroll down for results.','success');
  } catch(err) { showToast('Analysis failed: '+err.message,'error'); }
  finally {
    State.isAnalyzing=false;
    if(bl) bl.textContent='Run Deep Inspection'; if(sp) sp.classList.add('hidden'); if(ba) ba.disabled=false;
  }
}

function renderResults(report) {
  const rs=document.getElementById('resultsSection'); if(!rs) return;
  rs.classList.remove('hidden'); rs.scrollIntoView({behavior:'smooth',block:'start'});
  const badge=document.getElementById('overallIntegrityBadge'); const desc=document.getElementById('overallIntegrityDesc');
  if(badge){badge.textContent=report.integrity_status;badge.style.background=report.status_color+'22';badge.style.color=report.status_color;badge.style.border='1px solid '+report.status_color+'55';}
  if(desc) desc.textContent=report.status_description;
  const mid=document.getElementById('metaReportId'); const mhash=document.getElementById('metaDocHash');
  if(mid)   mid.textContent=report.report_id;
  if(mhash) mhash.textContent=report.document_fingerprint;
  const pf=report.summary?report.summary.plagiarism_score:0;
  const af=report.summary?report.summary.ai_score:0;
  setGaugeValue('gaugePlagCircle','plagScoreNumber',pf);
  setGaugeValue('gaugeAiCircle','aiScoreNumber',af);
  const pp=Math.round(pf*100); const ap=Math.round(af*100);
  const sents=report.sentences||[];
  const flagged=sents.filter(s=>s.is_plagiarized).length;
  const flaggedAi=sents.filter(s=>s.is_ai).length;
  const pfe=document.getElementById('plagFlaggedCount'); if(pfe) pfe.textContent=flagged+' / '+sents.length;
  const afe=document.getElementById('aiFlaggedCount');   if(afe) afe.textContent=flaggedAi+' / '+sents.length;
  const pse=document.getElementById('plagSourcesCount'); if(pse) pse.textContent=(report.matched_sources||[]).length+' sources';
  const ppill=document.getElementById('plagPill'); if(ppill) ppill.textContent=pp>40?'High Risk':pp>20?'Moderate':'Low';
  const apill=document.getElementById('aiPill');   if(apill) apill.textContent=ap>50?'AI Likely':ap>25?'Mixed':'Human';
  const aiA=report.ai_analysis||{};
  const cv=aiA.burstiness?aiA.burstiness.coefficient_of_variation:null;
  const mk=aiA.discourse?aiA.discourse.matches_count:0;
  const be=document.getElementById('aiBurstinessMetric'); if(be) be.textContent=(cv!==null?cv:'-')+(cv!==null&&parseFloat(cv)>=0.35?' (Natural)':' (Low Variance)');
  const me=document.getElementById('aiMarkersCount');    if(me) me.textContent=mk+' detected';
  const st=report.stats||{}; const g=id=>document.getElementById(id);
  if(g('statDocWords'))        g('statDocWords').textContent=(st.word_count||0).toLocaleString();
  if(g('statDocSentences'))    g('statDocSentences').textContent=st.sentence_count||0;
  if(g('statLexicalRichness')) g('statLexicalRichness').textContent=(st.lexical_richness_pct||0).toFixed(1)+'%';
  if(g('statAvgSentenceLen'))  g('statAvgSentenceLen').textContent=st.avg_sentence_length||(st.word_count&&st.sentence_count?Math.round(st.word_count/st.sentence_count):0);
  if(g('statEngineMethod'))    g('statEngineMethod').textContent=aiA.method||'SemanticEye Multi-Signal Ensemble';
  renderShapBars(report.shap_explanations||[]);
  renderDocumentView(sents);
  renderSourcesTable(report.matched_sources||[]);
  updateFilterCounts(sents);
  closeSentenceDrawer();
}

function setGaugeValue(cid,tid,frac) {
  const pct=Math.round(frac*100); const c=document.getElementById(cid); const t=document.getElementById(tid);
  if(!c||!t) return;
  c.style.strokeDasharray=GAUGE_CIRCUMFERENCE; c.style.strokeDashoffset=GAUGE_CIRCUMFERENCE*(1-frac);
  c.style.stroke=pct>55?'#e05c6a':pct>25?'#d4956a':'#7daa85';
  t.textContent=pct+'%';
}

function renderShapBars(features) {
  const grid=document.getElementById('shapFeatureGrid'); if(!grid) return;
  if(!features||!features.length){grid.innerHTML='';return;}
  grid.innerHTML=features.map(f=>{
    const bc=f.direction==='risk'?'#e05c6a':'#7daa85';
    return '<div class="shap-feature-row">'+
      '<div class="shap-feature-meta"><span class="shap-feature-name">'+escapeHtml(f.name)+'</span><span class="shap-feature-val">'+escapeHtml(f.val)+'</span></div>'+
      '<div class="shap-bar-bg"><div class="shap-bar-fill" style="width:'+Math.min(100,f.impact_pct)+'%;background:'+bc+';"></div></div>'+
      '<p class="shap-feature-exp">'+escapeHtml(f.explanation)+'</p></div>';
  }).join('');
}

function renderDocumentView(sentences) {
  const c=document.getElementById('documentRenderView'); if(!c) return;
  c.innerHTML=sentences.map((s,i)=>{
    const tag=s.tag||'original';
    const h=State.activeFilter!=='all'&&State.activeFilter!==tag;
    return '<span class="sent-span sent-tag-'+tag+(h?' hidden-filter':'')+'" id="sentSpan_'+i+'" onclick="selectSentence('+i+')">'+escapeHtml(s.text)+'</span> ';
  }).join('');
  renderSplitView(sentences);
}
function renderSplitView(sentences) {
  const sub=document.getElementById('splitSubmittedContent'); const ref=document.getElementById('splitReferenceContent');
  if(!sub||!ref) return;
  sub.innerHTML=sentences.map((s,i)=>'<div class="split-sentence sent-tag-'+(s.tag||'original')+'" onclick="selectSentence('+i+')">'+escapeHtml(s.text)+'</div>').join('');
  ref.innerHTML=sentences.map(s=>s.matched_sentence?'<div class="split-sentence" style="background:rgba(100,100,200,0.08);">'+escapeHtml(s.matched_sentence)+'</div>':'<div class="split-sentence" style="color:var(--text-muted);font-style:italic;">No match found</div>').join('');
}
function updateFilterCounts(sents) {
  const c={all:sents.length,plagiarized:0,ai:0,original:0};
  sents.forEach(s=>{ if(s.tag==='plagiarized')c.plagiarized++; else if(s.tag==='ai')c.ai++; else if(s.tag==='both'){c.plagiarized++;c.ai++;} else c.original++; });
  const g=id=>document.getElementById(id);
  if(g('countAll'))   g('countAll').textContent=c.all;
  if(g('countPlag'))  g('countPlag').textContent=c.plagiarized;
  if(g('countAi'))    g('countAi').textContent=c.ai;
  if(g('countClean')) g('countClean').textContent=c.original;
}
function selectSentence(idx) {
  if(!State.activeReport) return;
  const s=(State.activeReport.sentences||[])[idx]; if(!s) return;
  document.querySelectorAll('.sent-span').forEach((el,i)=>el.classList.toggle('active-selected',i===idx));
  State.activeSentenceIdx=idx; openSentenceDrawer(s,idx);
}
function openSentenceDrawer(s,idx) {
  const d=document.getElementById('sentenceDrawer'); if(!d) return;
  const b=document.getElementById('drawerSentenceBadge'); if(b) b.textContent='Sentence #'+(idx+1)+' - '+getBadgeLabel(s.tag);
  const it=document.getElementById('drawerInputText'); if(it) it.textContent=s.text;
  const ps=document.getElementById('drawerPlagScore');  if(ps) ps.textContent=(s.plag_pct||0)+'%';
  const sn=document.getElementById('drawerSourceName'); if(sn) sn.textContent=s.plag_source_title||'None';
  const st=document.getElementById('drawerSourceText'); if(st) st.textContent=s.matched_sentence||'No reference overlap found.';
  const dh=document.getElementById('drawerDiffHtml');   const dc=document.getElementById('drawerDiffContainer');
  if(s.word_diff_html&&dh&&dc){dh.innerHTML=s.word_diff_html;dc.classList.remove('hidden');}
  else if(dc) dc.classList.add('hidden');
  const as=document.getElementById('drawerAiScore');  if(as) as.textContent=(s.ai_pct||0)+'%';
  const ar=document.getElementById('drawerAiReason'); if(ar) ar.textContent=(s.ai_reasons&&s.ai_reasons.length)?s.ai_reasons.join(' | '):'Natural human sentence structure.';
  d.classList.remove('hidden');
}
function closeSentenceDrawer() {
  document.getElementById('sentenceDrawer')?.classList.add('hidden');
  document.querySelectorAll('.sent-span').forEach(el=>el.classList.remove('active-selected'));
  State.activeSentenceIdx=null;
}
function getBadgeLabel(tag) {
  return tag==='plagiarized'?'Copied':tag==='ai'?'AI-Generated':tag==='both'?'Copied and AI':'Original';
}
function switchInspectorView(view) {
  State.inspectorView=view;
  document.getElementById('viewBtnInteractive')?.classList.toggle('active',view==='interactive');
  document.getElementById('viewBtnSplit')?.classList.toggle('active',view==='split');
  document.getElementById('documentRenderView')?.classList.toggle('hidden',view!=='interactive');
  document.getElementById('splitDiffView')?.classList.toggle('hidden',view!=='split');
}
function filterSentences(filterType,btn) {
  State.activeFilter=filterType;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');
  if(!State.activeReport) return;
  (State.activeReport.sentences||[]).forEach((s,i)=>{
    document.getElementById('sentSpan_'+i)?.classList.toggle('hidden-filter',filterType!=='all'&&filterType!==s.tag);
  });
}
function renderSourcesTable(sources) {
  const tbody=document.getElementById('sourcesTableBody'); if(!tbody) return;
  if(!sources||!sources.length){tbody.innerHTML='<tr><td colspan="4" style="text-align:center;color:var(--clean-green);padding:20px;">No corpus source matches found. All sentences appear original.</td></tr>';return;}
  tbody.innerHTML=sources.map(src=>{
    const pct=Math.round((src.plagiarism_contribution||0)*100);
    return '<tr><td><b>'+escapeHtml(src.title||src.filename)+'</b></td>'+
      '<td><span style="background:#e05c6a22;color:#e05c6a;border:1px solid #e05c6a44;padding:3px 8px;border-radius:10px;">'+pct+'% match</span></td>'+
      '<td>'+(src.matched_sentences_count||0)+' sentence(s)</td>'+
      '<td><span class="badge-status status-flagged">Match Verified</span></td></tr>';
  }).join('');
}

function corpusCard(doc, mine) {
  return '<div class="corpus-item-card">'+
    '<div class="corpus-card-header">'+
      '<span class="corpus-doc-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path></svg></span>'+
      '<div style="flex:1;min-width:0;"><h4>'+escapeHtml(doc.title)+'</h4><span class="corpus-meta">'+doc.word_count.toLocaleString()+' words - '+doc.sentence_count+' sentences</span></div>'+
      (mine ? '<button class="btn-ghost-sm" style="color:var(--plag-red);" title="Delete" onclick="deleteCorpusDoc('+Number(doc.id)+')">&times;</button>' : '')+
    '</div>'+
    '<p class="corpus-snippet">'+escapeHtml(doc.snippet)+'</p>'+
  '</div>';
}
async function fetchCorpus() {
  const grid=document.getElementById('corpusGrid'); if(!grid) return;
  if(!State.currentUser){
    grid.innerHTML='<div style="padding:20px;color:var(--text-muted);">Please <a href="#" onclick="openAuthModal();return false;">sign in</a> to view the reference library and add your own private documents.</div>';
    return;
  }
  try {
    const data=await api('/api/reference-docs');
    State.corpus=data.builtin.concat(data.mine);
    grid.innerHTML=
      '<h3 class="corpus-section-title" style="grid-column:1/-1;">My private documents <span class="corpus-tag">Only you can see and match against these</span></h3>'+
      (data.mine.length ? data.mine.map(d=>corpusCard(d,true)).join('') : '<div style="grid-column:1/-1;color:var(--text-muted);">You haven\'t added any documents yet.</div>')+
      '<h3 class="corpus-section-title" style="grid-column:1/-1;margin-top:10px;">Built-in library <span class="corpus-tag">Shared, read-only</span></h3>'+
      data.builtin.map(d=>corpusCard(d,false)).join('');
  } catch(err) { grid.innerHTML='<div style="color:var(--plag-red);padding:20px;">Failed to load corpus: '+escapeHtml(err.message)+'</div>'; }
}
async function uploadToCorpus(e) {
  const input=e.target; const file=input.files[0]; if(!file) return;
  if(!requireLogin('Please sign in to add reference documents.')) { input.value=''; return; }
  const fd=new FormData(); fd.append('file',file);
  try {
    showToast('Uploading "'+file.name+'"...','info');
    const data=await api('/api/reference-docs',{method:'POST',body:fd});
    showToast('"'+file.name+'" added ('+data.sentence_count+' sentences)','success'); fetchCorpus();
  } catch(err) { showToast('Corpus upload failed: '+err.message,'error'); }
  finally { input.value=''; }
}
async function deleteCorpusDoc(id) {
  if(!confirm('Remove this document from your private library?')) return;
  try { await api('/api/reference-docs/'+id,{method:'DELETE'}); showToast('Document removed.','info'); fetchCorpus(); }
  catch(err) { showToast('Delete failed: '+err.message,'error'); }
}

function exportPrintReport() {
  if(!State.activeReport){showToast('No active report to export.','error');return;}
  const r=State.activeReport; const pw=window.open('','_blank');
  if(!pw){showToast('Pop-up blocked. Please allow pop-ups.','error');return;}
  const pp=Math.round((r.summary?r.summary.plagiarism_score:0)*100);
  const ap=Math.round((r.summary?r.summary.ai_score:0)*100);
  const st=r.stats||{}; const sents=r.sentences||[];
  pw.document.write('<!DOCTYPE html><html><head><title>SemanticEye Certificate</title><style>'+
    'body{font-family:sans-serif;margin:40px;color:#1e293b;line-height:1.6}'+
    '.hdr{border-bottom:2px solid #e2e8f0;padding-bottom:16px;margin-bottom:24px;display:flex;justify-content:space-between;align-items:center}'+
    '.brand{font-size:1.4rem;font-weight:800;color:#d87093}'+
    '.scores{display:flex;gap:20px;margin:24px 0}.sc{flex:1;padding:20px;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0}'+
    '.sn{font-size:2.2rem;font-weight:800;margin-top:4px}'+
    '.si{padding:10px 14px;margin-bottom:8px;border-radius:8px;border-left:4px solid #cbd5e1;background:#f8fafc;font-size:.9rem}'+
    '.s-plagiarized{border-left-color:#e05c6a;background:#fef2f2}.s-ai{border-left-color:#d4956a;background:#fffbeb}'+
    '.s-both{border-left-color:#b07db0;background:#fdf4ff}.s-original{border-left-color:#7daa85;background:#f0fdf4}'+
    '.sm{font-size:.78rem;color:#64748b;margin-top:4px}'+
    '.ft{margin-top:40px;border-top:1px solid #e2e8f0;padding-top:16px;font-size:.8rem;color:#94a3b8;text-align:center}'+
    '</style></head><body>'+
    '<div class="hdr"><div><div class="brand">SemanticEye Document Certificate</div>'+
    '<div style="font-size:.85rem;color:#64748b;">File: <b>'+escapeHtml(r.filename)+'</b> - ID: <code>'+escapeHtml(r.report_id)+'</code></div></div>'+
    '<div style="padding:4px 12px;border-radius:99px;font-weight:700;background:'+r.status_color+'22;color:'+r.status_color+';border:1px solid '+r.status_color+'44;">'+escapeHtml(r.integrity_status)+'</div></div>'+
    '<div class="scores">'+
    '<div class="sc"><div style="font-size:.85rem;color:#64748b;font-weight:600;">Plagiarism Score</div><div class="sn" style="color:'+(pp>40?'#e05c6a':'#7daa85')+'">'+pp+'%</div></div>'+
    '<div class="sc"><div style="font-size:.85rem;color:#64748b;font-weight:600;">AI Probability</div><div class="sn" style="color:'+(ap>50?'#d4956a':'#7daa85')+'">'+ap+'%</div></div>'+
    '<div class="sc"><div style="font-size:.85rem;color:#64748b;font-weight:600;">Total Words</div><div class="sn" style="font-size:1.4rem;color:#334155;margin-top:12px;">'+(st.word_count||0).toLocaleString()+'</div></div>'+
    '</div><div style="font-size:1.1rem;font-weight:700;margin:24px 0 12px;">Sentence-Level Breakdown</div>'+
    sents.map((s,i)=>'<div class="si s-'+(s.tag||'original')+'"><b style="color:#0f172a">[Sentence #'+(i+1)+']</b> '+escapeHtml(s.text)+
      '<div class="sm">'+escapeHtml(s.label||'')+' - Plag: '+(s.plag_pct||0)+'% - AI: '+(s.ai_pct||0)+'%'+(s.plag_source_title?' - Source: <b>'+escapeHtml(s.plag_source_title)+'</b>':'')+' </div></div>').join('')+
    '<div class="ft">Generated by SemanticEye - '+escapeHtml(r.timestamp)+'</div></body></html>');
  pw.document.close(); pw.focus(); setTimeout(()=>pw.print(),600);
}

function showToast(message,type) {
  type=type||'info';
  const c=document.getElementById('toastContainer'); if(!c) return;
  const t=document.createElement('div'); t.className='toast toast-'+type;
  t.innerHTML='<span>'+escapeHtml(message)+'</span>'; c.appendChild(t);
  setTimeout(()=>{ t.style.opacity='0'; t.style.transform='translateY(16px)'; t.style.transition='all 0.3s ease'; setTimeout(()=>t.remove(),320); },3800);
}
function escapeHtml(str) {
  if(!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function updateAuthNavUI() {
  const c=document.getElementById('authNavArea'); if(!c) return;
  if(State.currentUser) {
    c.innerHTML='<div class="user-pill-dropdown"><span class="user-pill-name">'+escapeHtml(State.currentUser.username)+'</span><button class="nav-ghost-btn" onclick="handleLogout()" title="Sign Out">Sign Out</button></div>';
  } else {
    c.innerHTML='<button class="nav-auth-btn" id="btnOpenAuth" onclick="openAuthModal()">Sign In</button>';
  }
}
function openAuthModal()  { document.getElementById('authModal')?.classList.remove('hidden'); }
function closeAuthModal() { document.getElementById('authModal')?.classList.add('hidden'); }
function switchAuthTab(mode) {
  const il=mode==='login';
  document.getElementById('authTabBtnLogin')?.classList.toggle('active',il);
  document.getElementById('authTabBtnRegister')?.classList.toggle('active',!il);
  document.getElementById('loginForm')?.classList.toggle('hidden',!il);
  document.getElementById('registerForm')?.classList.toggle('hidden',il);
  const t=document.getElementById('authModalTitle'); if(t) t.textContent=il?'Sign In to Account':'Create New Account';
}
function afterAuthChange() {
  // Never keep one account's data on screen after switching accounts.
  State.activeReport=null; State.corpus=[];
  document.getElementById('resultsSection')?.classList.add('hidden');
  updateAuthNavUI(); fetchCorpus(); fetchScanHistory();
}
async function handleLogin(e) {
  e.preventDefault();
  const id=document.getElementById('loginIdentifier').value.trim();
  const pw=document.getElementById('loginPassword').value;
  try {
    const data=await postJSON('/api/auth/login',{username_or_email:id,password:pw});
    State.currentUser=data.user; document.getElementById('loginPassword').value='';
    closeAuthModal(); afterAuthChange(); showToast('Welcome back, '+data.user.username+'!','success');
  } catch(err) { showToast('Login failed: '+err.message,'error'); }
}
async function handleRegister(e) {
  e.preventDefault();
  const u=document.getElementById('regUsername').value.trim();
  const em=document.getElementById('regEmail').value.trim();
  const pw=document.getElementById('regPassword').value;
  try {
    const data=await postJSON('/api/auth/register',{username:u,email:em,password:pw});
    State.currentUser=data.user; document.getElementById('regPassword').value='';
    closeAuthModal(); afterAuthChange(); showToast('Account created! Welcome, '+data.user.username+'!','success');
  } catch(err) { showToast('Registration failed: '+err.message,'error'); }
}
async function handleLogout() {
  try { await api('/api/auth/logout',{method:'POST'}); } catch(e) { /* already signed out */ }
  State.currentUser=null; afterAuthChange(); showToast('Logged out successfully.','info');
}
async function fetchScanHistory() {
  const tbody=document.getElementById('historyTableBody'); if(!tbody) return;
  if(!State.currentUser){ tbody.innerHTML=signInPromptRow(6); return; }
  tbody.innerHTML='<tr><td colspan="6" style="text-align:center;padding:24px;">Loading saved scans...</td></tr>';
  try {
    const data=await api('/api/history');
    if(!data.history||!data.history.length){tbody.innerHTML='<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-muted);">No saved scans found. Run a check to record your history!</td></tr>';return;}
    tbody.innerHTML=data.history.map(item=>{
      const ds=new Date(String(item.created_at).replace(' ','T')+'Z').toLocaleString();
      const pp=Math.round((item.plagiarism_score||0)*100); const ap=Math.round((item.ai_score||0)*100);
      const pc=pp>40?'#e05c6a':pp>15?'#d4956a':'#7daa85';
      const ac=ap>50?'#d4956a':ap>20?'#b07db0':'#7daa85';
      const rid=escapeHtml(item.report_id);
      return '<tr>'+
        '<td><b>'+escapeHtml(item.filename)+'</b></td>'+
        '<td><span style="background:'+pc+'22;color:'+pc+';border:1px solid '+pc+'44;padding:4px 10px;border-radius:12px;">'+pp+'%</span></td>'+
        '<td><span style="background:'+ac+'22;color:'+ac+';border:1px solid '+ac+'44;padding:4px 10px;border-radius:12px;">'+ap+'%</span></td>'+
        '<td>'+(item.total_words||0).toLocaleString()+' words</td>'+
        '<td style="font-size:.82rem;color:var(--text-secondary);">'+ds+'</td>'+
        '<td><div style="display:flex;gap:8px;">'+
          '<button class="btn-primary btn-sm" onclick="loadHistoryReport(\''+rid+'\')">View Report</button>'+
          '<button class="btn-ghost-sm" onclick="deleteHistoryItem(\''+rid+'\')" style="color:var(--plag-red);" title="Delete">&times;</button>'+
        '</div></td></tr>';
    }).join('');
  } catch(err) { tbody.innerHTML='<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--plag-red);">Failed to load history: '+escapeHtml(err.message)+'</td></tr>'; }
}
async function loadHistoryReport(rid) {
  try {
    showToast('Retrieving saved audit report...','info');
    const report=await api('/api/report/'+encodeURIComponent(rid));
    State.activeReport=report; switchMainTab('detector'); renderResults(report);
    showToast('Loaded report for "'+report.filename+'"','success');
  } catch(err) { showToast('Failed to load report: '+err.message,'error'); }
}
async function deleteHistoryItem(rid) {
  if(!confirm('Delete this scan record?')) return;
  try {
    await api('/api/history/'+encodeURIComponent(rid),{method:'DELETE'});
    showToast('Scan record deleted.','info'); fetchScanHistory();
  } catch(err) { showToast('Delete failed: '+err.message,'error'); }
}
