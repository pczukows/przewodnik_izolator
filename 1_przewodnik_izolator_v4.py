import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.html("""
    <style>
    /* Powiększenie tekstu etykiety w st.metric */
    [data-testid="stMetricLabel"] div p {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    /* Opcjonalnie: Powiększenie samej ikony pytajnika (tooltipu), aby pasowała do tekstu */
    [data-testid="stMetricLabel"] button {
        transform: scale(1.3);
        margin-left: 8px !important;
    }
    </style>
""")

st.set_page_config(page_title="Przewodnik i izolator", page_icon="⚡", layout="wide")

MATERIALS = {
    "Metal — miedź": {
        "kind": "conductor", "rho": 1.68e-8, "color": "#4b5563",
    },
    "Izolator — tworzywo (model)": {
        "kind": "insulator", "leakage_r": 4e10, "breakdown_voltage": 12000.0,
        "post_breakdown_r": 220.0, "color": "#66553d",
    },
    "Izolator — ceramika (model)": {
        "kind": "insulator", "leakage_r": 2e11, "breakdown_voltage": 20000.0,
        "post_breakdown_r": 480.0, "color": "#706d67",
    },
}


def conductor_current(u, length, area_mm2, rho, temperature):
    resistance = rho * (1 + 0.0039 * (temperature - 20)) * length / (area_mm2 * 1e-6)
    return u / resistance, resistance


def insulator_current(u, leakage_r, breakdown_voltage, post_breakdown_r, softness):
    width = max(25.0, breakdown_voltage * softness)
    activation = 1 / (1 + np.exp(np.clip(-(abs(u) - breakdown_voltage) / width, -60, 60)))
    excess = np.sign(u) * max(abs(u) - 0.92 * breakdown_voltage, 0)
    return u / leakage_r + activation * excess / post_breakdown_r, activation


def nice_current(value):
    for scale, unit in [(1, "A"), (1e-3, "mA"), (1e-6, "µA"), (1e-9, "nA")]:
        if abs(value) >= scale:
            return f"{value / scale:.3f} {unit}"
    return f"{value * 1e12:.3f} pA"


def microscopic_animation(kind, voltage, current, activation, material_color, speed):
    config = json.dumps({
        "kind": kind, "u": float(voltage), "i": float(current),
        "activation": float(activation), "color": material_color, "speed": float(speed),
    })
    html = r'''
<div id="box">
  <canvas id="canvas"></canvas>
  <div id="legend">
    <span><b class="atom"></b> atom/jon sieci</span>
    <span><b class="electron"></b> elektron</span>
    <span><b class="ionized"></b> atom po utracie elektronu</span>
    <span>→ kierunek ruchu</span>
  </div>
</div>
<style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#07111f;color:#e5e7eb;font-family:system-ui,sans-serif}
#box{position:relative;width:100%;height:690px}canvas{display:block;width:100%;height:100%}
#legend{position:absolute;left:22px;right:22px;bottom:8px;display:flex;flex-wrap:wrap;justify-content:center;gap:10px 25px;padding:8px;background:rgba(7,17,31,.88);border-radius:8px;font-size:14px}
#legend b{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:-1px;margin-right:5px}.atom{background:#dce6f1;border:2px solid #8291a6}.electron{background:#55c7fa}.ionized{background:#fb923c;border:2px solid #fde68a}
</style>
<script>
const q=__CONFIG__, canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
let W,H,dpr,last=performance.now(); const electrons=[], atoms=[];
function random(i,k){let x=Math.sin(i*9277.3+k*81.7)*43758.55;return x-Math.floor(x)}
for(let row=0;row<5;row++)for(let col=0;col<11;col++)atoms.push({row,col,x:(col+.5)/11,y:(row+.5)/5});
for(let n=0;n<46;n++){let t=random(n,1)*Math.PI*2;electrons.push({x:random(n,2),y:random(n,3),t,w:random(n,4)*2-1})}
function resize(){dpr=Math.min(devicePixelRatio||1,2);W=canvas.clientWidth;H=canvas.clientHeight;canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0)}addEventListener('resize',resize);resize();
function arrow(x,y,vx,vy,color,scale=1){let n=Math.hypot(vx,vy)||1,L=11*scale,ex=x+vx/n*L,ey=y+vy/n*L,a=Math.atan2(ey-y,ex-x);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(ex,ey);ctx.stroke();ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-5*Math.cos(a-.55),ey-5*Math.sin(a-.55));ctx.lineTo(ex-5*Math.cos(a+.55),ey-5*Math.sin(a+.55));ctx.closePath();ctx.fill()}
function electrode(x,w,color,label,y0,y1){ctx.fillStyle=color;ctx.fillRect(x,y0,w,y1-y0);ctx.fillStyle='#fff';ctx.font='bold 19px system-ui';ctx.textAlign='center';ctx.fillText(label,x+w/2,y0-13)}
function drawAtom(x,y,ionized=false){ctx.beginPath();ctx.arc(x,y,ionized?8.5:7,0,Math.PI*2);ctx.fillStyle=ionized?'#fb923c':'#dce6f1';ctx.fill();ctx.strokeStyle=ionized?'#fde68a':'#8291a6';ctx.lineWidth=ionized?3:2;ctx.stroke();if(ionized){ctx.fillStyle='#fff';ctx.font='bold 12px system-ui';ctx.textAlign='center';ctx.fillText('+',x,y+4)}}
function frame(now){
  const dt=Math.min((now-last)/1000,.04); last=now; ctx.clearRect(0,0,W,H);
  const pad=Math.max(92,Math.min(130,W*.095)),x0=pad,x1=W-pad,y0=92,y1=H-125,pw=x1-x0,ph=y1-y0;
  const zero=Math.abs(q.u)<1e-9, positive=q.u>0, fieldDir=positive?1:-1;
  const ew=Math.max(48,Math.min(68,W*.055));
  electrode(16,ew,zero?'#64748b':positive?'#ef4444':'#2563eb',zero?'0 V':positive?'+':'−',y0,y1);
  electrode(W-16-ew,ew,zero?'#64748b':positive?'#2563eb':'#ef4444',zero?'0 V':positive?'−':'+',y0,y1);
  ctx.fillStyle=q.color;ctx.fillRect(x0,y0,pw,ph);ctx.strokeStyle='#8491a3';ctx.lineWidth=2;ctx.strokeRect(x0,y0,pw,ph);
  ctx.textAlign='center';ctx.font='bold 17px system-ui';ctx.fillStyle=zero?'#cbd5e1':'#d8b4fe';ctx.fillText(zero?'BRAK RÓŻNICY POTENCJAŁÓW — E = 0':'POLE ELEKTRYCZNE E',W/2,31);if(!zero)arrow(W/2-fieldDir*78,58,fieldDir,0,'#d8b4fe',3.5);

  if(q.kind==='conductor'){
    atoms.forEach(a=>drawAtom(x0+a.x*pw,y0+a.y*ph));
    const driftSign=zero?0:-fieldDir, drift=driftSign*Math.min(.19,.02+Math.log10(1+Math.abs(q.i))*.015);
    electrons.forEach((p,n)=>{p.t+=(p.w*.9+Math.sin(now*.0007+n)*.3)*dt*q.speed;let vx=Math.cos(p.t)*.10+drift,vy=Math.sin(p.t)*.10;p.x=(p.x+vx*dt*q.speed+1)%1;p.y=(p.y+vy*dt*q.speed+1)%1;let x=x0+p.x*pw,y=y0+p.y*ph;ctx.beginPath();ctx.arc(x,y,3.8,0,7);ctx.fillStyle='#55c7fa';ctx.fill();arrow(x,y,vx,vy,'#8be0ff',.85)});
    ctx.fillStyle=zero?'#facc15':'#22c55e';ctx.font='bold 16px system-ui';ctx.fillText(zero?'RUCH CIEPLNY: KIERUNKI LOSOWE, PRĄD NETTO = 0':'RUCH CIEPLNY + DODATKOWA SKŁADOWA DRYFU ELEKTRONÓW',W/2,H-85);
  } else {
    // Przebicie jest pokazane jako sekwencja jonizacji w środkowym rzędzie atomów.
    const middleRow=2, threshold=.08, active=Math.max(0,(q.activation-threshold)/(1-threshold));
    const front=active*12.5; // front jonizacji przechodzi przez 11 atomów
    const electronDirection=-fieldDir;
    atoms.forEach(a=>{
      const distanceAlong=electronDirection>0?a.col:(10-a.col);
      const ionized=a.row===middleRow && active>0 && distanceAlong<front;
      const polarization=zero?0:-fieldDir*Math.min(.018,Math.abs(q.u)/20000*.014);
      drawAtom(x0+a.x*pw,y0+a.y*ph,ionized);
      if(!ionized){ctx.beginPath();ctx.arc(x0+(a.x+polarization)*pw,y0+a.y*ph,3,0,7);ctx.fillStyle='#55c7fa';ctx.fill()}
    });
    if(active>0){
      // Każdy zjonizowany atom oddaje elektron; elektrony poruszają się skokowo do kolejnych miejsc.
      const count=Math.max(1,Math.floor(2+active*14));
      for(let n=0;n<count;n++){
        const phase=(now*.00022*q.speed+n/count)%1;
        const start=electronDirection>0?0:1;
        const xp=(start+electronDirection*phase+1)%1;
        const yNorm=(middleRow+.5)/5;
        const x=x0+xp*pw, y=y0+yNorm*ph+Math.sin(phase*Math.PI*2+n)*6;
        ctx.beginPath();ctx.arc(x,y,4.2,0,7);ctx.fillStyle='#55c7fa';ctx.fill();arrow(x,y,electronDirection,0,'#8be0ff',1.0);
      }
      // Krótkie impulsy między sąsiednimi atomami zamiast falistego kanału.
      const y=y0+(middleRow+.5)/5*ph;ctx.strokeStyle='rgba(253,224,71,.48)';ctx.lineWidth=3;
      for(let col=0;col<10;col++){const order=electronDirection>0?col:9-col;if(order<front){let xa=x0+(col+.62)/11*pw,xb=x0+(col+1.38)/11*pw;ctx.beginPath();ctx.moveTo(xa,y);ctx.lineTo(xb,y);ctx.stroke()}}
      ctx.fillStyle='#ef4444';ctx.font='bold 16px system-ui';ctx.fillText('PRZEBICIE: JONIZACJA ATOMÓW I TRANSPORT UWOLNIONYCH ELEKTRONÓW',W/2,H-85);
    }else{
      ctx.fillStyle=zero?'#facc15':'#f59e0b';ctx.font='bold 16px system-ui';ctx.fillText(zero?'BRAK POLA: BRAK POLARYZACJI I PRZEPŁYWU NETTO':'POLARYZACJA ŁADUNKÓW ZWIĄZANYCH — BRAK SWOBODNEGO PRZEPŁYWU',W/2,H-85);
    }
  }
  requestAnimationFrame(frame)
}requestAnimationFrame(frame)
</script>
'''.replace("__CONFIG__", config)
    components.html(html, height=700, scrolling=False)


st.title("⚡ Co dzieje się w przewodniku i izolatorze?")
st.markdown("#### Animacja mikroskopowa: ruch cieplny i dryf w metalu oraz polaryzacja, jonizacja i przebicie w izolatorze.")

with st.sidebar:
    material_name=st.selectbox("Materiał",list(MATERIALS)); material=MATERIALS[material_name]
    voltage=st.slider("Napięcie U [V]",-12.,12.,0.,.1) if material["kind"]=="conductor" else st.slider("Napięcie U [V]",-25000.,25000.,0.,100.)
    length=st.slider("Długość [m]",.1,2.,1.,.05)
    area=st.slider("Przekrój [mm²]",.25,10.,1.5,.25)
    temperature=st.slider("Temperatura [°C]",0,120,20,disabled=material["kind"]!="conductor")
    softness=st.slider("Łagodność progu przebicia",.004,.04,.012,.002,disabled=material["kind"]!="insulator")
    speed=st.slider("Szybkość animacji",.25,2.,1.,.25)
    st.warning("To tylko model. Nie wykonuj doświadczeń wysokonapięciowych poza odpowiednio wyposażonym laboratorium.")

if material["kind"]=="conductor":
    current,resistance=conductor_current(voltage,length,area,material["rho"],temperature)
    activation=0.; breakdown_voltage=None
    state="Brak prądu netto" if abs(voltage)<1e-9 else "Przewodzenie omowe"
else:
    breakdown_voltage=material["breakdown_voltage"]*length
    leakage_r=material["leakage_r"]*length/area
    post_r=material["post_breakdown_r"]*length/area
    current,activation=insulator_current(voltage,leakage_r,breakdown_voltage,post_r,softness)
    resistance=abs(voltage/current) if abs(current)>1e-18 else np.inf
    state="Brak polaryzacji" if abs(voltage)<1e-9 else ("Izolacja / polaryzacja" if activation<.1 else ("Rozwój jonizacji" if activation<.8 else "Przebicie"))

m1,m2,m3,m4=st.columns(4)
m1.metric("Napięcie U",f"{voltage:,.1f} V");m2.metric("Prąd I",nice_current(current));m3.metric("Opór U/I","nieokreślony przy U=0" if abs(voltage)<1e-9 else f"{resistance:.3e} Ω");m4.metric("Stan",state)

left,right=st.columns([5,1],gap="large")
with left:
    st.subheader("Widok mikroskopowy materiału")
    microscopic_animation(material["kind"],voltage,current,activation,material["color"],speed)
with right:
    st.subheader("Jak czytać widok?")
    if material["kind"]=="insulator":
        st.markdown("- Przed przebiciem widoczna jest **polaryzacja** związanych elektronów.\n- W środkowym rzędzie rozwija się  ścieżka jonizacji.\n- Pomarańczowe atomy oznaczają atomy, które utraciły elektron.\n- Niebieskie elektrony przemieszczają się między kolejnymi miejscami.\n- Obserwujemy transport nośników ładunku elektrycznego.")
    else:
        st.markdown("- Każdy elektron ma wektor chwilowej prędkości.\n- Przy 0 V ruch jest losowy i prąd netto wynosi zero.\n- Napięcie dodaje dodatkową składową dryfu.")
    st.metric("|E|",f"{abs(voltage)/length:.3e} V/m");st.metric("P=U·I",f"{voltage*current:.3e} W")

st.subheader("Charakterystyka prąd vs. napięcie, (I–U)")
if material["kind"]=="conductor":
    uu=np.linspace(-12,12,401); ii=[conductor_current(x,length,area,material["rho"],temperature)[0] for x in uu]
else:
    uu=np.linspace(-25000,25000,801); ii=[insulator_current(x,leakage_r,breakdown_voltage,post_r,softness)[0] for x in uu]
fig=go.Figure(go.Scatter(x=uu,y=ii,line=dict(color="#38bdf8",width=4),name="I(U)"))
fig.add_trace(go.Scatter(x=[voltage],y=[current],mode="markers",marker=dict(size=13,color="#facc15"),name="Punkt pracy"))
if material["kind"]=="insulator":
    fig.add_vline(x=breakdown_voltage,line_dash="dash",line_color="#ef4444")
    fig.add_vline(x=-breakdown_voltage,line_dash="dash",line_color="#ef4444")
fig.update_layout(height=420,paper_bgcolor="#07111f",plot_bgcolor="#07111f",font=dict(color="#e5e7eb"),xaxis_title="U [V]",yaxis_title="I [A]",margin=dict(l=10,r=10,t=20,b=10))
st.plotly_chart(fig,width='stretch',config={"displaylogo":False})

with st.expander("Interpretacja przebicia"):
    st.markdown("Pomarańczowy jon pozostaje w sieci, a uwolniony elektron przemieszcza się pod wpływem pola i może uczestniczyć w dalszej jonizacji. W rzeczywistym materiale możemy mieć wiele ścieżek jonizacji. Możliwe są rozgałęzienia i powstawanie nowych ścieżek jonizacji.")
